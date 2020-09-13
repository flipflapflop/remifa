/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "convert.cuh"
#include "lu_nopiv.cuh"

#include <iostream>

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace /* anon */ {

   // Dynamically allocated shared memory
   extern __shared__ char SharedMemory[];

   // Load diagonal block as well as block bx into shared memory
   // workspace
   // Note: we assume that m > n
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_block_load(
         std::int64_t bx, // Block row index
         std::int64_t m, // Number of rows in matrix
         std::int64_t n, // Number of columns in matrix
         T const *const d, // Workspace with copy of the digonal tile
         std::int64_t ldd, // Workspace leading dimension
         T *const a, // Matrix block column pointer
         std::int64_t lda, // Matrix leading dimensions
         T *const sdata // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         ) {

      std::int64_t tx = threadIdx.x;
      std::int64_t ty = threadIdx.y;

      std::int64_t ld_sdata = (TILES+1)*TILE_SIZE;

      // Load diagonal block A_kk
      sdata[tx + ty*ld_sdata] =
         ( (tx < n) && (ty < n)) ? // Note that m > n
         d[tx + ty*ldd] : (T) 0.0;
         // a[tx + ty*lda] : (T) 0.0;

      // Load off-diag blocks A_ik
      for (std::int64_t r=0; r<TILES; ++r) {
         std::int64_t a_x = tx + r*TILE_SIZE + TILES*TILE_SIZE*bx; // Row index in a
         if (a_x < n) continue; // Loop if we're in the diag block
         std::int64_t sdata_x = tx + (r+1)*TILE_SIZE; // Row index in sdata
         sdata[sdata_x + ty*ld_sdata] =
            ( (a_x < m) && (ty < n)) ?
            a[a_x + ty*lda] : (T) 0.0;
      }

   }

   // Store sub-diagonal block in shared memory into block bx in a
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_block_store(
         std::int64_t bx, // Block row index
         std::int64_t m, // Number of rows in matrix
         std::int64_t n, // Number of columns in matrix
         T *const sdata, // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         T *const a, // Data pointer
         std::int64_t lda // Input matrix leading dimensions
         ) {

      std::int64_t tx = threadIdx.x;
      std::int64_t ty = threadIdx.y;

      std::int64_t ld_sdata = (TILES+1)*TILE_SIZE;

      // Store diagonal block (block idx 0 only)
      if (bx == 0) {
         // printf("[dev_block_store] store diagonal\n");         
         if ( (tx < n) && (ty < n) ) {
            a[tx + ty*lda] = sdata[tx + ty*ld_sdata];
            // d[tx + ty*lda] = sdata[tx + ty*ld_sdata];
         }
      }
      
      // Store off-diag block A_ik
      for (std::int64_t r=0; r<TILES; ++r) {

         std::int64_t a_x = tx + r*TILE_SIZE + TILES*TILE_SIZE*bx; // Row index in a
         if (a_x < n) continue; // Loop if we're in the diag block
         std::int64_t sdata_x = tx + (r+1)*TILE_SIZE; // Row index in sdata
         if ((a_x < m) && (ty < n))
            a[a_x + ty*lda] = sdata[sdata_x + ty*ld_sdata];
      }  
   }

   // Note: assume that m > n and n <= TILE_SIZE 
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_lu_nopiv_block(
         unsigned int bx,
         int m, int n,
         T const *const d, int ldd,
         T *const l, int ldl,
         int *const stat // Info parameter
         ) {
      
      // Contains TILES tiles i.e. dimensions (TILES*TILE_SIZE,TILE_SIZE)
      T *swork = reinterpret_cast<T*>(SharedMemory); 
      int ld_swork = (TILES+1)*TILE_SIZE;

      dev_block_load<T, TILE_SIZE, TILES>(bx, m, n, d, ldd, l, ldl, swork);
      __syncthreads();

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      for (int k = 0; k < n; ++k) {

         T d11 = swork[k+ld_swork*k];
         // if ( fabs(d11) <  1e-20 ) {
         //    if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //       printf("[dev_lu_nopiv_block] Zero pivot detected, d11 = %.3e\n", d11);
         //       stat[0] = k;
         //    }
         //    return;
         // }

         // if ((bx == 0) && (ty == 0) && (tx == 0))
            // printf("[dev_lu_nopiv_block] d11 = %f\n", d11);
            // printf("[dev_lu_nopiv_block] bx = %d, tx = %d, ty = %d, d11 = %f\n", bx, tx, ty, d11);
         
         d11 = (T)1.0/d11;
         __syncthreads();

         // Scale subdiag rows
         if (ty == k) {
            // Diagonal tile
            if (tx > k && tx < n) {
               // printf("[dev_lu_nopiv_block] tx = %d, ty = %d, aik = %f\n", tx, ty, swork[tx + k*ld_swork]);
               swork[tx + k*ld_swork] *= d11;
            }
            // Subdiagonal tiles
            for (int r=0; r<TILES; ++r) {
               int sdata_x = tx + (r+1)*TILE_SIZE;
               swork[sdata_x + k*ld_swork] *= d11;
            }
         }
         __syncthreads();

         // Update trailing submatrix
         if (ty > k && ty < n) {
            // Update A_kk
            if (tx > k)
               swork[tx + ty*ld_swork] -= swork[tx + k*ld_swork]*swork[k + ty*ld_swork];
            
            // Update A_ik
            for (int r=0; r<TILES; ++r) {
               int sdata_x = tx + (r+1)*TILE_SIZE; // Row index in sdata
               swork[sdata_x + ty*ld_swork] -= swork[sdata_x + k*ld_swork]*swork[k + ty*ld_swork];
            }
         }
         __syncthreads();

      }

      // We manage to eliminate all columns
      if ((bx == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
         stat[0] = n;

      // Store W into A (A_ik)
      dev_block_store<T, TILE_SIZE, TILES>(bx, m, n, swork, l, ldl);
      __syncthreads();

   }

   // Store right-diagonal block in shared memory into a
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_br_block_store(
         std::int64_t bx, // Block row index
         std::int64_t m, // Number of rows in matrix
         std::int64_t n, // Number of columns in matrix
         T *const sdata, // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         T *const a, // Data pointer
         std::int64_t lda // Input matrix leading dimensions
         ) {

      std::int64_t tx = threadIdx.x;
      std::int64_t ty = threadIdx.y;

      std::int64_t ld_sdata = n;
      
      // Store off-diag block A_ik
      for (std::int64_t r=0; r<TILES; ++r) {

         std::int64_t a_y = ty + r*TILE_SIZE + TILES*TILE_SIZE*bx; // Row index in a
         if (a_y < n) continue; // Loop if we're in the diag block
         std::int64_t sdata_y = ty + (r+1)*TILE_SIZE; // Row index in sdata
         if ((a_y < m) && (tx < n))
            a[tx + a_y*lda] = sdata[tx + sdata_y*ld_sdata];
      }  
   }

   // Load block from matrix block row of size n*m
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_br_block_load(
         std::int64_t bx, // Block row index
         std::int64_t m, // Number of rows in matrix
         std::int64_t n, // Number of columns in matrix
         T const *const d, // Workspace with copy of the digonal tile
         std::int64_t ldd, // Workspace leading dimension
         T *const a, // Matrix block column pointer
         std::int64_t lda, // Matrix leading dimensions
         T *const sdata // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         ) {

      std::int64_t tx = threadIdx.x;
      std::int64_t ty = threadIdx.y;

      std::int64_t ld_sdata = n;

      // Load diagonal block A_kk
      sdata[tx + ty*ld_sdata] =
         ( (tx < n) && (ty < n)) ? // Note that m > n
         d[tx + ty*ldd] : (T) 0.0;
      // a[tx + ty*lda] : (T) 0.0;

      // Load off-diag blocks A_ik
      for (std::int64_t r=0; r<TILES; ++r) {
         std::int64_t a_y = ty + r*TILE_SIZE + TILES*TILE_SIZE*bx; // Row index in a
         if (a_y < n) continue; // Loop if we're in the diag block
         std::int64_t sdata_y = ty + (r+1)*TILE_SIZE; // Row index in sdata
         sdata[tx + sdata_y*ld_sdata] =
            ( (a_y < m) && (tx < n)) ?
            a[tx + a_y*lda] : (T) 0.0;
      }

   }

   // Note: assume that m > n and n <= TILE_SIZE 
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __device__ void
   dev_lu_nopiv_br_block(
         unsigned int bx,
         int m, int n,
         T const *const d, int ldd,
         T *const l, int ldl,
         int *const stat // Info parameter
         ) {
      
      // Contains TILES tiles i.e. dimensions (TILES*TILE_SIZE,TILE_SIZE)
      T *swork = reinterpret_cast<T*>(SharedMemory); 
      int ld_swork = n;

      dev_br_block_load<T, TILE_SIZE, TILES>(bx, m, n, d, ldd, l, ldl, swork);
      __syncthreads();

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      for (int k = 0; k < n; ++k) {

         T d11 = swork[k+ld_swork*k];
         // if ( fabs(d11) <  1e-20 ) {
         //    if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //       printf("[dev_lu_nopiv_block] Zero pivot detected, d11 = %.3e\n", d11);
         //       stat[0] = k;
         //    }
         //    return;
         // }

         // if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //    printf("[dev_lu_nopiv_br_block] d11 = %f\n", d11);
         // }
         
         d11 = (T)1.0/d11;
         __syncthreads();

         // Scale subdiag rows
         if (ty == k) {
            // Diagonal tile
            if (tx > k && tx < n) {
               // printf("[dev_lu_nopiv_br_block] tx = %d, ty = %d, aik = %f\n", tx, ty, swork[tx + k*ld_swork]);
               swork[tx + k*ld_swork] *= d11;
            }
         }
         __syncthreads();

         // Update trailing submatrix
         if (tx > k) {
            // Update A_kk
            if (ty > k)
               swork[tx + ty*ld_swork] -= swork[tx + k*ld_swork]*swork[k + ty*ld_swork];
            
            // Update A_kj
            for (int r=0; r<TILES; ++r) {
               int sdata_y = ty + (r+1)*TILE_SIZE; // Row index in sdata
               swork[tx + sdata_y*ld_swork] -= swork[tx + k*ld_swork]*swork[k + sdata_y*ld_swork];
            }
         }
         __syncthreads();

      }

      // We manage to eliminate all columns
      if ((bx == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
         stat[0] = n;

      // Store W into A (A_ik)
      dev_br_block_store<T, TILE_SIZE, TILES>(bx, m, n, swork, l, ldl);
      __syncthreads();

   }

   // Perform the unpivoted LU factorization of a block-column m x n
   // with m >= n and n <= TILE_SIZE.   
   template<typename T,
            int TILE_SIZE,
            int TILES>
   __global__ void
   dev_lu_nopiv_panel(
         std::int64_t m, std::int64_t n,
         T const *const d, std::int64_t ldd,
         T *const l, std::int64_t ldl,
         T *const u, std::int64_t ldu,
         int *const stat // Info parameter
         ) {

      unsigned int bx = blockIdx.x;
      unsigned int by = blockIdx.y;
      // printf("[dev_lu_nopiv_panel] bx = %d, by = %d\n", bx, by);

      if (by==0) {
         dev_lu_nopiv_block<T, TILE_SIZE, TILES>(
               bx, m, n,
               d, ldd,
               l, ldl, stat);
      }
      else if (m > n) {
         dev_lu_nopiv_br_block<T, TILE_SIZE, TILES>(
               bx, m, n,
               d, ldd,
               u, ldu, stat);
      }
   }

}

namespace remifa {

   template<typename T, int ib>
   void lu_nopiv_panel(
         const cudaStream_t stream,
         int m, int n,
         T const *const d, int ldd,
         T *l, int ldl,
         T *u, int ldu,
         int *stat) {

      dim3 threads(ib, ib);
      // int nblocks = (std::max((m - n - 1), 0) / (BLOCKS*BLOCK_SIZE)) + 1; 
      // int nblocks = (std::max((m - n - 1), 0) / (BLOCKS*BLOCK_SIZE)) + 2; 
      // int nblocks = (std::max((m - n), 0) / (BLOCKS*BLOCK_SIZE)) + 1; 
      // int nblocks = (std::max((m - n), 0) / (BLOCKS*BLOCK_SIZE)) + 1; 
      int nblocks = (m / (BLOCKS*ib)) + 1; 

      // std::cout << "[lu_nopiv_panel] nblocks = " << nblocks << ", m = " << m << ", n = " << n << std::endl; 
      // Thread blocks with idx (:,0) for the block column and thread
      // blocks with idx (:,1) for block row
      dim3 grid(nblocks, 2);
      // dim3 grid(nblocks, 1);
      
      // Calculate the size of the shared memory workspace per thread
      // blocks
      // size_t smsize = 0;
      // BLOCKS tiles per thread blocks plus diagonal tile
      size_t smsize = (BLOCKS+1)*ib*ib*sizeof(T);
      
      dev_lu_nopiv_panel
         <T, ib, BLOCKS>
         <<<grid, threads, smsize, stream>>>
         (m, n,
          d, ldd,
          l, ldl,
          u, ldu,
          stat);
   }

   // Half-precision
   template void lu_nopiv_panel<half>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<half,1>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<half,2>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<half,4>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<half,16>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<half,32>(
         const cudaStream_t stream, int m, int n,
         half const *const d, int ldd,
         half *l, int ldl, half *u, int ldu, 
         int *stat);
   // Single precision
   template void lu_nopiv_panel<float>(
         const cudaStream_t stream, int m, int n,
         float const *const d, int ldd,
         float *l, int ldl, float *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<float,1>(
         const cudaStream_t stream, int m, int n,
         float const *const d, int ldd,
         float *l, int ldl, float *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<float,2>(
         const cudaStream_t stream, int m, int n,
         float const *const d, int ldd,
         float *l, int ldl, float *u, int ldu, 
         int *stat);
   template void lu_nopiv_panel<float,4>(
         const cudaStream_t stream, int m, int n,
         float const *const d, int ldd,
         float *l, int ldl, float *u, int ldu, 
         int *stat);
   // Double precision
   template void lu_nopiv_panel<double>(
         const cudaStream_t stream, int m, int n,
         double const *const d, int ldd,
         double *l, int ldl, double *u, int ldu,
         int *stat);

}
