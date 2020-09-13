/// @file
/// @copyright 2020- 
/// @author Florent Lopez
#include "convert.cuh"
//STD
#include <cassert>
#include <iostream>
// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#endif

// MAGMA see magmablas/hlaconvert.cu
const int max_blocks = 65535;

// MAGMA see magmablas/hlaconvert.cu
// #define BLK_X 32
#define BLK_X 64
// #define BLK_X 128
// #define BLK_X 256
#define BLK_Y BLK_X

namespace remifa {

   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_dp2hp_device(
         std::int64_t m, std::int64_t n,
         double const* dA, std::int64_t ldda,
         half *dB, std::int64_t lddb )
   {
      std::int64_t ind = blockIdx.x*BLK_X + threadIdx.x;
      std::int64_t iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);
      /* do only rows inside matrix */
      if ( ind < m ) {
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         if ( full ) {
            // full block-column
#pragma unroll
            for( std::int64_t j=0; j < BLK_Y; ++j ) {
               // dB[j*lddb] = __float2half( dA[j*ldda] );
               dB[j*lddb] =  dA[j*ldda] ;
               // if (__hisnan(dB[j*lddb]) || __hisnan(dB[j*lddb])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
         else {
            // partial block-column
            for( std::int64_t j=0; j < BLK_Y && iby+j < n; ++j ) {
               // dB[j*lddb] = __float2half( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda];
               // if (__hisnan(dB[j*lddb]) || __hisnan(dB[j*lddb])) {
               //    printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
      }
   }


   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_sp2hp_device(
         std::int64_t m, std::int64_t n,
         const float  *dA, std::int64_t ldda,
         half *dB, std::int64_t lddb )
   {
      std::int64_t ind = blockIdx.x*BLK_X + threadIdx.x;
      std::int64_t iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);

      /* do only rows inside matrix */
      if ( ind < m ) {
         // std::int64_t dpl_a = ind + iby*ldda;
         // std::int64_t dpl_b = ind + iby*lddb;
         // if (dpl_a < 0 || dpl_b < 0) {
         //    printf("[convert_sp2hp_device] Error detected\n");
         // }
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         // dA += dpl_a;
         // dB += dpl_b;
         
         // double e = 0.0;

         if ( full ) {
            // full block-column
#pragma unroll
            for( std::int64_t j=0; j < BLK_Y; ++j ) {
               // dB[j*lddb] = __float2half( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda] ;

               // dB[j*lddb] = (__half) cutlass::NumericConverter
               //    <__half, float,
               //     // cutlass::FloatRoundStyle::round_toward_zero
               //     cutlass::FloatRoundStyle::round_to_nearest
               //     // cutlass::FloatRoundStyle::round_toward_infinity
               //     >::convert((float)dA[j*ldda]);

               
               
               // if (__hisnan(dB[j*lddb]) || __hisnan(dB[j*lddb])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }

            }
         }
         else {
            // partial block-column
            for( std::int64_t j=0; j < BLK_Y && iby+j < n; ++j ) {
               // dB[j*lddb] = __float2half( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda];

               // dB[j*lddb] = (__half) cutlass::NumericConverter
               //    <__half, float,
               //     // cutlass::FloatRoundStyle::round_toward_zero
               //     cutlass::FloatRoundStyle::round_to_nearest
               //     // cutlass::FloatRoundStyle::round_toward_infinity
               //     >::convert((float)dA[j*ldda]);

               // if (__hisnan(dB[j*lddb]) || __hisnan(dB[j*lddb])) {
               //    printf("[convert_sp2hp_device] NaN detected\n");
               // }

            }
         }
      }
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_hp2sp_device(
         std::int64_t m, std::int64_t n,
         const half *dA, std::int64_t ldda,
         float  *dB, std::int64_t lddb )
   {
      std::int64_t ind = blockIdx.x*BLK_X + threadIdx.x;
      std::int64_t iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);
      /* do only rows inside matrix */
      if ( ind < m ) {
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         if ( full ) {
            // full block-column
#pragma unroll
            for( std::int64_t j=0; j < BLK_Y; ++j ) {

               // dB[j*lddb] = __half2float( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda];

               // B[j*lddb] = (float) cutlass::NumericConverter
               //    <float, cutlass::half_t,
               //     // cutlass::FloatRoundStyle::round_toward_zero
               //     cutlass::FloatRoundStyle::round_to_nearest
               //     // cutlass::FloatRoundStyle::round_toward_infinity
               //     >::convert((cutlass::half_t)dA[j*ldda]);

               
               // if (__hisnan(dA[j*ldda]) || __hisnan(dA[j*ldda])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
         else {

            // partial block-column
            for( std::int64_t j=0; j < BLK_Y && iby+j < n; ++j ) {
               // dB[j*lddb] = __half2float( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda];

               // dB[j*lddb] = (float) cutlass::NumericConverter
               //    <float, cutlass::half_t,
               //     // cutlass::FloatRoundStyle::round_toward_zero
               //     cutlass::FloatRoundStyle::round_to_nearest
               //     // cutlass::FloatRoundStyle::round_toward_infinity
               //     >::convert((cutlass::half_t)dA[j*ldda]);

               // if (__hisnan(dA[j*ldda]) || __hisnan(dA[j*ldda])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
      }
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_hp2dp_device(
         std::int64_t m, std::int64_t n,
         const half *dA, std::int64_t ldda,
         double *dB, std::int64_t lddb )
   {
      std::int64_t ind = blockIdx.x*BLK_X + threadIdx.x;
      std::int64_t iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);
      /* do only rows inside matrix */
      if ( ind < m ) {
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         if ( full ) {
            // full block-column
#pragma unroll
            for( std::int64_t j=0; j < BLK_Y; ++j ) {
               // dB[j*lddb] = __half2float( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda] ;
               // if (__hisnan(dA[j*ldda]) || __hisnan(dA[j*ldda])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
         else {
            // partial block-column
            for( std::int64_t j=0; j < BLK_Y && iby+j < n; ++j ) {
               // dB[j*lddb] = __half2float( dA[j*ldda] );
               dB[j*lddb] = dA[j*ldda];
               // if (__hisnan(dA[j*ldda]) || __hisnan(dA[j*ldda])) {
               //       printf("[convert_sp2hp_device] NaN detected\n");
               // }
            }
         }
      }
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_dp2hp_kernel(
         int m, int n,
         double const* dA, int ldda,
         half *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_dp2hp_device(m, n, dA, ldda, dB, lddb);
#endif
   }
   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_hp2dp_kernel(
         int m, int n,
         const half *dA, int ldda,
         double *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_hp2dp_device(m, n, dA, ldda, dB, lddb);
#endif
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_sp2hp_kernel(
         int m, int n,
         const float  *dA, int ldda,
         half *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
#endif
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_hp2sp_kernel(
         int m, int n,
         const half *dA, int ldda,
         float  *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_hp2sp_device(m, n, dA, ldda, dB, lddb);
#endif
   }

   template<typename TA, typename TAO> 
   __global__
   void convert_kernel(int m, int n, const TA  *dA, int ldda, TAO *dB, int lddb );

   // Template specialization
   template<>
   __global__
   void convert_kernel<double, half>(
         int m, int n,
         double const* dA, int ldda,
         half *dB, int lddb ) {
      convert_dp2hp_device(m, n, dA, ldda, dB, lddb);
   }
   template<>
   __global__
   void convert_kernel<float, half>(
         int m, int n,
         const float  *dA, int ldda,
         half *dB, int lddb ) {
      convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
   }
   template<>
   __global__
   void convert_kernel<half, double>(
         int m, int n,
         half const* dA, int ldda,
         double *dB, int lddb ) {
      convert_hp2dp_device(m, n, dA, ldda, dB, lddb);
   }
   template<>
   __global__
   void convert_kernel<half, float>(
         int m, int n,
         const half  *dA, int ldda,
         float *dB, int lddb ) {
      convert_hp2sp_device(m, n, dA, ldda, dB, lddb);
   }

   // @brief Convert matrix a of type float into half prec and put
   // result in aout
   template<typename TA, typename TAO> 
   void convert(
         cudaStream_t const stream,
         int m, int n,
         TA *const a, int lda, 
         TAO *const aout, int ldaout) {
            
      assert( BLK_X == BLK_Y );
      std::int64_t const super_NB = max_blocks*BLK_X;
      dim3 super_grid(
            (m + super_NB - 1) / super_NB, 
            (n + super_NB - 1) / super_NB);
    
      dim3 threads( BLK_X, 1 );
      dim3 grid;
      // std::cout << "super_grid.x = " << super_grid.x << std::endl;
      // std::cout << "super_grid.y = " << super_grid.y << std::endl;
      std::int64_t mm, nn;
      for( std::int64_t i=0; i < super_grid.x; ++i ) {
         mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
         grid.x = (mm + BLK_X - 1) / BLK_X;
         for( std::int64_t j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = (nn + BLK_Y - 1) / BLK_Y;

            // std::cout << "mm = " << mm << std::endl;
            // std::cout << "nn = " << nn << std::endl;
            // std::cout << "lda = " << lda << std::endl;
            // std::cout << "grid.x = " << grid.x << std::endl;
            // std::cout << "grid.y = " << grid.y << std::endl;

            convert_kernel 
               <<< grid, threads, 0, stream >>>
               (mm, nn, &a[i*super_NB + j*super_NB*lda], lda, &aout[i*super_NB + j*super_NB*ldaout], ldaout);
         }
      }
   }
   // FP32 to FP16
   template void convert<float, half>(
         cudaStream_t const stream, int m, int n, float *const a, int lda, 
         half *const aout, int ldaout);
   // FP64 to FP16
   template void convert<double, half>(
         cudaStream_t const stream, int m, int n, double *const a, int lda, 
         half *const aout, int ldaout);
   // FP16 to FP32
   template void convert<half, float>(
         cudaStream_t const stream, int m, int n, half *const a, int lda, 
         float *const aout, int ldaout);
   // FP16 to FP64
   template void convert<half, double>(
         cudaStream_t const stream, int m, int n, half *const a, int lda, 
         double *const aout, int ldaout);

   template<> void convert<half, half>(
         cudaStream_t const stream, int m, int n, half *const a, int lda, 
         half *const aout, int ldaout) {
      // FIXME: perform copy?
   }
   template<> void convert<float, float>(
         cudaStream_t const stream, int m, int n, float *const a, int lda, 
         float *const aout, int ldaout) {
      // FIXME: perform copy?
   }

   
} // End of namespace remifa
