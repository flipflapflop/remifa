/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "convert.cuh"
#include "errchk.hxx"
#include "gemm.cuh"
#include "lu_nopiv.cuh"
#include "utils.hxx"
#include "wrappers.hxx"

#include <cassert>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
#endif

#define BLOCKS 1 // Number of tiles per Thread blocks
#define BLOCK_SIZE 8 // Thread block size
#define OUTER_BLOCK_SIZE 256

namespace remifa {

   // LU factorization left-looking algoithm using one level of
   // blocking.
   template<
      typename T, // Working precision
      remifa::compute_type U, // Update compute type
      std::int64_t ib=BLOCK_SIZE, /*int nb=OUTER_BLOCK_SIZE,*/
      typename P=T, // Panel prec (to accumulate the update)
      typename F=T, // Factor prec
      typename GemmImpl=remifa::CublasImpl,
      bool verbose = false,
      typename CpyType = half
      >
   void lu_nopiv_inner_ll(
         const cublasHandle_t cuhandle, 
         std::int64_t m, // Number of rows 
         std::int64_t n, // Number of columns
         // int nb,
         T *const d_l, // Matrix pointer on device 
         std::int64_t ld_l, // Matrix leadind dim on device
         T *const d_u, // Matrix pointer on device 
         std::int64_t ld_u, // Matrix leadind dim on device
         F *work, // Workspace
         int *d_info, // Info device
         int slices=0, uint8_t *workspace=nullptr,
         CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
         std::int64_t ld_a_cpy=-1
         ) {

      // Error handling
      std::string context = "lu_nopiv_inner_ll";

      if (verbose) {
         std::cout << "[" << context << "]"
                   << " W=" << type_name<T>()
                   << ", IB=" << ib
                   << ", I=" << compute_type_name<U>()
                   << ", IP=" << type_name<P>()
                   << ", IF=" << type_name<F>()
                   << std::endl;
      }

      // Check arguments
      if (m <= 0) {
         throw std::invalid_argument("Given number of rows invalid: m = " + std::to_string(m));
      }
      else if (n <= 0) {
         throw std::invalid_argument("Given number of columns invalid: n = " + std::to_string(n));
      }
      
      if (std::is_same<T, float>::value) {
         // W=fp32

         // Check compute type
         if (compute_type::TC32==U) {
            // U=TC32
            if (d_a_cpy == nullptr) {
               throw std::invalid_argument("A fp16 buffer should be provided to this routine when W=fp32 and U=TC32");
            }
            else if (ld_a_cpy <= 0) {
               throw std::invalid_argument("Invalid leading dimension for matrix copy buffer: ld_a_cpy = " + std::to_string(ld_a_cpy));
            }
         }
         else if (compute_type::TC16==U) {
            throw std::invalid_argument("TC16 compute type unavailable with W=fp32");
         }

         // Check panel type
         if (!std::is_same<P, float>::value) {
            throw std::invalid_argument("Only fp32 factor type available with W=fp32");
         }

         // Check factor type
         if (!std::is_same<F, float>::value) {
            throw std::invalid_argument("Only fp32 factor type available with W=fp32");
         }
      }
      else if (std::is_same<T, half>::value) {
         if (compute_type::FP32==U) {
            throw std::invalid_argument("FP32 compute type unavailable with W=fp16");
         }
      }
         
      // using GemmImpl = typename std::conditional<use_cutlass, remifa::CutlassImpl, remifa::CublasImpl>::type;
      using GemmOp = typename std::conditional<
         (remifa::compute_type::TC16==U) || (remifa::compute_type::TC32==U) , remifa::TC, remifa::FP>::type;
      using QueueType = typename std::conditional
         <std::is_same<GemmImpl, remifa::CublasImpl>::value,
         cublasHandle_t, cudaStream_t>::type;
      // using QueueType = typename std::conditional<use_cutlass, cudaStream_t, cublasHandle_t>::type;
      using AccType = typename std::conditional<
         (remifa::compute_type::TC16==U) || (remifa::compute_type::FP16==U) , half, float>::type;
      
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status

      cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      remifa::cublas_check_error(custat, context);

      QueueType queue;
      if (std::is_same<QueueType, cudaStream_t>::value) {
         queue = (QueueType) stream;
      }
      else if (std::is_same<QueueType, cublasHandle_t>::value) {
         queue = (QueueType) cuhandle;
      }
      else {
         throw std::invalid_argument("Type QueueType not recognized");
      }

      // Workspace
      F *d_d = nullptr;
      std::int64_t lddd = ib;
      assert(nullptr != work);
      d_d = work;
      
      // Output buffer for computing L factor update
      P* d_l_out = nullptr;
      std::int64_t ld_l_out = -1;
      // Output buffer for computing U factor update
      P* d_u_out = nullptr;
      std::int64_t ld_u_out = -1;
      
      if (std::is_same<T, P>::value && std::is_same<T, F>::value) {
         // P and F are equal to the working precision, therefore
         // there is no need for a buffer
         
         d_l_out = reinterpret_cast<P*>(d_l);
         ld_l_out = ld_l;
         d_u_out = reinterpret_cast<P*>(d_u);
         ld_u_out = ld_u;
      }
      else {
         // The panel precision is different from the working
         // precision. We allocate buffer for computing the panel in
         // the requested precision OP.

         // TODO: allow for supling this routine with two wokspaces
         // for l_out and u_out to avoid memeory reallocation.

         if (verbose) {
            std::cout << "[" << context << "] " << "Allocate L and U buffers" << std::endl;
         }
         
         ld_l_out = ld_l;
         cuerr = cudaMalloc((void**)&d_l_out, ld_l_out*ib*sizeof(P));      
         remifa::cuda_check_error(cuerr, context);      
         ld_u_out = ib;
         cuerr = cudaMalloc((void**)&d_u_out, ld_u_out*m*sizeof(P));      
         remifa::cuda_check_error(cuerr, context);      
      }

      if (!std::is_same<T, F>::value) {
         // If factor precision is different from working precision,
         // then we need to copy and convert matrix entries from the
         // first block-column/block-row into buffer

         if (verbose) {
            std::cout << "[" << context << "] " << "Copy first panel of A into buffers" << std::endl;
         }

         std::int64_t updm = m;
         std::int64_t in = std::min(n, ib);         
         remifa::convert(stream, updm, in, d_l, ld_l, d_l_out, ld_l_out);
         remifa::convert(stream, in, updm, d_u, ld_u, d_u_out, ld_u_out);
      }
      
      // Number of block-columns
      std::int64_t inc = (n-1) / ib + 1; 

      // Factor outer block
      for (std::int64_t k = 0; k < inc; ++k) {
         
         // std::cout << "[" << context << "] k = " << k << std::endl;
         
         // Factor kth block column
         std::int64_t iofs = k*ib; // Number of eliminated columns
         std::int64_t cblkm = m-iofs; // Block column height
         std::int64_t cblkn = std::min(n-iofs, ib); // Block column width
         // std::cout << "[" << context << "] cblkm = " << cblkm << ", cblkn = " << cblkn << std::endl;
         std::int64_t iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            
         if (iofs>0) {

            if (std::is_same<T, P>::value) {
               d_l_out = reinterpret_cast<P*>(&d_l[iofs + iofs*ld_l]);
               ld_l_out = ld_l;
               d_u_out =  reinterpret_cast<P*>(&d_u[iofs + iofs*ld_u]);
               ld_u_out = ld_u;
            }
            else {
               remifa::convert(stream, cblkm, cblkn, &d_l[iofs+iofs*ld_l], ld_l, d_l_out, ld_l_out);
               if (cblkm>cblkn) {
                  remifa::convert(stream, cblkn, cblkm, &d_u[iofs+iofs*ld_u], ld_u, d_u_out, ld_u_out);
               }
            }

            // Update L inner panel

            if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {
               // std::cout << "[" << context << "] TETETETTETE L" << std::endl;
               gemmex<GemmImpl, GemmOp, AccType>
                  (queue, cblkm, cblkn, iofs,
                   AccType(-1.0),
                   (half*)&d_a_cpy[iofs         ], ld_a_cpy,
                   (half*)&d_a_cpy[iofs*ld_a_cpy], ld_a_cpy,
                   AccType(1.0),
                   d_l_out,  ld_l_out,
                   slices, workspace);
            }
            else {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue, cblkm, cblkn, iofs,
                   AccType(-1.0),
                   &d_l[iofs     ], ld_l,
                   &d_u[iofs*ld_u], ld_u,
                   AccType(1.0),
                   d_l_out,  ld_l_out,
                   slices, workspace);
            }

            // Update U inner panel
            if (cblkm>cblkn) {
               // Update U inner panel

               // std::cout << "[" << context << "] Update U, k = " << k << std::endl;
               // std::cout << "[" << context << "] cblkm = " << cblkm << ", cblkn = " << cblkn << std::endl;

               if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {
                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue,
                      cblkn, cblkm-cblkn, iofs,
                      AccType(-1.0),
                      &d_a_cpy[iofs      ], ld_a_cpy,
                      &d_a_cpy[iofst*ld_a_cpy], ld_a_cpy,
                      AccType(1.0),
                      &d_u_out[cblkn*ld_u_out], ld_u_out,
                      slices, workspace);
               }
               else {

                  // std::cout << "[" << context << "]"
                  //           << ", &d_l[iofs      ] = " << &d_l[iofs      ]
                  //           << ", &d_u[iofst*ld_u] = " << &d_u[iofst*ld_u]
                  //           << ", &d_u_out[cblkn*ld_u_out] = " << &d_u_out[cblkn*ld_u_out]
                  //           << std::endl;

                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue,
                      cblkn, cblkm-cblkn, iofs,
                      AccType(-1.0),
                      &d_l[iofs      ], ld_l,
                      &d_u[iofst*ld_u], ld_u,
                      AccType(1.0),
                      &d_u_out[cblkn*ld_u_out], ld_u_out,
                      slices, workspace);

                  // gemmex<GemmImpl, GemmOp, AccType>
                  //    (queue,
                  //     cblkn, cblkm, iofs,
                  //     AccType(-1.0),
                  //     &d_l[iofs     ], ld_l,
                  //     &d_u[iofs*ld_u], ld_u,
                  //     AccType(1.0),
                  //     d_u_out, ld_u_out,
                  //     slices, workspace);

               }
               
               if ((!std::is_same<T, P>::value) && (std::is_same<T, F>::value)) {
                  // remifa::convert(stream, cblkn, cblkm-cblkn, &d_u_out[cblkn*ld_u_out], ld_u_out, &d_u[iofs+iofst*ld_u], ld_u);
                  remifa::convert(stream, cblkn, cblkm, d_u_out, ld_u_out, &d_u[iofs+iofs*ld_u], ld_u);
               }

            } // cblkm>cblkn    
            
            if ((!std::is_same<T, P>::value)  && (std::is_same<T, F>::value)) {
               remifa::convert(stream, cblkm, cblkn, d_l_out, ld_l_out, &d_l[iofs+iofs*ld_l], ld_l);
            }

         } // iofs>0

         // If factorization prec is the same as the working prec,
         // then we factor the outer block directly into the input
         // matrix, otherwise we perform the factorization into the panel buffer

         if (std::is_same<T, F>::value) {

            cudaMemcpy2DAsync(
                  reinterpret_cast<T*>(d_d), lddd*sizeof(T),
                  &d_l[iofs+iofs*ld_l], ld_l*sizeof(T),
                  cblkn*sizeof(T), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);

            // std::cout << "[" << context << "] cblkm = " << cblkm << ", cblkn = " << cblkn << std::endl;

            lu_nopiv_panel
               <T, ib>
               (stream, cblkm, cblkn,
                reinterpret_cast<T*>(d_d), lddd,
                &d_l[iofs+iofs*ld_l], ld_l,
                &d_u[iofs+iofs*ld_u], ld_u,
                d_info);


         }
         else {

            cudaMemcpy2DAsync(
                  (F*)d_d, lddd*sizeof(F),
                  (F*)d_l_out, ld_l_out*sizeof(F),
                  cblkn*sizeof(F), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);

            lu_nopiv_panel
               <F, ib>
               (stream, cblkm, cblkn,
                (F*) d_d, lddd,
                (F*) d_l_out, ld_l_out,
                (F*) d_u_out, ld_u_out,
                d_info);

            // Factor precision F is different from working precision
            // W: copy and convert computed factor into A.
            
            // remifa::convert(stream, cblkn, cblkm, d_u_out, ld_u_out, &d_u[iofs+iofs*ld_u], ld_u);

            if (cblkm>cblkn) {
               remifa::convert(stream, cblkn, cblkm-cblkn, &d_u_out[cblkn*ld_u_out], ld_u_out, &d_u[iofs+iofst*ld_u], ld_u);
            }
            remifa::convert(stream, cblkm, cblkn, d_l_out, ld_l_out, &d_l[iofs+iofs*ld_l], ld_l);
         } // T=F

         if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {
            // Copy of L and U factors into buffers
            if (cblkm>cblkn) {
               remifa::convert(stream, cblkn, cblkm-cblkn, &d_u[iofs+iofst*ld_u], ld_u, &d_a_cpy[iofs+iofst*ld_a_cpy], ld_a_cpy);
            }
            remifa::convert(stream, cblkm, cblkn, &d_l[iofs+iofs *ld_l], ld_l, &d_a_cpy[iofs+iofs *ld_a_cpy], ld_a_cpy);
         }
         
      } // Inner blocks loop

   }

}
