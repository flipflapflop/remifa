#define BLOCKS 1 // Number of tiles per Thread blocks
#define BLOCK_SIZE 8 // Thread block size
#define OUTER_BLOCK_SIZE 256

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

namespace remifa {

   template<
      typename T, // Working precision
      remifa::compute_type U, // Update compute type
      std::int64_t ib=BLOCK_SIZE, /*int nb=OUTER_BLOCK_SIZE,*/
      typename GemmImpl=remifa::CublasImpl,
      // bool use_cutlass=false,
      bool verbose = false,
      typename CpyType = half
      >
   void lu_nopiv_inner_rl(
         const cublasHandle_t cuhandle, 
         std::int64_t m, // Number of rows 
         std::int64_t n, // Number of columns
         std::int64_t nb, // Outer block size
         T *const d_a, // Matrix pointer on device 
         std::int64_t ld_a, // Matrix leadind dim on device
         T *work, // Workspace
         int *d_info, // Info device
         CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
         std::int64_t ld_a_cpy=-1
         ) {

      // Error handling
      std::string context = "lu_nopiv_inner_rl";

      if (verbose) {
         std::cout << "[" << context << "]"
                   << " W=" << type_name<T>()
                   << ", I=" << compute_type_name<U>()
                   << std::endl;
      }

      // using GemmImpl = typename std::conditional<use_cutlass, remifa::CutlassImpl, remifa::CublasImpl>::type;
      using GemmOp = typename std::conditional<
         (remifa::compute_type::TC16==U) || (remifa::compute_type::TC32==U) , remifa::TC, remifa::FP>::type;
      using QueueType = typename std::conditional
         <std::is_same<GemmImpl, remifa::CublasImpl>::value,
         cublasHandle_t, cudaStream_t>::type;
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
      T *d_d = nullptr;
      std::int64_t lddd = ib;

      // if (nullptr == work) {
      //    cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(T));
      //    remifa::cuda_check_error(cuerr, context);
      // } else {
      assert(nullptr != work);
      d_d = work;
      // }

      T alpha = -1.0, beta = 1.0;
      float alpha_fp32 = -1.0, beta_fp32 = 1.0;

      std::int64_t inc = (n-1) / ib + 1; 
      std::int64_t ofst = nb;
         
      // Factor outer block
      for (std::int64_t k = 0; k < inc; ++k) {

         // std::cout << "[" << context << "] k = " << k << std::endl;
         // Factor kth block column
         std::int64_t iofs = k*ib; // Number of eliminated columns
         std::int64_t cblkm = m-iofs; // Block column height
         std::int64_t cblkn = std::min(n-iofs, ib); // Block column width

         // Trailing submatrix info
         std::int64_t iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
         std::int64_t tblkm = m-iofst; // Width of trailing submatrix in outer block
         std::int64_t tblkn = n-iofst; // Width of trailing submatrix in outer block

         cudaMemcpy2DAsync(
               d_d, lddd*sizeof(T),
               &d_a[iofs+iofs*ld_a], ld_a*sizeof(T),
               cblkn*sizeof(T), cblkn,
               cudaMemcpyDeviceToDevice,
               stream);

         lu_nopiv_panel(
               stream, cblkm, cblkn,
               d_d, lddd,
               &d_a[iofs+iofs*ld_a], ld_a,
               &d_a[iofs+iofs*ld_a], ld_a,
               d_info);

         if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {
            // Copy of L and U factors into buffers
            if (cblkm>cblkn) {
               remifa::convert(stream, cblkn, cblkm-cblkn, &d_a[iofs+iofst*ld_a], ld_a, &d_a_cpy[iofs+iofst*ld_a_cpy], ld_a_cpy);
            }
            remifa::convert(stream, cblkm, cblkn, &d_a[iofs+iofs *ld_a], ld_a, &d_a_cpy[iofs+iofs *ld_a_cpy], ld_a_cpy);
         }
         
         if (tblkn>0) {
            // Update L

            if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   tblkm, tblkn, ib,
                   AccType(-1.0),
                   &d_a_cpy[iofst+ iofs *ld_a_cpy], ld_a_cpy,
                   &d_a_cpy[iofs + iofst*ld_a_cpy], ld_a_cpy,
                   AccType(1.0),
                   &d_a[iofst+iofst*ld_a], ld_a);
            }
            else {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   tblkm, tblkn, ib,
                   AccType(-1.0),
                   &d_a[iofst+ iofs *ld_a], ld_a,
                   &d_a[iofs + iofst*ld_a], ld_a,
                   AccType(1.0),
                   &d_a[iofst+iofst*ld_a], ld_a);
            }            
            
            //
            // Update U
            if (tblkm>tblkn) {
               // std::cout << "[" << context << "] TETETTETET U" << std::endl;

               if (std::is_same<T, float>::value && (remifa::compute_type::TC32==U)) {

                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue,
                      tblkn, tblkm-tblkn, ib,
                      AccType(-1.0),
                      &d_a_cpy[iofst+ iofs*ld_a_cpy], ld_a_cpy,
                      &d_a_cpy[iofs + ofst*ld_a_cpy], ld_a_cpy,
                      AccType(1.0),
                      &d_a[iofst+ofst*ld_a], ld_a);

               }
               else {
                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue,
                      tblkn, tblkm-tblkn, ib,
                      AccType(-1.0),
                      &d_a[iofst+ iofs*ld_a], ld_a,
                      &d_a[iofs + ofst*ld_a], ld_a,
                      AccType(1.0),
                      &d_a[iofst+ofst*ld_a], ld_a);
               }

            }
         }
            
      } // Inner blocks loop
      
   }
   
} // End of namespace
