/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "convert.cuh"
#include "errchk.hxx"
#include "gemm.cuh"
#include "lu_nopiv.cuh"
#include "lu_nopiv_inner_rl.hxx"
#include "utils.hxx"
#include "wrappers.hxx"

#include <cassert>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
// #include "cutlass/layout/layout.h"
// // #include "cutlass/util/reference/device/gemm.h"
// #include "cutlass/gemm/device/gemm.h"
#endif

#define BLOCKS 1 // Number of tiles per Thread blocks
// #define BLOCKS 2 // Number of tiles per Thread blocks
// #define BLOCKS 3 // Number of tiles per Thread blocks
// #define BLOCKS 32 // Number of tiles per Thread blocks
// #define BLOCK_SIZE 2 // Thread block size
// #define BLOCK_SIZE 4 // Thread block size
#define BLOCK_SIZE 8 // Thread block size
// #define BLOCK_SIZE 16 // Thread block size
// #define OUTER_BLOCK_SIZE 16
// #define OUTER_BLOCK_SIZE 32
// #define OUTER_BLOCK_SIZE 64
// #define OUTER_BLOCK_SIZE 96
// #define OUTER_BLOCK_SIZE 128
// #define OUTER_BLOCK_SIZE 160
#define OUTER_BLOCK_SIZE 256
// #define OUTER_BLOCK_SIZE 384
// #define OUTER_BLOCK_SIZE 512
// #define OUTER_BLOCK_SIZE 768

namespace remifa {
   
   template<
      typename T, // Working prec
      remifa::compute_type I=FP32, // Inner update comupte type
      remifa::compute_type O=FP32, // Outer update comupte type
      int ib=BLOCK_SIZE,
      // int nb=OUTER_BLOCK_SIZE,
      bool use_cutlass=false,
      typename CpyType = half
      >
   void lu_nopiv_rl(
         const cublasHandle_t cuhandle, 
         std::int64_t m, // Number of rows 
         std::int64_t n, // Number of columns
         std::int64_t nb,
         T *const d_a, // Matrix pointer on device 
         std::int64_t ldda, // Matrix leadind dim on device
         int *d_info, // Info device
         CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
         std::int64_t ld_a_cpy=-1
         ) {

      // Error handling
      std::string context = "lu_nopiv_rl";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status

      // Number of (outer) block columns
      std::int64_t const nc = (n-1) / nb +1;

      std::cout << "[" << context << "]"
                << " W = " << type_name<T>()
                << ", I = " << compute_type_name<I>()
                << ", O = " << compute_type_name<O>()
                << ", Use cutlass = " << use_cutlass
                << std::endl;

      if (std::is_same<T, float>::value && (remifa::compute_type::TC32==O) &&
          d_a_cpy == nullptr) {
         throw std::invalid_argument("A fp16 buffer should be provided");
      }
      
      using GemmImpl = typename std::conditional<use_cutlass, remifa::CutlassImpl, remifa::CublasImpl>::type;
      using GemmOp = typename std::conditional<
         (remifa::compute_type::TC16==O) || (remifa::compute_type::TC32==O) , remifa::TC, remifa::FP>::type;
      using QueueType = typename std::conditional<use_cutlass, cudaStream_t, cublasHandle_t>::type;
      using AccType = typename std::conditional<
         (remifa::compute_type::TC16==O) || (remifa::compute_type::FP16==O) , half, float>::type;

      cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      remifa::cublas_check_error(custat, context);

      QueueType queue;
      if (use_cutlass) {
         queue = (QueueType) stream;
      }
      else {
         queue = (QueueType) cuhandle;
      }

      T alpha = -1.0, beta = 1.0;
      float alpha_fp32 = -1.0, beta_fp32 = 1.0;

      // Workspace
      T *d_d = nullptr;
      std::int64_t lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(T));
      remifa::cuda_check_error(cuerr, context);

      // Allocate buffer for storing a FP16 copy of the panel
      half *d_l_tmp = nullptr;
      std::int64_t lddl = ldda;
      // cuerr = cudaMalloc((void**)&d_l_tmp, lddl*nb*sizeof(half));      
      // remifa::cuda_check_error(cuerr, context);      
      half *d_u_tmp = nullptr;
      std::int64_t lddu = ldda;
      // int lddu = nb;
      // cuerr = cudaMalloc((void**)&d_u_tmp, lddu*n*sizeof(half));
      // remifa::cuda_check_error(cuerr, context);      

      // if (std::is_same<remifa::compute_type::TC32, O>::value &&
      //     !std::is_same<T, half>::value) {
      // }
      
      // Loop over outer blocks
      for (std::int64_t kk = 0; kk < nc; ++kk) {

         std::int64_t ofs = kk*nb;
         std::int64_t ofst = (kk+1)*nb;
         std::int64_t in = std::min(n-ofs, nb);
         std::int64_t inc = (in-1) / ib + 1; 
         std::int64_t updm = m-ofs;

         //
         // Perform right-looking blocked LU factorization of the
         // `kk`-th outer block
         //
         lu_nopiv_inner_rl
            <T, I, ib, GemmImpl>
            (cuhandle, 
             updm, in, nb, 
             &d_a[ofs + ofs*ldda], ldda,
             d_d, d_info,
             &d_a_cpy[ofs + ofs*ldda], ld_a_cpy);

         // std::cout << "[" << context << "]"
         //           << " kk = " << kk
         //           << std::endl;
            
         // Copy factor into fp16 buffer if necessary for TC udpate
         if (std::is_same<T, float>::value && (remifa::compute_type::TC32 == O) /*&&
             // If I == TC32 then the factors have already been
             // converted into fp16 buffer
             (remifa::compute_type::TC32 != I)*/) {
            // Copy of L and U factors into buffers
            if (updm>in) {
               remifa::convert(stream, in, updm-in, &d_a[ofs+ofst*ldda], ldda, &d_a_cpy[ofs+ofst*ld_a_cpy], ld_a_cpy);
            }
            remifa::convert(stream, updm, in, &d_a[ofs+ofs *ldda], ldda, &d_a_cpy[ofs+ofs *ld_a_cpy], ld_a_cpy);
         }

         //
         // Perform trailing submatrix update w.r.t previous panel
         // factorization
         //
         std::int64_t tm = m-ofst; // Width of trailing submatrix
         std::int64_t tn = n-ofst; // Width of trailing submatrix
         if (tn>0) {
            if (std::is_same<T, float>::value && (remifa::compute_type::TC32==O)) {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   tm, tn, nb,
                   AccType(-1.0),
                   &d_a_cpy[ofst + ofs *ld_a_cpy], ld_a_cpy,
                   &d_a_cpy[ofs  + ofst*ld_a_cpy], ld_a_cpy,
                   AccType(1.0),
                   &d_a[ofst + ofst*ldda], ldda);
            }
            else {
               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   tm, tn, nb,
                   AccType(-1.0),
                   &d_a[ofst + ofs *ldda], ldda,
                   &d_a[ofs  + ofst*ldda], ldda,
                   AccType(1.0),
                   &d_a[ofst + ofst*ldda], ldda);
            }
            
         }

      } // Outer blocks loop

      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);

      // Cleanup memory
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      remifa::cuda_check_error(cuerr, context);

   }
   
} // End of remifa namespace
