/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "convert.cuh"
#include "errchk.hxx"
#include "gemm.cuh"
#include "lu_nopiv.cuh"
#include "lu_nopiv_inner_ll.hxx"
#include "utils.hxx"
#include "wrappers.hxx"

#include <cassert>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
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
      std::int64_t ib=BLOCK_SIZE,
      /*int nb=OUTER_BLOCK_SIZE,*/
      typename OP=T, // Outer panel prec
      typename IP=T, // Inner panel prec
      typename OF=T, // Outer factor prec
      typename IF=OF, // Inner factor prec
      typename GemmImpl=remifa::CublasImpl,
      // bool use_cutlass=false,
      typename CpyType = half
      >
   void lu_nopiv_ll(
         const cublasHandle_t cuhandle, 
         std::int64_t m, // Number of rows 
         std::int64_t n, // Number of columns
         std::int64_t nb, // Outer block size
         T *const d_a, // Matrix pointer on device 
         std::int64_t ldda, // Matrix leadind dim on device
         int *d_info, // Info device
         int slices=0, uint8_t *workspace=nullptr,
         CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
         std::int64_t ld_a_cpy=-1 // Matrix copy leading dimension
         ) {

      // Error handling
      std::string context = "lu_nopiv_ll";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status

      //
      // Input parameter checking
      //
      
      // Only support TC16 for fp16 working prec
      if ((remifa::compute_type::TC16 == O) &&
          !(std::is_same<OP, half>::value)) {
         std::cout << "[" << context << "] Update compute type NOT supported " << std::endl;
         std::exit(0);
      }
      // Only support FP16 for fp16 working prec
      else if ((remifa::compute_type::FP16 == O) &&
          !(std::is_same<OP, half>::value)) {
         std::cout << "[" << context << "] Update compute type NOT supported " << std::endl;
         std::exit(0);
      }

      // Aliases
      // using GemmImpl = typename std::conditional
      //    <use_cutlass, remifa::CutlassImpl, remifa::CublasImpl>::type;
      using GemmOp = typename std::conditional
         <(remifa::compute_type::TC16==O) || (remifa::compute_type::TC32==O),
          remifa::TC, remifa::FP>::type;
      using QueueType = typename std::conditional
         <std::is_same<GemmImpl, remifa::CublasImpl>::value,
         cublasHandle_t, cudaStream_t>::type;
      using AccType = typename std::conditional
         <(remifa::compute_type::TC16==O) || (remifa::compute_type::FP16==O) , half, float>::type;

      if (std::is_same<GemmImpl, remifa::CublasImpl>::value) {
         std::cout << "[" << context << "]" << " cuBLAS implementation" << std::endl;
      }
      else if (std::is_same<GemmImpl, remifa::CutlassImpl>::value) {
         std::cout << "[" << context << "]" << " Cutlass implementation" << std::endl;
      }
      else if (std::is_same<GemmImpl, remifa::CutlassSplitKImpl>::value) {
         std::cout << "[" << context << "]"
                   << " Cutlass SplitK implementation, slices = "
                   << slices
                   << std::endl;
      }
      else {
         throw std::invalid_argument("Gemm implementation NOT supported");
      }
      
      // Number of block columns
      std::int64_t const nc = (n-1) / nb +1;

      std::cout << "[" << context << "]"
                << " W=" << type_name<T>()
                << ", I=" << compute_type_name<I>()
                << ", O=" << compute_type_name<O>()
                << ", OP=" << type_name<OP>()
                << ", OF=" << type_name<OF>()
                << ", IP=" << type_name<IP>()
                << ", IF=" << type_name<IF>()
                << std::endl;
      // std::cout << "[spldlt::gpu::factor] nc = " << nc << std::endl;

      if (std::is_same<T, float>::value && (remifa::compute_type::TC32==O) &&
          d_a_cpy == nullptr) {
         throw std::invalid_argument("A fp16 buffer should be provided");
      }

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

      T alpha = -1.0, beta = 1.0;
      half alpha_fp16 = -1.0, beta_fp16 = 1.0;
      float alpha_fp32 = -1.0, beta_fp32 = 1.0;
      OP alpha_op = -1.0, beta_op = 1.0;
      
      // Workspace at prec determined by IF
      std::int64_t lddd = ib;
      // OP *d_d = nullptr;
      // cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(OP));
      IF *d_d = nullptr;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(IF));
      remifa::cuda_check_error(cuerr, context);

      OP *d_l_tmp = nullptr;
      std::int64_t lddl = -1;
      OP *d_u_tmp = nullptr;
      std::int64_t lddu = -1;


      if (std::is_same<T, OP>::value && std::is_same<T, OF>::value) {
         d_l_tmp = (OP*) d_a;
         lddl = ldda;
         d_u_tmp = (OP*) d_a;
         lddu = ldda;
      }
      else {

         // The panel precision is different from the working
         // precision. We allocate buffer for computing the panel in
         // the requested precision OP.
         std::cout << "[" << context << "] " << "Allocate L and U buffers" << std::endl;
         lddl = ldda;
         cuerr = cudaMalloc((void**)&d_l_tmp, lddl*nb*sizeof(OP));      
         remifa::cuda_check_error(cuerr, context);      
         lddu = nb;
         cuerr = cudaMalloc((void**)&d_u_tmp, lddu*n*sizeof(OP));      
         remifa::cuda_check_error(cuerr, context);      
      }

      if (!std::is_same<T, OF>::value) {
         // The following intial convertion from matrix A into buffer
         // is only necessary if factorization prec is different from
         // working precision.
         
         int updm = m;
         int in = std::min(n, nb);
         remifa::convert(stream, updm, in, d_a, ldda, d_l_tmp, lddl);
         remifa::convert(stream, in, updm, d_a, ldda, d_u_tmp, lddu);
      }
      
      // cuBLAS type associated with the working precision
      cudaDataType_t wp_cutype = cublas_type<T>();
      cudaDataType_t op_cutype = cublas_type<OP>();
      
      // Loop over outer blocks
      for (std::int64_t kk = 0; kk < nc; ++kk) {

         // std::cout << "[" << context << "] kk = " << kk << std::endl;

         std::int64_t ofs = kk*nb;
         std::int64_t ofst = (kk+1)*nb;
         // Width of outer block column 
         std::int64_t in = std::min(n-ofs, nb);
         // Width of trailing submatrix
         std::int64_t updm = m-ofs;
         
         if (ofs>0) {

            //
            // First update block column (L factor) 
            //
            
            if (std::is_same<T, OP>::value) {
               // If outer panel precison and working precision match,
               // then we simply use the pointer `d_l_tmp` and
               // `d_u_tmp` to point to current outer block-column/row
               // in A.
               d_l_tmp = (OP*) &d_a[ofs+ofs*ldda];
               lddl = ldda;
               d_u_tmp =  (OP*) &d_a[ofs+ofs*ldda];
               lddu = ldda;
            } else {
               // Put L panel into buffer at precisin OP
               remifa::convert(stream, updm, in, &d_a[ofs+ofs*ldda], ldda, d_l_tmp, lddl);
            }
            
            if ((remifa::compute_type::FP32 == O) ||
                (remifa::compute_type::FP16 == O)) {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   updm, in, ofs,
                   AccType(-1.0),
                   &d_a[ofs], ldda,
                   &d_a[ofs*ldda], ldda,
                   AccType(1.0),
                   d_l_tmp, lddl,
                   slices, workspace);
               
            }
            else if(remifa::compute_type::TC32 == O) {
               //
               // O==TC32 && W=fp32 or fp16

               // Aliases to fp16 input factors
               half *d_l_f16 = nullptr;
               std::int64_t ld_l_f16 = -1;
               half *d_u_f16 = nullptr;
               std::int64_t ld_u_f16 = -1;

               if (std::is_same<T, half>::value) {
                  // If working prec is fp16, get factors from
                  // original matrix
                  d_l_f16 = (half*)d_a;
                  ld_l_f16 = ldda;
                  d_u_f16 = (half*)d_a;
                  ld_u_f16 = ldda;
               }
               else {
                  // Else, if working prec is fp32, get factors
                  // from temporary buffer containing fp16 copy of
                  // factors
                  d_l_f16 = d_a_cpy;
                  ld_l_f16 = ld_a_cpy;
                  d_u_f16 = d_a_cpy;
                  ld_u_f16 = ld_a_cpy;
               }

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   updm, in, ofs,
                   AccType(-1.0),
                   &d_l_f16[ofs], ld_l_f16,
                   &d_u_f16[ofs*ldda], ld_u_f16,
                   AccType(1.0),
                   d_l_tmp, lddl,
                   slices, workspace);
               
            } // O==TC32
            else if(remifa::compute_type::TC16 == O) {

               gemmex<GemmImpl, GemmOp, AccType>
                  (queue,
                   updm, in, ofs,
                   AccType(-1.0),
                   &d_a[ofs], ldda,
                   &d_a[ofs*ldda], ldda,
                   AccType(1.0),
                   d_l_tmp, lddl,
                   slices, workspace);

            }
            else {
               std::cout << "[" << context << "] Update compute type NOT supported " << std::endl;
               std::exit(0);
            }

            
            if (updm>in) {

               // std::cout << "[" << context << "] Update U" << std::endl;
               
               //
               // Second update block-row (U factor) 
               //

               if (!std::is_same<T, OP>::value) {
                  // Convert U panel from A to buffer at precisin OP
                  // remifa::convert(stream, in, updm-in, &d_a[ofs+(ofs+in)*ldda], ldda, &d_u_tmp[in*lddu], lddu);

                  // Convert the block-row including the diagonal
                  // block
                  remifa::convert(stream, in, updm, &d_a[ofs+ofs*ldda], ldda, d_u_tmp, lddu);
               }
               
               if((remifa::compute_type::FP32 == O) ||
                  (remifa::compute_type::FP16 == O)) {
                  // O=fp32 and OP=fp32
                  
                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue, in, updm-in, ofs,
                      AccType(-1.0),
                      &d_a[ofs], ldda,
                      &d_a[(ofs+in)*ldda], ldda,
                      AccType(1.0),
                      &d_u_tmp[in*lddu], lddu,
                      slices, workspace);

               } // O=fp16 and OP=fp16
               else if(remifa::compute_type::TC32 == O) {

                  //
                  // O=TC32 and OP=fp32 or OP=fp16

                  // Aliases to input fp16 factors
                  half *d_l_f16 = nullptr;
                  std::int64_t ld_l_f16 = -1;
                  half *d_u_f16 = nullptr;
                  std::int64_t ld_u_f16 = -1;

                  if (std::is_same<T, half>::value) {
                     // If working prec is fp16, get factors from
                     // original matrix
                     d_l_f16 = (half*)d_a;
                     ld_l_f16 = ldda;
                     d_u_f16 = (half*)d_a;
                     ld_u_f16 = ldda;
                  }
                  else {
                     // Else, if working prec is fp32, get factors
                     // from temporary buffer containing fp16 copy of
                     // factors
                     d_l_f16 = d_a_cpy;
                     ld_l_f16 = ld_a_cpy;
                     d_u_f16 = d_a_cpy;
                     ld_u_f16 = ld_a_cpy;
                  }
                  
                  if (std::is_same<T, OP>::value) {

                     // std::cout << "[" << context << "] TETETETET" << std::endl;

                     gemmex<GemmImpl, GemmOp, AccType>
                        (queue, in, updm-in, ofs,
                         AccType(-1.0),
                         &d_l_f16[ofs          ], ld_l_f16,
                         &d_u_f16[(ofs+in)*ldda], ld_u_f16,
                         AccType(1.0),
                         &d_u_tmp[in*lddu], lddu,
                         slices, workspace);

                  }
                  else {

                     gemmex<GemmImpl, GemmOp, AccType>
                        (queue, in, updm, ofs,
                         AccType(-1.0),
                         &d_l_f16[ofs     ], ld_l_f16,
                         &d_u_f16[ofs*ldda], ld_u_f16,
                         AccType(1.0),
                         d_u_tmp, lddu,
                         slices, workspace);
                  }

               } // O compute type
               else if(remifa::compute_type::TC16 == O) {
                  //
                  // O=TC16 and OP=fp16

                  gemmex<GemmImpl, GemmOp, AccType>
                     (queue, in, updm-in, ofs,
                      AccType(-1.0),
                      &d_a[ofs], ldda,
                      &d_a[(ofs+in)*ldda], ldda,
                      AccType(1.0),
                      &d_u_tmp[in*lddu], lddu,
                      slices, workspace);
               }

               // If panel prec if different from factorization prec,
               // then we need to convert the panel entries (into the
               // original matrix) to factrization prec.
               if ((!std::is_same<T, OP>::value) && (!std::is_same<OF, OP>::value)) {
                  // Put U factor entries, in prec OP, back into matrix with prec T
                  remifa::convert(stream, in, updm-in, &d_u_tmp[in*lddu], lddu, &d_a[ofs+(ofs+in)*ldda], ldda);
               }
               
            } // updm>in
            else if (!std::is_same<T, OF>::value) {
               // updm == in i.e. we are processing the last update,
               // and factor prec is different from working prec. This
               // means that we compute the factors in buffers rather
               // than original matrix. In this case we need to copy
               // the updated L factors into U buffer which has not
               // been updated.
               
               // std::cout << "[" << context << "] Copy L into U buffer, updm = " << updm << ", in = " << in << std::endl;
               // cudaMemcpyAsync(d_u_tmp, d_l_tmp, updm*in*sizeof(OP), cudaMemcpyDeviceToDevice, stream);

               cudaMemcpy2DAsync(
                     reinterpret_cast<OP*>(d_u_tmp), lddu*sizeof(OP),
                     d_l_tmp, lddl*sizeof(OP),
                     updm*sizeof(OP), in,
                     cudaMemcpyDeviceToDevice,
                     stream);

            }
            
            if ((!std::is_same<T, OP>::value) && (!std::is_same<OF, OP>::value)) {
               // Put L back into matrix
               remifa::convert(stream, updm, in, d_l_tmp, lddl, &d_a[ofs+ofs*ldda], ldda);
            }
            
         } // ofs>0

         // If factorization prec is the same as the working prec,
         // then we factor the outer block directly into the input
         // matrix, otherwise we perform the factorization into the panel buffer
         if (std::is_same<T, OF>::value) {
         // Perform outer block factorization at precision T (working
         // precision)
            lu_nopiv_inner_ll
               <T, I, ib, IP, IF, GemmImpl, /*verbose*/false>
               (cuhandle,
                updm, in,
                (T*) &d_a[ofs + ofs*ldda], ldda,
                (T*) &d_a[ofs + ofs*ldda], ldda,
                (IF*) d_d, d_info,
                slices, workspace,
                &d_a_cpy[ofs + ofs*ldda], ld_a_cpy);

         // std::cout << "[" << context << "] " << " &d_a[ofs + ofs*ldda] = " << &d_a[ofs + ofs*ldda]
         //           << ", ldda = " <<  ldda
         //           << std::endl;
         // std::cout << "[" << context << "] " << " d_u_tmp = " << d_u_tmp
         //           << ", lddu = " << lddu
         //           << std::endl;

         }
         else {
            // Perform outer block factorization at precision OP (outer
            // panel precision)

            // std::cout << "[" << context << "] factor panel, kk = " << kk << std::endl;
                           
            // FIXME: convert panel into buffer if OP != OF?
            // if (!std::is_same<OF, OP>::value) ..
            
            lu_nopiv_inner_ll
               <OF, I, ib, IP, IF, GemmImpl, /*verbose*/false>
               (cuhandle,
                updm, in,
                (OF*) d_l_tmp, lddl,
                (OF*) d_u_tmp, lddu,
                (IF*) d_d, d_info,
                slices, workspace,
                &d_a_cpy[ofs + ofs*ldda], ld_a_cpy);

            if (!(std::is_same<OF, float>::value && (remifa::compute_type::TC32==I))) {
               // No need to convert back to fp16 buffer if I=TC32
               // because factor are converted in the inner factor
               // routine.
               
               // Convert and put the computed back into the input matrix
               // Number of block-columns
               std::int64_t inc = (in-1) / ib + 1;
               // std::cout << "[" << context << "] ib = " << ib << std::endl;

               // Factor outer block
               for (std::int64_t k = 0; k < inc; ++k) {
         
                  // std::cout << "[" << context << "] k = " << k << std::endl;
                  // Factor kth block column
                  std::int64_t iofs = k*ib; // Number of eliminated columns
                  std::int64_t cblkm = updm-iofs; // Block column height
                  std::int64_t cblkn = std::min(in-iofs, ib); // Block column width
                  std::int64_t iofst = (k+1)*ib; // Offset to trailing submatrix in outer block

                  // std::cout << "[" << context << "] cblkm = " << cblkm << ", cblkn = " << cblkn << std::endl;
                  // std::cout << "[" << context << "] iofs = " << iofs << ", iofst = " << iofst << std::endl;

                  if (cblkm>cblkn) {
                     remifa::convert(stream, cblkn, cblkm-cblkn, &d_u_tmp[iofs+iofst*lddu], lddu, &d_a[ofs+iofs+(ofs+iofst)*ldda], ldda);
                     // remifa::convert(stream, cblkn, cblkm, &d_u_tmp[iofs+iofs*lddu], lddu, &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda);
                  }
                  remifa::convert(stream, cblkm, cblkn, &d_l_tmp[iofs+iofs*lddl], lddl, &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda);
               }
            }
         } // T=OF
         
         // if (!std::is_same<T, OP>::value) {

         //    // Number of block-columns
         //    int inc = (in-1) / ib + 1;
         //    // Factor outer block
         //    for (int k = 0; k < inc; ++k) {
         
         //       // std::cout << "[" << context << "] k = " << k << std::endl;
         //       // Factor kth block column
         //       int iofs = k*ib; // Number of eliminated columns
         //       int cblkm = updm-iofs; // Block column height
         //       int cblkn = std::min(in-iofs, ib); // Block column width
         //       // std::cout << "[" << context << "] cblkm = " << cblkm << ", cblkn = " << cblkn << std::endl;
         //       int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block

         //       remifa::convert(stream, cblkm, cblkn, &d_l_tmp[iofs+iofs*lddl], lddl, &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda);
         //       if (cblkm>cblkn) {
         //          remifa::convert(stream, cblkn, cblkm-cblkn, &d_u_tmp[iofs+iofst*lddu], lddu, &d_a[ofs+iofs+(ofs+iofst)*ldda], ldda);
         //       }
         //    }
            
         // if (updm>in) {
         //    // Put U back into matrix
         //    remifa::convert(stream, in, updm-in, &d_u_tmp[in*lddu], lddu, &d_a[ofs+(ofs+in)*ldda], ldda);
         // } // updm>in
         // // Put L back into matrix
         // remifa::convert(stream, updm, in, d_l_tmp, lddl, &d_a[ofs+ofs*ldda], ldda);

         // } // !std::is_same<T, OP>::value

         // Copy factor into fp16 buffer if necessary for TC udpate
         if (std::is_same<T, float>::value && (remifa::compute_type::TC32==O) &&
             // If I == TC32 then the factors have already been
             // converted into fp16 buffer
             (remifa::compute_type::TC32 != I)) {
            // Copy of L and U factors into buffers
            if (updm>in) {
               remifa::convert(stream, in, updm-in, &d_a[ofs+ofst*ldda], ldda, &d_a_cpy[ofs+ofst*ld_a_cpy], ld_a_cpy);
            }
            remifa::convert(stream, updm, in, &d_a[ofs+ofs *ldda], ldda, &d_a_cpy[ofs+ofs *ld_a_cpy], ld_a_cpy);
         }
         
      } // Outer blocks loop

      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);

      // Cleanup memory
      if (!std::is_same<T, OP>::value) {
         cuerr = cudaFree(d_l_tmp);
         remifa::cuda_check_error(cuerr, context);
         cuerr = cudaFree(d_u_tmp);
         remifa::cuda_check_error(cuerr, context);
      }
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      remifa::cuda_check_error(cuerr, context);

   }

   // Helper functions
   
   //
   // W=fp32 O=OP=OF=fp32 I=IP=IF=fp32
   void lu_nopiv_ll_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda, int *d_info);
   //
   // W=fp32, O=OP=OF=fp32, I=IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda, int *d_info);

   //
   // W=fp32, O=OP=OF=fp32, I=IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32
   void lu_nopiv_ll_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda, int *d_info, half *d_a_f16, int ld_a_f16);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, half *d_a_f16, int ld_a_f16);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=fp32, IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f32_fp32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace, half *d_a_f16, int ld_a_f16);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32
   void lu_nopiv_ll_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, half *d_a_f16, int ld_a_f16);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, half *d_a_f16, int ld_a_f16);

   //
   // W=fp32 O=tc32, OP=OF=fp32 and I=tc32, IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f32_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, float *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace, half *d_a_f16, int ld_a_f16);

   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16
   void lu_nopiv_ll_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=OP=OF=fp16 I=IP=IF=fp16, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);
   
   //
   // W=fp16 O=TC16, OP=OF=fp16 I=fp16, IP=IF=fp16
   void lu_nopiv_ll_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=fp16, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=fp16, IP=IF=fp16, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16
   void lu_nopiv_ll_f16_tc16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC16, OP=OF=fp16 I=TC16, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16_tc16xtc16(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16
   void lu_nopiv_ll_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda, int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp16, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP16, IP=IF=fp16
   void lu_nopiv_ll_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);
   
   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP16, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP16, IP=IF=fp16, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=FP16, IP=IF=fp16
   void lu_nopiv_ll_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=FP16, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=fp32, OF=fp16 I=FP16, IP=IF=fp16, CUTLASS
   void lu_nopiv_ll_cutlass_splitk_f16_f16xtc32_fp32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32
   void lu_nopiv_ll_f16_f32xt32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f16_f32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb, half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=FP32, IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_f32xtc32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info, int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32
   void lu_nopiv_ll_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info);
   
   //
   // W=fp16 O=TC32, OP=OF=fp16 I=TC32, IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp16xfp32_fp16xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace);

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32
   void lu_nopiv_ll_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32, CUTLASS
   void lu_nopiv_ll_cutlass_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info);

   //
   // W=fp16 O=TC32, OP=OF=fp32 I=TC32, IP=IF=fp32, CUTLASS SplitK
   void lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp32xfp32_fp32xfp32(
         const cublasHandle_t cuhandle, int m, int n, int nb,
         half *const d_a, int ldda,
         int *d_info,
         int slices, uint8_t *workspace);

} // End of remifa namespace
