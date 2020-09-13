/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include <iostream>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
#endif

#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace remifa {
   
   struct CublasImpl;
   struct CutlassImpl;
   struct CutlassSplitKImpl;

   struct TC;
   struct FP;

   template<
      typename GemmImpl=remifa::CublasImpl,
      typename Op=remifa::FP,
      typename AccType,
      typename InType,
      typename OutType=InType,
      typename StreamType=cudaStream_t
      >
   void gemmex(
         StreamType const stream,
         int m, int n, int k,
         AccType alpha,
         InType const *a, int lda,
         InType const *b, int ldb,
         AccType beta,
         OutType *c, int ldc,
         int slices=0, uint8_t *workspace=nullptr);

   template<
      typename GemmImpl=remifa::CutlassSplitKImpl,
      typename Op=remifa::FP,
      typename ElementOutput, typename ElementAccumulator=ElementOutput
      >
   size_t gemm_workspace_size(
         int m, int n, int k,
         int slices);
   
#if defined(HAVE_CUTLASS)

   template<typename T>
   void cutlass_gemm(
         cudaStream_t const stream,
         int M, int N, int K,
         T alpha,
         T const *A, int lda,
         T const *B, int ldb,
         T beta,
         T *C, int ldc);

   template<typename T>
   void cutlass_gemm_splitk(
         cudaStream_t const stream,
         int M, int N, int K,
         T alpha,
         T const *A, int lda,
         T const *B, int ldb,
         T beta,
         T *C, int ldc,
         int slices, uint8_t *workspace);

   template<typename T, typename Output=T>
   void cutlass_gemm_tensor_op_t32(
         cudaStream_t const stream,
         int M, int N, int K,
         float alpha,
         T const *A, int lda,
         T const *B, int ldb,
         float beta,
         Output *C, int ldc);

   template<typename T, typename Output=T>
   void cutlass_gemm_splitk_tensor_op_t32(
         cudaStream_t const stream,
         int M, int N, int K,
         float alpha,
         T const *A, int lda,
         T const *B, int ldb,
         float beta,
         Output *C, int ldc,
         int slices, uint8_t *workspace);

   template<typename T, typename Output=T>
   void cutlass_gemm_tensor_op_t16(
         cudaStream_t const stream,
         int M, int N, int K,
         half alpha,
         T const *A, int lda,
         T const *B, int ldb,
         half beta,
         Output *C, int ldc);

   template<typename T, typename Output=T>
   void cutlass_gemm_splitk_tensor_op_t16(
         cudaStream_t const stream,
         int M, int N, int K,
         half alpha,
         T const *A, int lda,
         T const *B, int ldb,
         half beta,
         Output *C, int ldc,
         int slices, uint8_t *workspace);

#endif

}
