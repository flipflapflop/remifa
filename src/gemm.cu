/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "errchk.hxx"
#include "gemm.cuh"

#include <iostream>

namespace remifa {
   
   ////////////////////////////////
   // fp32

   // fp32, cuBLAS

   template<>
   void gemmex
   <remifa::CublasImpl, remifa::FP, float, float, float, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";
      
      cublasStatus_t custat = cublasSgemm(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
      remifa::cublas_check_error(custat, context);
      
   }

   // fp32, CUTLASS

   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::FP, float, float, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {
#if defined(HAVE_CUTLASS)

      cutlass_gemm(stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
   }

   // fp32, CUTLASS SplitK

   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::FP, float, float, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {
#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk(stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);

#endif
   }

   
   ////////////////////////////////
   // fp16

   // fp16, cuBLAS

   template<>
   void gemmex
   <remifa::CublasImpl, remifa::FP, half, half, half, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";
      
      cublasStatus_t custat = cublasHgemm(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
      remifa::cublas_check_error(custat, context);
      
   }

   // fp16, CUTLASS

   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::FP, half, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm(stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif      
   }

   // fp16, CUTLASS SplitK

   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::FP, half, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk(stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);
#endif      
   }

   
   ////////////////////////////////
   // TC16

   // tc16, cuBLAS

   template<>
   void gemmex
   <remifa::CublasImpl, remifa::TC, half, half, half, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";

      cublasStatus_t custat = cublasGemmEx(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, CUDA_R_16F, lda,
            b, CUDA_R_16F, ldb,
            &beta,
            c, CUDA_R_16F, ldc,
            CUDA_R_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      remifa::cublas_check_error(custat, context);
      
   }

   // tc16, CUTLASS

   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::TC, half, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_tensor_op_t16(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif      

   }

   // tc16, CUTLASS SplitK

   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::TC, half, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    half alpha,
    half const *a, int lda,
    half const *b, int ldb,
    half beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {
#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk_tensor_op_t16(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);
#endif      

   }

   ////////////////////////////////
   // TC32

   // tc32, cuBLAS

   // InType=fp16 and OutType=fp16
   
   template<>
   void gemmex
   <remifa::CublasImpl, remifa::TC, float, half, half, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";

      cublasStatus_t custat = cublasGemmEx(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, CUDA_R_16F, lda,
            b, CUDA_R_16F, ldb,
            &beta,
            c, CUDA_R_16F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      remifa::cublas_check_error(custat, context, "Failed to launch cublasGemmEx");
   }

   // InType=fp16 and OutType=fp32

   template<>
   void gemmex
   <remifa::CublasImpl, remifa::TC, float, half, float, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";

      // std::cout << "[gemmex] m = " << m
      //           << ", n = " << n
      //           << ", k = " << k
      //           << ", lda = " << lda
      //           << ", ldc = " << ldc
      //           << ", ldb = " << ldb << std::endl;
      
      cublasStatus_t custat = cublasGemmEx(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, CUDA_R_16F, lda,
            b, CUDA_R_16F, ldb,
            &beta,
            c, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      remifa::cublas_check_error(custat, context, "Failed to launch cublasGemmEx");      
   }

   // InType=fp32 and OutType=fp32

   template<>
   void gemmex
   <remifa::CublasImpl, remifa::TC, float, float, float, cublasHandle_t>
   (cublasHandle_t const cuhandle,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

      std::string context = "gemmex";

      cublasStatus_t custat = cublasGemmEx(
            cuhandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a, CUDA_R_32F, lda,
            b, CUDA_R_32F, ldb,
            &beta,
            c, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      remifa::cublas_check_error(custat, context);      
   }

   
   // tc32, CUTLASS

   // InType=fp16 and OutType=fp16
   
   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::TC, float, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {
#if defined(HAVE_CUTLASS)

      cutlass_gemm_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
   }

   // InType=fp16 and OutType=fp32

   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::TC, float, half, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

      // std::cout << "[gemmex] TETETETTETE " << std::endl;
#if defined(HAVE_CUTLASS)

      cutlass_gemm_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
   }

   // InType=fp32 and OutType=fp32

   template<>
   void gemmex
   <remifa::CutlassImpl, remifa::TC, float, float, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
   }

   // tc32, CUTLASS SplitK

   // InType=fp16 and OutType=fp16
   
   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::TC, float, half, half, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    half *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);
#endif

   }

   // InType=fp16 and OutType=fp32

   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::TC, float, half, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    half const *a, int lda,
    half const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);
#endif

   }

   // InType=fp32 and OutType=fp32

   template<>
   void gemmex
   <remifa::CutlassSplitKImpl, remifa::TC, float, float, float, cudaStream_t>
   (cudaStream_t const stream,
    int m, int n, int k,
    float alpha,
    float const *a, int lda,
    float const *b, int ldb,
    float beta,
    float *c, int ldc,
    int slices, uint8_t *workspace) {

#if defined(HAVE_CUTLASS)

      cutlass_gemm_splitk_tensor_op_t32(
            stream, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, slices, workspace);
#endif

   }

}
