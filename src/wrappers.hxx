/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// CuSOLVER
#include <cusolverDn.h>

namespace remifa {

   /// @brief spldlt::operation enumerates operations that can be applied
   /// to a matrix * argument of a BLAS call.
   enum operation {
      /// No operation (i.e. non-transpose). Equivalent to BLAS op='N'.
      OP_N,
      /// Transposed. Equivalent to BLAS op='T'.
      OP_T
   };

   /// @brief spldlt::diagonal enumerates nature of matrix diagonal.
   enum diagonal {
      /// All diagonal elements are assumed to be identically 1.0
      DIAG_UNIT,
      /// Diagonal elements are specified in matrix data
      DIAG_NON_UNIT
   };
   
   /// @brief spldlt::fillmode enumerates which part of the matrix is
   /// specified.
   enum fillmode {
      /// The lower triangular part of the matrix is specified
      FILL_MODE_LWR,
      /// The upper triangular part of the matrix is specified
      FILL_MODE_UPR
   };

   // @brief bub::side enumerates whether the primary operand is
   //  applied on the left or right of a secondary operand
   enum side {
      /// Primary operand applied on left of secondary
      SIDE_LEFT,
      /// Primary operand applied on right of secondary
      SIDE_RIGHT
   };

   enum norm {
      // 
      NORM_M,
      // One norm
      NORM_ONE,
      // Infinity norm
      NORM_INF,
      // Frobenius norm
      NORM_FRO
   };

   enum layout {
      // Row major layout
      ROW_MAJOR,
      // Column major layout
      COLUMN_MAJOR,
   };

   template <typename T>
   int latms(const int m, const int n, const char dist,
              int* iseed, const char sym, T* d, const int mode,
              const T cond, const T dmax, const int kl,
               const int ku, const char pack, T* a,
              const int lda, T* work);
   
   // _LATMR
   template <typename T>
   int latmr(
         int m, int n, char dist, int *iseed, char sym, T *d, int mode, T cond,
         T dmax, char rsign, char grad, T *dl, int model, T condl, T *dr, int moder,
         T condr, char pivtng, int *ipivot,  int kl, int ku, T sparse, T anorm, char pack,
         T *a, int lda,
         // Workspace, not referenced if pivtng = 'N'
         int *iwork 
         );
   
   // _LAPMR
   template <typename T>
   void lapmr(bool forwrd, int m, int n, T *x, int ldx, int *k );

   // _LASWP
   template <typename T> 
   void host_laswp(int n, T *a, int lda, int k1, int k2, int *perm, int incx);

   template <typename T> 
   void host_axpy(int n, const T a, const T *x, const int incx, T *y, const int incy);

   template <typename T> 
   double host_lange(remifa::norm norm, const int m, const int n, const T *a, const int lda);

   /* _POTRF */
   template <typename T>
   int host_potrf(enum remifa::fillmode uplo, int n, T* a, int lda);

   /* _TRSM */
   template <typename T>
   void host_trsm(enum remifa::side side, enum remifa::fillmode uplo,
                  enum remifa::operation transa, enum remifa::diagonal diag,
                  int m, int n, T alpha, const T* a, int lda, T* b, int ldb);

   /* _SYRK */
   template <typename T>
   void host_syrk(enum remifa::fillmode uplo, enum remifa::operation trans,
                  int n, int k, T alpha, const T* a, int lda, T beta, T* c, int ldc);

   /* _GEMM */
   template <typename T>
   void host_gemm(enum remifa::operation transa, enum remifa::operation transb,
                  int m, int n, int k, T alpha, const T* a, int lda, const T* b,
                  int ldb, T beta, T* c, int ldc);

   // GETRF
   template <typename T>
   int host_getrf(int m, int n, T* a, int lda, int *ipiv);


   // GEQRF
   template <typename T>
   int host_geqrf(int m, int n, T *a, int lda, T *tau, T *work, int lwork);

   // ORMQR
   template <typename T>
   int host_ormqr(enum remifa::side side, enum remifa::operation trans,
                  int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc,
                  T *work, int lwork);

   // GEMV
   template <typename T>
   void host_gemv(enum remifa::operation trans, int m, int n, T alpha, T const* a, int lda,
             T const* x, int incx, T beta, T *y, int incy);

   // TRMV
   template <typename T>
   void host_trmv(
         enum remifa::fillmode uplo, enum remifa::operation trans,
         enum remifa::diagonal diag, int n, T const* a, int lda,
             T *x, int incx);

   // NRM2
   template <typename T>
   T host_nrm2(int n, T const* x, int incx);

   // DOT
   template <typename T>
   T host_dot(int n, T const* x, int incx, T const* y, int incy);

   // GETRS
   template <typename T>
   int host_getrs(enum remifa::operation trans, int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb);

   // GEEQU
   template <typename T>
   int geequ(
         int m, int n, T const* a, int lda,
         T *r, T *c, T* rowcnd, T* colcnd, T* amax);

   // GEEQUB
   template <typename T>
   int geequb(
         int m, int n, T const* a, int lda,
         T *r, T *c, T* rowcnd, T* colcnd, T* amax);

   // LAQGE
   template <typename T>
   void laqge(int m, int n, T *a, int lda, T *r, T *c, T rowcnd, T colcnd, T amax, char *equed);

   // // GEEQUB
   // template <typename T>
   // void geequb(
   //       int m, int n, T const* a, int lda,
   //       T *r, T *c, T* rowcnd, T* rowcnd, T* amax);

   
   // IAMAX
   template <typename T> 
   cublasStatus_t dev_iamax(
         cublasHandle_t handle,
         int n,
         T *x, int incx,
         int *result
         );
   
   // SWAP   
   template <typename T> 
   cublasStatus_t dev_swap(
         cublasHandle_t handle,
         int n,
         T *x, int incx,
         T *y, int incy
         );
   
   // GEMV
   template <typename T> 
   cublasStatus_t dev_gemv(
         cublasHandle_t handle, cublasOperation_t trans,
         int m, int n, T const* alpha, T const* a, int lda,
         T const* x, int incx, T const* beta,
         T *y, int incy);
   
   // _GETRF BufferSize
   template <typename T> 
   cusolverStatus_t dev_getrf_buffersize(
         cusolverDnHandle_t handle, int m, int n, T *a, int lda, int *lwork);

   // _GETRF
   template <typename T> 
   cusolverStatus_t dev_getrf(
         cusolverDnHandle_t handle, int m, int n,
         T *a, int lda,
         T *workspace,
         int *devIpiv,
         int *devInfo);
   
   // _POTRF BufferSize
   template <typename T> 
   cusolverStatus_t dev_potrf_buffersize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, int *lwork);

   // _POTRF
   template <typename T>
   cusolverStatus_t dev_potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, T *work, int lwork, int *info);

   // _SYRK
   template <typename T>
   cublasStatus_t dev_syrk(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const T *alpha, const T *a, int lda, const T *beta, T *c, int ldc);

   // _GEMM
   template <typename T>
   cublasStatus_t dev_gemm(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const T *alpha, const T *a, int lda,
         const T *b, int ldb, const T *beta, T *c, int ldc);

   // _TRSM
   template <typename T>
   cublasStatus_t dev_trsm(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag,
         int m, int n,
         const T *alpha,
         const T *a, int lda,
         T *b, int ldb);
   
   // _GEQRF bufferSize
   template <typename T>
   cusolverStatus_t dev_geqrf_buffersize(
         cusolverDnHandle_t handle, int m, int n, T *a, int lda, int *lwork);
      
   // _GEQRF
   template <typename T>
   cusolverStatus_t dev_geqrf(
         cusolverDnHandle_t handle,
         int m, int n, T *a, int lda,
         T *tau, T *work, int lwork,
         int *info);

   // _ORMQR bufferSize
   template <typename T>
   cusolverStatus_t dev_ormqr_buffersize(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const T *a, int lda,
         const T *tau,
         const T *c, int ldc,
         int *lwork);

   // _ORMQR
   template <typename T>
   cusolverStatus_t dev_ormqr(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const T *a, int lda,
         const T *tau,
         T *c, int ldc,
         T *work, int lwork,
         int *dinfo);
}
