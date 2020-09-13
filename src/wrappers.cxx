/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// REMIFA 
#include "wrappers.hxx"

// STD
#include <stdexcept>

// CuSOLVER
#include <cusolverDn.h>
#include <cuda_fp16.h>

extern "C" {
   // AXPY
   void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
   void saxpy_(const int *n, const float *a, const float *x, const int *incx, float *y, const int *incy);
   // LANGE
   double dlange_(char *norm, int *m, int *n, const double *a, int *lda);
   // POTRF
   void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
   void spotrf_(char *uplo, int *n, float  *a, int *lda, int *info);
   // TRSM
   void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const double *alpha, const double *a, int *lda, double *b, int *ldb);
   void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const float  *alpha, const float  *a, int *lda, float  *b, int *ldb);
   // GEMM
   void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float *beta, float* c, int* ldc);
   // SYRK
   void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, const double *a, int *lda, double *beta, double *c, int *ldc);
   void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, const float *a, int *lda, float *beta, float *c, int *ldc);

   // GETRF
   void dgetrf_(int *m, int *n, double* a, int *lda, int *ipiv, int *info);
   void sgetrf_(int *m, int *n, float* a, int *lda, int *ipiv, int *info);

   void dlaswp_(int *n, double* a, int *lda, int *k1, int *k2, int *ipiv, const int *incx);
   // GEQRF
   void sgeqrf_(int *m, int *n, float* a, int* lda, float *tau, float *work, int *lwork, int *info);
   void dgeqrf_(int *m, int *n, double* a, int* lda, double *tau, double *work, int *lwork, int *info);
   // ORMQR
   void sormqr_(char *side, char* trans, int *m, int *n, int *k, float* a, int* lda, float *tau, float* c, int* ldc, float *work, int *lwork, int *info);
   // GEMV
   void sgemv_(char* trans, int *m, int *n, float* alpha, float const* a, int* lda, const float *x, int const* incx, float *beta, float *y, int const *incy);
   void dgemv_(char* trans, int *m, int *n, double* alpha, double const* a, int* lda, const double *x, int const* incx, double *beta, double *y, int const *incy);
   // NRM2
   float snrm2_(int *n, float const* x, int const* incx);
   double dnrm2_(int *n, double const* x, int const* incx);
   // DOT
   float sdot_(int *n, float const* x, int const* incx, float const* y, int const* incy);
   double ddot_(int *n, double const* x, int const* incx, double const* y, int const* incy);
   // GETRS
   void sgetrs_(char *trans, int *n, int *nrhs, float const* a, int *lda, int const* ipiv, float const* b, int *ldb, int *info);
   void dgetrs_(char *trans, int *n, int *nrhs, double const* a, int *lda, int const* ipiv, double const* b, int *ldb, int *info);

   // GEEQU
   void dgeequ_(int *m, int *n, double const* a, int *lda, double *r, double *c, double* rowcnd, double* colcnd, double* amax, int *info);
   void sgeequ_(int *m, int *n, float const* a, int *lda, float *r, float *c, float* rowcnd, float* colcnd, float* amax, int *info);

   // GEEQUB
   void dgeequb_(int *m, int *n, double const* a, int *lda, double *r, double *c, double* rowcnd, double* colcnd, double* amax, int *info);
   void sgeequb_(int *m, int *n, float const* a, int *lda, float *r, float *c, float* rowcnd, float* colcnd, float* amax, int *info);

   // LAQGE
   void dlaqge_(int *m, int *n, double *a, int *lda, double *r, double *c, double* rowcnd, double* colcnd, double* amax, char *equed);
   void slaqge_(int *m, int *n, float *a, int *lda, float *r, float *c, float* rowcnd, float* colcnd, float* amax, char *equed);

   // LAPMR
   void dlapmr_(bool *forwrd, int *m, int *n, double *x, int *ldx, int *k );
   void slapmr_(bool *forwrd, int *m, int *n, float *x, int *ldx, int *k );

   // LATMR
   void slatmr_(
         int const* m, int const* n, char const* dist, int *iseed, char const* sym, float const* d,
         int const* mode, float const*cond, float const* dmax, char const* rsign, char const* grade,
         float *dl, int const* model, float const* condl, float *dr, int const* moder,
         float const* condr, char const* pivtng, int const* ipivot, int *kl, int const* ku,
         float const* sparse, float const* anorm, char const* pack, float *a,
         int const* lda, int *iwork, int *info);

   void dlatmr_(
         int const* m, int const* n, char const* dist, int *iseed, char const* sym, double const* d,
         int const* mode, double const*cond, double const* dmax, char const* rsign, char const* grade,
         double *dl, int const* model, double const* condl, double *dr, int const* moder,
         double const* condr, char const* pivtng, int const* ipivot, int *kl, int const* ku,
         double const* sparse, double const* anorm, char const* pack, double *a,
         int const* lda, int *iwork, int *info);

   // LATMS
   void slatms_(
         const int* m, const int* n, const char* dist,
         int* iseed, const char* sym, float* d, const int* mode,
         const float* cond, const float* dmax, const int* kl,
         const int* ku, const char* pack,
         float* a, const int* lda, float* work, int* info );
   
   void dlatms_(
         const int* m, const int* n, const char* dist,
         int* iseed, const char* sym, double* d, const int* mode,
         const double* cond, const double* dmax, const int* kl,
         const int* ku, const char* pack,
         double* a, const int* lda, double* work, int* info );

   void strmv_(char* uplo, char* trans, char* diag, const int* n,
               float const* a, const int* lda, float *x, const int* incx);
   void dtrmv_(char* uplo, char* trans, char* diag, const int* n,
               double const* a, const int* lda, double *x, const int* incx);
}

namespace remifa {

   // STRMV
   template <>
   void host_trmv<float>(
         enum remifa::fillmode uplo, enum remifa::operation trans,
         enum remifa::diagonal diag, int n, float const* a, int lda,
         float *x, int incx) {
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      char fdiag = (diag==remifa::DIAG_UNIT) ? 'U' : 'N';

      strmv_(&fuplo, &ftrans, &fdiag, &n,
             a, &lda, x, &incx);
   }
   // DTRMV
   template <>
   void host_trmv<double>(
         enum remifa::fillmode uplo, enum remifa::operation trans,
         enum remifa::diagonal diag, int n, double const* a, int lda,
         double *x, int incx) {
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      char fdiag = (diag==remifa::DIAG_UNIT) ? 'U' : 'N';

      dtrmv_(&fuplo, &ftrans, &fdiag, &n,
             a, &lda, x, &incx);
   }

   
   // SLATMS
   template <>
   int latms<float>(
         const int m, const int n, const char dist,
         int* iseed, const char sym, float* d, const int mode,
         const float cond, const float dmax, const int kl,
         const int ku, const char pack, float* a,
         const int lda, float* work) {
      
      int info;
      slatms_( &m, &n, &dist, iseed, &sym, d, &mode,
               &cond, &dmax, &kl, &ku, &pack, a, &lda, work, &info );
      return info;
   }
   // DLATMS
   template <>
   int latms<double>(
         const int m, const int n, const char dist,
         int* iseed, const char sym, double* d, const int mode,
         const double cond, const double dmax, const int kl,
         const int ku, const char pack, double* a,
         const int lda, double* work) {
      
      int info;
      dlatms_( &m, &n, &dist, iseed, &sym, d, &mode,
               &cond, &dmax, &kl, &ku, &pack, a, &lda, work, &info );
      return info;
   }
   
   // SLATMR
   template <>
   int latmr<float>(
         int m, int n, char dist, int *iseed, char sym, float *d, int mode, float cond,
         float dmax, char rsign, char grad, float *dl, int model, float condl, float *dr, int moder,
         float condr, char pivtng, int *ipivot, int kl, int ku, float sparse, float anorm, char pack,
         float *a, int lda, int *iwork) {

      int info;
      slatmr_(
            &m, &n, &dist, iseed, &sym, d, &mode, &cond, &dmax, &rsign, &grad, dl,
            &model, &condl, dr, &moder, &condr, &pivtng, ipivot, &kl, &ku, &sparse,
            &anorm, &pack, a, &lda, iwork, &info);
      return info;
   }
   // DLATMR
   template <>
   int latmr<double>(
         int m, int n, char dist, int *iseed, char sym, double *d, int mode, double cond,
         double dmax, char rsign, char grad, double *dl, int model, double condl, double *dr, int moder,
         double condr, char pivtng, int *ipivot, int kl, int ku, double sparse, double anorm, char pack,
         double *a, int lda, int *iwork) {

      int info;
      dlatmr_(
            &m, &n, &dist, iseed, &sym, d, &mode, &cond, &dmax, &rsign, &grad, dl,
            &model, &condl, dr, &moder, &condr, &pivtng, ipivot, &kl, &ku, &sparse,
            &anorm, &pack, a, &lda, iwork, &info);
      return info;
   }
   
   // SLAPMR
   template <>
   void lapmr<float>(bool forwrd, int m, int n, float *x, int ldx, int *k ) {
      slapmr_(&forwrd, &m, &n, x, &ldx, k );
   }
   
   // SLAQGE
   template <>
   void laqge<float>(
         int m, int n, float *a, int lda, float *r, float *c, float rowcnd,
         float colcnd, float amax, char *equed) {
      slaqge_(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, equed);
   }
   // DLAQGE
   template <>
   void laqge<double>(
         int m, int n, double *a, int lda, double *r, double *c, double rowcnd,
         double colcnd, double amax, char *equed) {
      dlaqge_(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, equed);
   }
   
   
   // SGEEQU
   template <>
   int geequ<float>(
         int m, int n, float const* a, int lda,
         float *r, float *c, float* rowcnd, float* colcnd, float* amax) {
      int info;
      sgeequ_(&m, &n, a, &lda, r, c, rowcnd, colcnd, amax, &info);
      return info;
   }
   // DGEEQU
   template <>
   int geequ<double>(
         int m, int n, double const* a, int lda,
         double *r, double *c, double* rowcnd, double* colcnd, double* amax) {
      int info;
      dgeequ_(&m, &n, a, &lda, r, c, rowcnd, colcnd, amax, &info);
      return info;
   }

   // SGEEQUB
   template <>
   int geequb<float>(
         int m, int n, float const* a, int lda,
         float *r, float *c, float* rowcnd, float* colcnd, float* amax) {
      int info;
      sgeequb_(&m, &n, a, &lda, r, c, rowcnd, colcnd, amax, &info);
      return info;
   }
   // DGEEQUB
   template <>
   int geequb<double>(
         int m, int n, double const* a, int lda,
         double *r, double *c, double* rowcnd, double* colcnd, double* amax) {
      int info;
      dgeequb_(&m, &n, a, &lda, r, c, rowcnd, colcnd, amax, &info);
      return info;
   }

   
   // SDOT
   template<>
   float host_dot<float>(int n, float const* x, int incx, float const* y, int incy) {
      return sdot_(&n, x, &incx, y, &incy);
   }
   // DDOT
   template<>
   double host_dot<double>(int n, double const* x, int incx, double const* y, int incy) {
      return ddot_(&n, x, &incx, y, &incy);
   }
       
   // SNRM2
   template<>
   float host_nrm2<float>(int n, float const* x, int incx) {
      return snrm2_(&n, x, &incx);
   }
   // DNRM2
   template<>
   double host_nrm2<double>(int n, double const* x, int incx) {
      return dnrm2_(&n, x, &incx);
   }
   
   // SGEMV
   template<>
   void host_gemv<float>(enum remifa::operation trans, int m, int n, float alpha, float const* a, int lda,
                    float const* x, int incx, float beta, float *y, int incy) {
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }
   // DGEMV
   template<>
   void host_gemv<double>(enum remifa::operation trans, int m, int n, double alpha, double const* a, int lda,
                    double const* x, int incx, double beta, double *y, int incy) {
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      dgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }
   
   // _LASWP
   template<>
   void host_laswp<double>(int n, double *a, int lda, int k1, int k2, int *perm, int incx) {
      dlaswp_(&n, a, &lda, &k1, &k2, perm, &incx);
   }

   // SAXPY
   template<>
   void host_axpy<float>(int n, float a, const float *x, int incx, float *y, int incy) {
      saxpy_(&n, &a, x, &incx, y, &incy);
   }
   // DAXPY
   template<>
   void host_axpy<double>(int n, double a, const double *x, int incx, double *y, int incy) {
      daxpy_(&n, &a, x, &incx, y, &incy);
   }

   // _LANGE
   template<>
   double host_lange<double>(remifa::norm norm, int m, int n, const double *a, int lda){
      char fnorm;
      switch(norm) {
      case remifa::NORM_M:
         fnorm = 'M';
         break;
      case remifa::NORM_ONE:
         fnorm = '1';
         break;
      case remifa::NORM_INF:
         fnorm = 'I';
         break;
      case remifa::NORM_FRO:
         fnorm = 'F';
         break;
      }
      return dlange_(&fnorm, &m, &n, a, &lda);
   }

   // DPOTRF
   template<>
   int host_potrf<double>(enum remifa::fillmode uplo, int n, double* a, int lda) {
      char fuplo;
      switch(uplo) {
      case remifa::FILL_MODE_LWR: fuplo = 'L'; break;
      case remifa::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
      }
      int info;
      dpotrf_(&fuplo, &n, a, &lda, &info);
      return info;
   }
   // SPOTRF
   template<>
   int host_potrf<float>(enum remifa::fillmode uplo, int n, float* a, int lda) {
      char fuplo;
      switch(uplo) {
      case remifa::FILL_MODE_LWR: fuplo = 'L'; break;
      case remifa::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
      }
      int info;
      spotrf_(&fuplo, &n, a, &lda, &info);
      return info;
   }

   /* _TRSM */
   template <>
   void host_trsm<double>(
         enum remifa::side side, enum remifa::fillmode uplo,
         enum remifa::operation transa, enum remifa::diagonal diag,
         int m, int n,
         double alpha, const double* a, int lda,
         double* b, int ldb) {
      char fside = (side==remifa::SIDE_LEFT) ? 'L' : 'R';
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftransa = (transa==remifa::OP_N) ? 'N' : 'T';
      char fdiag = (diag==remifa::DIAG_UNIT) ? 'U' : 'N';
      dtrsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
   }
   // STRSM
   template <>
   void host_trsm<float>(
         enum remifa::side side, enum remifa::fillmode uplo,
         enum remifa::operation transa, enum remifa::diagonal diag,
         int m, int n,
         float alpha, const float* a, int lda,
         float* b, int ldb) {
      char fside = (side==remifa::SIDE_LEFT) ? 'L' : 'R';
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftransa = (transa==remifa::OP_N) ? 'N' : 'T';
      char fdiag = (diag==remifa::DIAG_UNIT) ? 'U' : 'N';
      strsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
   }

   // DGEMM
   template <>
   void host_gemm<double>(
         enum remifa::operation transa, enum remifa::operation transb,
         int m, int n, int k, double alpha, const double* a, int lda,
         const double* b, int ldb, double beta, double* c, int ldc) {
      char ftransa = (transa==remifa::OP_N) ? 'N' : 'T';
      char ftransb = (transb==remifa::OP_N) ? 'N' : 'T';
      dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }
   // SGEMM
   template <>
   void host_gemm<float>(
         enum remifa::operation transa, enum remifa::operation transb,
         int m, int n, int k, float alpha, const float * a, int lda,
         const float * b, int ldb, float beta, float* c, int ldc) {
      char ftransa = (transa==remifa::OP_N) ? 'N' : 'T';
      char ftransb = (transb==remifa::OP_N) ? 'N' : 'T';
      sgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }
   
   // DSYRK
   template <>
   void host_syrk<double>(
         enum remifa::fillmode uplo, enum remifa::operation trans,
         int n, int k, double alpha, const double* a, int lda,
         double beta, double* c, int ldc) {
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      dsyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
   }
   // SSYRK
   template <>
   void host_syrk<float>(
         enum remifa::fillmode uplo, enum remifa::operation trans,
         int n, int k, float alpha, const float* a, int lda,
         float beta, float* c, int ldc) {
      char fuplo = (uplo==remifa::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      ssyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
   }

   // DGETRF
   template <>
   int host_getrf<double>(int m, int n, double* a, int lda, int *ipiv) {
      int info;
      dgetrf_(&m, &n, a, &lda, ipiv, &info);
      return info;
   }
   template <>
   int host_getrf<float>(int m, int n, float* a, int lda, int *ipiv) {
      int info;
      sgetrf_(&m, &n, a, &lda, ipiv, &info);
      return info;
   }

   // DGEQRF
   template <>
   int host_geqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork) {
      int info;
      dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
   }
   template <>
   int host_geqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork) {
      int info;
      sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
   }

   // SORMQR
   template <>
   int host_ormqr<float>(enum remifa::side side, enum remifa::operation trans,
                         int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc,
                         float *work, int lwork) {
      int info;
      char fside = (side==remifa::SIDE_LEFT) ? 'L' : 'R';
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      sormqr_(&fside, &ftrans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
      return info;
   }

   // SGETRS
   template <>
   int host_getrs<float>(enum remifa::operation trans, int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) {
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      int info;
      sgetrs_(&ftrans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      return info;      
   }
   // DGETRS
   template <>
   int host_getrs<double>(enum remifa::operation trans, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
      char ftrans = (trans==remifa::OP_N) ? 'N' : 'T';
      int info;
      dgetrs_(&ftrans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
      return info;      
   }

   // ISAMAX
   template <> 
   cublasStatus_t dev_iamax<float>(
         cublasHandle_t handle,
         int n,
         float *x, int incx,
         int *result
         ) {
      return cublasIsamax(handle, n, x, incx, result);
   }
   // IDAMAX
   template <> 
   cublasStatus_t dev_iamax<double>(
         cublasHandle_t handle,
         int n,
         double *x, int incx,
         int *result
         ) {
      return cublasIdamax(handle, n, x, incx, result);
   }


   // SSWAP
   template <>
   cublasStatus_t dev_swap<float>(
         cublasHandle_t handle,
         int n,
         float *x, int incx,
         float *y, int incy) {
      return cublasSswap(handle, n, x, incx, y, incy);
   }
   // DSWAP
   template <>
   cublasStatus_t dev_swap<double>(
         cublasHandle_t handle,
         int n,
         double *x, int incx,
         double *y, int incy) {
      return cublasDswap(handle, n, x, incx, y, incy);
   }
   
   // SGEMV
   template <>
   cublasStatus_t dev_gemv<float>(
         cublasHandle_t handle, cublasOperation_t trans,
         int m, int n, float const* alpha, float const* a, int lda,
         float const* x, int incx, float const* beta,
         float *y, int incy) {
      return cublasSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
   }
   // DGEMV
   template <>
   cublasStatus_t dev_gemv<double>(
         cublasHandle_t handle, cublasOperation_t trans,
         int m, int n, double const* alpha, double const* a, int lda,
         double const* x, int incx, double const* beta,
         double *y, int incy) {
      return cublasDgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
   }
   
   // SGETRF BufferSize
   template<>
   cusolverStatus_t dev_getrf_buffersize<float>(cusolverDnHandle_t handle, int m, int n, float *a, int lda, int *lwork) {
      return cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, lwork);
   }
   // DGETRF BufferSize
   template<>
   cusolverStatus_t dev_getrf_buffersize<double>(cusolverDnHandle_t handle, int m, int n, double *a, int lda, int *lwork) {
      return cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, lwork);
   }

   // SGETRF
   template<>
   cusolverStatus_t dev_getrf<float>(
         cusolverDnHandle_t handle, int m, int n, float *a, int lda,
         float *workspace, int *devIpiv, int *devInfo) {
      return cusolverDnSgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
   }
   // DGETRF
   template<>
   cusolverStatus_t dev_getrf<double>(
         cusolverDnHandle_t handle, int m, int n, double *a, int lda,
         double *workspace, int *devIpiv, int *devInfo) {
      return cusolverDnDgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
   }
      
   // SPOTRF BufferSize
   template<>
   cusolverStatus_t dev_potrf_buffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, int *lwork) {
      return cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }
   // DPOTRF BufferSize
   template<>
   cusolverStatus_t dev_potrf_buffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, int *lwork) {
      return cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }

   ////////////////////////////////////////

   // SPOTRF
   template<>
   cusolverStatus_t dev_potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, float *work, int lwork, int *info) {
      return cusolverDnSpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   // DPOTRF
   template<>
   cusolverStatus_t dev_potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, double *work, int lwork, int *info) {
      return cusolverDnDpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }

   ////////////////////////////////////////

   // SSYRK
   template<>
   cublasStatus_t dev_syrk<float>(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const float *alpha, const float *a, int lda, const float *beta, float *c, int ldc) {
      return cublasSsyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
   }
   // DSYRK
   template<>
   cublasStatus_t dev_syrk<double>(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const double *alpha, const double *a, int lda, const double *beta, double *c, int ldc) {
      return cublasDsyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
   }

   ////////////////////////////////////////

   // HGEMM
   template<>
   cublasStatus_t dev_gemm<half>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const half *alpha, const half *a, int lda,
         const half *b, int ldb, const half *beta, half *c, int ldc) {
      return cublasHgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   // SGEMM
   template<>
   cublasStatus_t dev_gemm<float>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, float const* alpha, const float *a, int lda,
         float const* b, int ldb, float const* beta, float *c, int ldc) {
      return cublasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   // DGEMM
   template<>
   cublasStatus_t dev_gemm<double>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const double *alpha, const double *a, int lda,
         const double *b, int ldb, const double *beta, double *c, int ldc) {
      return cublasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }

   ////////////////////////////////////////
   
   // STRSM
   template<>
   cublasStatus_t dev_trsm<float>(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
         const float *alpha,
         const float *a, int lda,
         float *b, int ldb) {
      return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
   }
   // DTRSM
   template<>
   cublasStatus_t dev_trsm<double>(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
         const double *alpha,
         const double *a, int lda,
         double *b, int ldb) {
      return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
   }

   ////////////////////////////////////////

   // SGEQRF BufferSize
   template<>
   cusolverStatus_t dev_geqrf_buffersize<float>(
         cusolverDnHandle_t handle, int m, int n, float *a, int lda, int *lwork) {
      return cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, lwork);
   }
   // DGEQRF BufferSize
   template<>
   cusolverStatus_t dev_geqrf_buffersize<double>(
         cusolverDnHandle_t handle, int m, int n, double *a, int lda, int *lwork) {
      return cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, lwork);
   }

   // SGEQRF
   template<>
   cusolverStatus_t dev_geqrf<float>(
         cusolverDnHandle_t handle,
         int m, int n, float *a, int lda,
         float *tau, float *work, int lwork,
         int *info) {
      return cusolverDnSgeqrf(handle, m, n, a, lda, tau, work, lwork, info);
   }
   // DGEQRF
   template<>
   cusolverStatus_t dev_geqrf<double>(
         cusolverDnHandle_t handle,
         int m, int n, double *a, int lda,
         double *tau, double *work, int lwork,
         int *info) {
      return cusolverDnDgeqrf(handle, m, n, a, lda, tau, work, lwork, info);
   }

   ////////////////////////////////////////

   // SORMQR BufferSize
   template <>
   cusolverStatus_t dev_ormqr_buffersize<float>(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const float *a, int lda,
         const float *tau,
         const float *c, int ldc,
         int *lwork) {
      return cusolverDnSormqr_bufferSize(
            handle, side, trans, m, n, k,
            a, lda, tau, c, ldc,
            lwork);
   }
   // DORMQR BufferSize
   template <>
   cusolverStatus_t dev_ormqr_buffersize<double>(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const double *a, int lda,
         const double *tau,
         const double *c, int ldc,
         int *lwork) {
      return cusolverDnDormqr_bufferSize(
            handle, side, trans, m, n, k,
            a, lda, tau, c, ldc,
            lwork);
   }

   // SORMQR
   template <>
   cusolverStatus_t dev_ormqr<float>(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const float *a, int lda,
         const float *tau,
         float *c, int ldc,
         float *work, int lwork,
         int *dinfo) {
      return cusolverDnSormqr(
            handle, side, trans,
            m, n, k,
            a, lda,
            tau,
            c, ldc,
            work, lwork,
            dinfo);
   }
   // DORMQR
   template <>
   cusolverStatus_t dev_ormqr<double>(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const double *a, int lda,
         const double *tau,
         double *c, int ldc,
         double *work, int lwork,
         int *dinfo) {
      return cusolverDnDormqr(
            handle, side, trans,
            m, n, k,
            a, lda,
            tau,
            c, ldc,
            work, lwork,
            dinfo);
   }

} // end of namespace remifa
