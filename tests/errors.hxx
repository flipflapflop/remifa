/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "wrappers.hxx"

// STD
#include <vector>

namespace remifa { 
namespace tests {

/// @brief Calculate componentwise error
///
/// error = max_i (A\hat{x}-b)_i/((|A|+|\hat{L}||\hat{U}|)|x|)_i
///
/// assuming x=1
///
/// lu Computed factors L (lower triagular) and U (Upper triangular)
/// ldlu leading dimmension for lu array
template<typename T>
double unsym_compwise_error(
      int m, int n,
      T const* a, int lda,
      T const* rhs, int nrhs,
      T const* soln, int ldsoln,
      T const* lu, int ldlu) {

   double cwerr = static_cast<double>(0.0);

   int lda_f64 = m;
   std::size_t nelems_a = (std::size_t)lda_f64*n;
   std::vector<double> a_f64(nelems_a);
   std::vector<double> soln_f64(m);
   std::vector<double> resid(m);
   std::vector<double> ax(m);
   std::vector<double> ux(m);
   std::vector<double> lux(m);

   double zero_f64 = static_cast<double>(0.0);
   double one_f64 = static_cast<double>(1.0);
   double negone_f64 = static_cast<double>(-1.0);
         
   // Assume there is only one rhs
   int64_t r = 0;

   for (int64_t j=0; j<n; ++j) {
      resid[j] = static_cast<double>(rhs[r*ldsoln+j]);
      ax[j] = (double)0.0;
      ux[j] = (double)0.0;
      lux[j] = (double)0.0;
      soln_f64[j] = static_cast<double>(soln[j]);
      for(int64_t i=0; i<m; ++i) {
         a_f64[(std::size_t)j*lda_f64+i] = static_cast<double>(a[(std::size_t)j*lda+i]);
      }
   }

   // Calculate Ax-b
   host_gemv(
         remifa::operation::OP_N, m, m, one_f64, &a_f64[0], lda_f64,
         &soln_f64[0], 1, negone_f64, &resid[0], 1);


   // Calculate |A| into A
   for(int64_t j=0; j<n; ++j) {
      for (int64_t i=0; i<m; ++i) {
         a_f64[(std::size_t)j*lda_f64+i] =
            std::fabs(a_f64[(std::size_t)j*lda_f64+i]);
      }
   }

   // Calculate |x| into x
   for(int64_t j=0; j<n; ++j) {
      soln_f64[j] = std::fabs(soln_f64[j]);
   }

   // Compute |A||x|
   host_gemv(
         remifa::operation::OP_N, m, m, one_f64, &a_f64[0], lda_f64,
         &soln_f64[0], 1, zero_f64, &ax[0], 1);
      
   // double minax = (double)0.0;
   // Compute resid and |A||x| assuming x=1 
   // for (int i=0; i<m; ++i) {
   //    for(int j=0; j<n; ++j) {
   //       // resid[i] -= (double)a[j*lda+i] * (double)soln[r*ldsoln+j];
   //       // ax[i] += (double)fabs(a[j*lda+i]); // Assuming x=1
   //       // ax[i] += (double)fabs(a[j*lda+i]) * (double)fabs(soln[r*ldsoln+j]);
   //    }
   //    if (i == 0)
   //       minax = ax[i];
   //    minax = std::min(minax, ax[i]);
   // }
   // printf("[unsym_compwise_error] minax = %e\n", minax);

   // Calculate |L||U| into A
   for(int j=0; j<n; ++j) {
      for (int i=0; i<m; ++i) {
         a_f64[(std::size_t)j*lda_f64+i] =
            std::fabs(static_cast<double>(lu[(std::size_t)j*ldlu+i]));
      }
      lux[j] = std::fabs(soln_f64[j]);
   }
      
   // Compute |\hat{L}||\hat{U}||x|
      
   // Compute |\hat{U}||x|
   host_trmv(
         remifa::FILL_MODE_UPR, remifa::operation::OP_N,
         remifa::diagonal::DIAG_NON_UNIT, n, &a_f64[0], lda_f64, &lux[0], 1);

      
   // for (int i=0; i<m; ++i) {
   //    for (int j=i; j<n; ++j) {
   //       // ux[i] += (double)std::fabs(lu[j*lda+i]); // assuming x=1
   //       ux[i] += (double)std::fabs(lu[j*lda+i]) * (double)fabs(soln[r*ldsoln+j]);
   //    }
   // }

   // Compute |\hat{L}|(|\hat{U}||x|)
   host_trmv(
         remifa::FILL_MODE_LWR, remifa::operation::OP_N,
         remifa::diagonal::DIAG_UNIT, n, &a_f64[0], lda_f64, &lux[0], 1);

         
   // for (int i=0; i<m; ++i) {
   //    lux[i] += ux[i];
   //    for (int j=0; j<i; ++j) {
   //       lux[i] += (double)std::fabs(lu[j*lda+i]) * (double)ux[j];
   //    }
   // }

   double minlux = (double)lux[0];
   double maxlux = (double)0.0;
   int maxidx = 0;
   double maxresid = (double)0.0;

   double cwerr_ax = static_cast<double>(0.0);

   for(int i=0; i<m; ++i) {
      // resid[i] /= (ax[i]);
      resid[i] = std::fabs(resid[i]);

      maxresid = std::max(maxresid, resid[i]);

      double d1 = 1.0/(ax[i]);
      if (d1 > 0.0) {
         double r_abs_ax = std::fabs(resid[i]) * d1;
         cwerr_ax = std::max(cwerr_ax, r_abs_ax);
      }
         
      double d2 = (double) 1.0 / (ax[i]+lux[i]);
      if (d2 > 0.0) {
         resid[i] = resid[i] * d2;
      }
      // printf("[unsym_compwise_error] resid[%d] = %e\n", i, std::fabs(resid[i]));
      // resid[i] /= (lux[i]);
      if (std::fabs(resid[i]) > cwerr) {
         maxidx = i;
      }
      cwerr = std::max(cwerr, (double)std::fabs(resid[i])); 
      // cwerr = std::min(cwerr, (double)std::fabs(resid[i]));
      // minlux = std::min(minlux, (double)std::fabs(lux[i]));
      // maxlux = std::max(maxlux, (double)std::fabs(lux[i]));
      minlux = std::min(minlux, (double)std::fabs(lux[i]));
      maxlux = std::max(maxlux, (double)std::fabs(lux[i]));

   }

   // printf("[unsym_compwise_error] minlux = %e\n", minlux);
   // printf("[unsym_compwise_error] maxlux = %e\n", maxlux);
   // printf("[unsym_compwise_error] maxidx = %d\n", maxidx);

   printf("[unsym_compwise_error] r_inf = %e\n", maxresid);
   printf("[unsym_compwise_error] max lux[i] = %e\n", maxlux);
   printf("[unsym_compwise_error] min lux[i] = %e\n", minlux);

   printf("[unsym_compwise_error] max_i |Ax-b|_i / (|A||x|)_i = %e\n", cwerr_ax);

   return cwerr;
}

// Calculate scaled backward error ||Ax-b|| / ( ||A|| ||x|| + ||b|| ).
// All norms are infinity norms execpt for matrix A which is one norm.
template<typename T>
double unsym_backward_error(
      int m, int n, T const* a, int lda, T const* rhs, int nrhs,
      T const* soln, int ldsoln) {
         
   int const ldrhs = m; // Assume ldrhs = m
         
   /* Allocate memory */
   std::vector<double> resid(m, static_cast<double>(0.0));
   std::vector<double> rowsum(n, static_cast<double>(0.0));
   // double *resid = new double[m];
   // double *rowsum = new double[n];

   // memset(rowsum, 0, n*sizeof(double));
   for(int j=0; j<n; ++j) {
      for(int i=0; i<m; ++i) {
         rowsum[j] += static_cast<double>(fabs(a[j*lda+i]));
      }
   }         
   double anorm = 0.0;
   for(int j=0; j<n; ++j)
      anorm = std::max(anorm, rowsum[j]);

   // std::cout << "anorm = " << anorm << std::endl;

   /* Calculate residual vector and anorm */
   double worstbwderr = 0.0;
   for(int r=0; r<nrhs; ++r) {
      // memcpy(resid, &rhs[r*ldrhs], m*sizeof(T));
      for(int i=0; i<m; ++i)
         resid[i] = static_cast<double>(rhs[r*ldsoln+i]);
            
      for(int j=0; j<n; ++j) {
         for(int i=0; i<m; ++i) {
            resid[i] -= static_cast<double>(a[j*lda+i])*static_cast<double>(soln[r*ldsoln+j]); 
         }
      }

      /* Check scaled backwards error */
      double rhsnorm=0.0, residnorm=0.0, solnnorm=0.0;
      for(int i=0; i<m; ++i) {
         // Calculate max norms
         rhsnorm = std::max(rhsnorm, fabs(static_cast<double>(rhs[r*ldrhs+i])));
         residnorm = std::max(residnorm, fabs(resid[i]));
         if(std::isnan(resid[i])) residnorm = resid[i]; 
      }

      for(int i=0; i<n; ++i) {
         solnnorm = std::max(solnnorm, fabs(static_cast<double>(soln[i+r*ldsoln])));
      }
            
      worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm + rhsnorm));
      // worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm));
      if(std::isnan(residnorm)) worstbwderr = residnorm;
   }

   /* Cleanup */
   // delete[] resid;
   // delete[] rowsum;

   // Return largest error
   return worstbwderr;

}

template<typename T>
double factor_backward_error(
      int m, int n, T const* a, int lda,
      T const* l
      ) {


   std::size_t nelems = (std::size_t)m*n;

   double *e = new double[nelems];
   double *denom = new double[nelems];
   double *l_hat = new double[nelems];
   double *u_hat = new double[nelems];

   // std::memset(e, 0, nelems*sizeof(double));
   // std::memset(denom, 0, nelems*sizeof(double));
   std::memset(l_hat, 0, nelems*sizeof(double));
   std::memset(u_hat, 0, nelems*sizeof(double));
         
   for (int j=0; j<m; ++j) {
      // Put computed U factor entries into U_hat
      for (int i=0; i<=j; ++i) {
         u_hat[i+j*lda] = l[i+j*lda];
         e[i+j*lda] = a[i+j*lda];
      }
      // Put ones on L_hat diagonal
      l_hat[j+j*lda] = static_cast<double>(1.0);
      // Put computed L factor entries into L_hat
      for (int i=j+1; i<m; ++i) {
         l_hat[i+j*lda] = l[i+j*lda]; 
         e[i+j*lda] = a[i+j*lda];
      }
   }

   remifa::host_gemm(
         remifa::operation::OP_N, remifa::operation::OP_N,
         m, m, m,
         static_cast<double>(-1.0),
         l_hat, lda, u_hat, lda,
         static_cast<double>(1.0),
         e, lda);

   // Calculte absolute value of L_hat and U_hat
   for (int j=0; j<m; ++j) {
      for (int i=0; i<m; ++i) {
         u_hat[i+j*lda] = std::fabs(u_hat[i+j*lda]);
         l_hat[i+j*lda] = std::fabs(l_hat[i+j*lda]);
      }
   }
         
   remifa::host_gemm(
         remifa::operation::OP_N, remifa::operation::OP_N,
         m, m, m,
         static_cast<double>(1.0),
         l_hat, lda, u_hat, lda,
         static_cast<double>(0.0),
         denom, lda);

   double factor_err = 0.0;
   double abs_factor_err = 0.0;

   double min_lu_hat = denom[0];
   double max_lu_hat = 0;

   for (int j=0; j<m; ++j) {
      for (int i=0; i<m; ++i) {
         // printf("denom = %le\n", denom[i+j*lda]);

         e[i+j*lda] = std::fabs(e[i+j*lda]);
         abs_factor_err = std::max(abs_factor_err, std::fabs(e[i+j*lda]));

         // double d = 1.0 / denom[i+j*lda];
                     
         if(denom[i+j*lda]>0.0) {
            e[i+j*lda] /= denom[i+j*lda];
            min_lu_hat = std::min(min_lu_hat, denom[i+j*lda]);
            max_lu_hat = std::max(min_lu_hat, denom[i+j*lda]);
         }
         // e[i+j*lda] = std::fabs(e[i+j*lda]);
         factor_err = std::max(factor_err, e[i+j*lda]);
      }
   }

   printf("factor cwerr (abs) = %le\n", abs_factor_err);         
   printf("factor cwerr = %le\n", factor_err);         

   printf("min_lu_hat = %le\n", min_lu_hat);         
   printf("max_lu_hat = %le\n", max_lu_hat);         

   delete[] l_hat;
   delete[] u_hat;
   delete[] e;
   delete[] denom;

   return factor_err;
}
   
}} // End of namespace remifa::tests
