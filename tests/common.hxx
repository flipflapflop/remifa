/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "errchk.hxx"
#include "errors.hxx"
#include "genmat.hxx"
#include "wrappers.hxx"
#include "convert.cuh"

// STD
#include <cstdio>
#include <random>
#include <chrono>

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#if defined(HAVE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#endif

namespace remifa { 
namespace tests {

   // Working precision
   enum prec {
      // Half
      FP16,
      // Single
      FP32,
      // Double
      FP64,
      // Mixed
      TC32,
      // Default
      DEFAULT
   };

   // Algorithm to be tested
   enum algo {
      /// Use routine from cuSOLVER 
      cuSOLVER,
      /// Use routine from cuSOLVER using half precision
      cuSOLVER_HP,
      /// Use the CUTLASS library
      CUTLASS,
      /// Use the CUTLASS library using half precision
      CUTLASS_WMMA_HP,
      /// REMIFA routine
      REMIFA,
      /// REMIFA Left-Looking routine
      REMIFA_LL,
      /// REMIFA Right-Looking routine
      REMIFA_RL,
      /// REMIFA routine using half-precision
      REMIFA_HP,
      /// REMIFA routine, Left-Looking, working prec FP16, panel FP16 and Update F32 
      REMIFA_MP,
      /// REMIFA routine, Left-Looking, working prec FP32, panel FP32 and Update F32 
      REMIFA_RL_TC32,
      /// REMIFA routine, Left-Looking, working prec FP16, panel TC32 and Update TC32 
      REMIFA_LL_WHFSUS,
      /// REMIFA routine, Left-Looking, working prec FP32, panel TC32 and Update TC32 
      REMIFA_LL_WSFSUS,
      /// REMIFA routine, Left-Looking, working prec FP32, panel TC16 and Update TC16 
      REMIFA_LL_WSFHUH
   };

   // Matrix type used for test
   enum mat {
      /// Matrix with given condition number
      COND,
      /// Matrix with random entries in [-1,1]
      RAND,
      /// Tri-diagonally dominant matrix
      SYMRAND,
      /// Matrix with random entries in [0,1]
      POSRAND,
      /// Diagonally dominant matrix
      DIAGDOM,
      /// Diagonally dominant matrix with non begative entries 
      DIAGDOM_POS,
      /// Tri-diagonally dominant matrix
      TRIDIAGDOM,
      /// LAPACk matrix generator LATMR
      LATMR,
      /// LAPACk matrix generator LATMS
      LATMS,
      /// Arrow head matrix
      ARROWHEAD,
      /// Arrow head matrix in [0,1]
      ARROWHEAD_POS,
      /// Arrow head matrix in [0,0.1]
      ARROWHEAD_POS_01,
      /// Arrow head matrix in [-1,1]
      ARROWHEAD_POS_NEG,
      // Sparse matrix (MatrixMarket format)
      MTX,
   };

   template<typename T, typename TConv>
   double device_conv_compwise_error(int m, T a, int lda, TConv d_aconv, int ld_d_aconv) {

      std::string context = "conv_compwise_error";
      cudaError_t cuerr;
      cublasStatus_t custat;

      // Create CUDA stream
      cudaStream_t stream;
      cuerr = cudaStreamCreate(&stream);
      remifa::cuda_check_error(cuerr, context);

      double *d_a_tilde = nullptr;
      cuerr = cudaMalloc((void**)&d_a_tilde, (std::size_t)m*lda*sizeof(double));
      remifa::cuda_check_error(cuerr, context, "d_a_tilde allocation failed");
      // Copy and convert fp16 (TConv) matrix into double precision matrix (on the GPU device)
      remifa::convert(stream, m, m, d_aconv, ld_d_aconv, d_a_tilde, lda);
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);
      // Allocate error matrix `e` and host side converted matrix
      double *e = new double[(std::size_t)m*lda];
      double *a_tilde = new double[(std::size_t)m*lda];
      custat = cublasGetMatrix(m, m, sizeof(double), d_a_tilde, lda, &a_tilde[0], lda);
      remifa::cublas_check_error(custat, context, "Failed to retreive d_a_tilde from GPU device");

      double conv_err = 0.0;
      double abs_conv_err = 0.0;
      int imax=0, jmax=0;
      for (int j=0; j<m; ++j) {
         for (int i=0; i<m; ++i) {
               
            e[j*lda+i] = std::fabs(a[j*lda+i]-a_tilde[j*lda+i]);
            
            abs_conv_err = std::max(abs_conv_err, e[j*lda+i]);

            if (std::fabs(a[j*lda+i]) > 0.0) {
               e[j*lda+i] /= std::fabs(a[j*lda+i]);
            }

            if (e[j*lda+i] > conv_err) {
               imax = i;
               jmax = j;
            }

            conv_err = std::max(conv_err, e[j*lda+i]);

         }
      }

      // printf("|a(imax,jmax)| = %le\n", std::fabs(a[jmax*lda+imax])); 
      // printf("|a~(imax,jmax)| = %le\n", std::fabs(a_tilde[jmax*lda+imax])); 
      // printf("|a(imax,jmax) - a~(imax,jmax)| = %le\n", std::fabs((double)a[jmax*lda+imax]-(double)a_tilde[jmax*lda+imax])); 
      
      printf("conv cwerr (abs) = %le\n", abs_conv_err);         
      printf("conv cwerr = %le\n", conv_err);         
      
      cuerr = cudaStreamSynchronize(stream);

      cuerr = cudaFree(d_a_tilde);
      remifa::cuda_check_error(cuerr, context);
      cuerr = cudaStreamDestroy(stream);
      remifa::cuda_check_error(cuerr, context);

      delete[] e;
      delete[] a_tilde;

      return conv_err;
   }
   
   template<typename T>
   void cast_fp16(int m, int n, T* a, int lda) {
#if defined(HAVE_CUTLASS)
      for(int j=0; j<n; ++j) {
         for(int i=0; i<m; ++i) {
            // if (i != j) {
            //    a[i*lda+i] += fabs(a[j*lda+i]);
            // }
            a[j*lda+i] = (T)cutlass::half_t(a[j*lda+i]);
         }
      }
#endif
   }
   
   template<typename T>
   void gen_latms(int m, int n, T* a, int lda, int mode) {

      T *work = new T[3*m];

      // char dist = 'U'; // 'U' => UNIFORM( 0, 1 )  ( 'U' for uniform );
      char dist = 'S';  // 'S': symmetric uniform distribution (-1, 1)

      int iseed[4] = {0,0,0,1};

      char sym = 'P'; // symmetric with non-negative eigenvalues
      // char sym = 'N'; // non symmetric

      T *d = new T[m];
      T cond = 1e2;

      // T dmax = 1.0;
      T dmax = T(m);
      // T dmax = 100;

      // Matrix band
      int ku = m; // Upper band width
      int kl = m; // Lower band width

      char pack = 'N'; // No packing
         
      int info = latms(
            m, n, dist,
            iseed, sym, d, mode,
            cond, dmax, kl,
            ku, pack, a,
            lda, work);
      
      delete[] work;
      delete[] d;
   }
   
   template<typename T>
   void gen_latmr(int m, int n, T* a, int lda, int mode) {

      // char dist = 'U'; // Uniform in [0,1]
      char dist = 'S'; // Uniform in [-1,1]
      int iseed[4] = {0,0,0,0};
      char sym = 'H'; // Generated matrix is Hermitian.
      // char sym = 'N'; // Generated matrix is non symmetric.
      T *lagen_d = new T[m];
         
      // int mode = 1; // sets d(1)=1 and d(2:n)=1.0/cond.
      // int mode = 2; // Sets d(1:n-1)=1 and d(n)=1.0/cond.
      // int mode = 3; // Sets d(i)=cond**(-(i-1)/(n-1))
      // int mode = 4; // sets d(i)=1 - (i-1)/(n-1)*(1 - 1/cond)
      // sets d to random numbers in the range ( 1/cond , 1 ) such
      // that their logarithms are uniformly distributed.
      // int mode = 5;
      T lagen_cond = 1;
      T dmax = T(m);

      // char rsign = 'T'; // diagonal entries are multiplied 1 or -1 with a probability of 0.5.
      char rsign = 'F'; // diagonal entries are unchanged
      char grade = 'N'; // there is no grading      
      T *lagen_dl = new T[m];
      int model = 1;
      T condl = 1;
      T *lagen_dr = new T[m];
      int moder = 1;
      T condr = 1;

      char pivtng = 'N'; // no pivoting permutation
      int *ipivot = nullptr;

      T sparse = 0.0;

      // Matrix band
      int ku = m; // Upper band width
      int kl = m; // Lower band width

      T anorm = -1.0; // < 0 => No scaling

      char pack = 'N'; // No packing

      int *iwork = nullptr;
      
      int info = latmr(
            m, m, dist, &iseed[0], sym, lagen_d, mode, lagen_cond,
            dmax, rsign, grade, lagen_dl, model, condl, lagen_dr, moder,
            condr, pivtng, ipivot, kl, ku, sparse, anorm, pack,
            a, lda, iwork);

      std::cout << "[gen_latmr] info = " << info << std::endl;
      
      // for(int i=0; i<m; ++i) a[i*lda+i] = std::copysign(T(m),a[i*lda+i]); 

      delete[] lagen_d;
      delete[] lagen_dl;
      delete[] lagen_dr;
   }
   
   // Generates a random dense positive definte matrix. Entries are
   // Unif[-1,1].
   template<typename T>
   void gen_mat(int m, int n, T* a, int lda) {
      /* Fill matrix with random numbers from Unif [-1.0,1.0] */
      for(int j=0; j<n; ++j)
         for(int i=0; i<m; ++i)
            a[j*lda+i] = 1.0 - (2.0*rand()) / RAND_MAX ;
   }

   template<typename T>
   void gen_randmat(int m, int n, T* a, int lda, typename std::default_random_engine::result_type seed=1u) {

      std::cout << "[gen_randmat]" << std::endl;

      std::default_random_engine generator(seed);
      std::uniform_real_distribution<T> distribution(-1.0, 1.0);
      // std::uniform_real_distribution<T> distribution(-1e3, 1e3);
      // std::uniform_real_distribution<T> distribution(-1.0/T(n), 1.0/T(n));
      // std::uniform_real_distribution<T> distribution(-1e-3, 1e-3);
      // std::uniform_real_distribution<T> distribution(0.0, 1e-2);
      
      for(int j=0; j<n; ++j)
         for(int i=0; i<m; ++i) {
            // a[j*lda+i] =  1.0;
            a[j*lda+i] =  distribution(generator);

// #if defined(HAVE_CUTLASS)
//             a[j*lda+i] = (T)cutlass::half_t(a[j*lda+i]);
// #endif

         }
   }

   template<typename T>
   void gen_randmat_sym(int n, T* a, int lda) {

      std::cout << "[gen_randmat_sym]" << std::endl;
      
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution(-1.0, 1.0);

      // Fill lower triangle with random values
      for(int j=0; j<n; ++j)
         for(int i=j; i<n; ++i)
            a[j*lda+i] = distribution(generator);

      // Symmetrize
      for(int j=0; j<n; ++j)
         for(int i=0; i<j; ++i)
            a[j*lda+i] = a[i*lda+j]; 
      
   }

   template<typename T>
   void genpos_randmat_sym(int n, T* a, int lda) {
      
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution(0.0, 1.0);

      // Fill lower triangle with random values
      for(int j=0; j<n; ++j)
         for(int i=j; i<n; ++i)
            a[j*lda+i] = distribution(generator);

      // Symmetrize
      for(int j=0; j<n; ++j)
         for(int i=0; i<j; ++i)
            a[j*lda+i] = a[i*lda+j]; 
      
   }

   // Generates a random dense positive definte matrix. Entries are
   // Unif[-1,1]. Only lower triangle is used, rest is filled with
   // NaNs.
   template<typename T>
   void gen_sym_indef(int n, T* a, int lda) {

      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution(-1.0, 1.0);

      /* Fill matrix with random numbers from Unif [-1.0,1.0] */
      for(int j=0; j<n; ++j) {
         for(int i=j; i<n; ++i) {
            a[j*lda+i] = distribution(generator) ;
            // a[j*lda+i] = (T)cutlass::half_t(distribution(generator));

         }
      }
      // Fill upper triangle with NaN
      // for(int j=0; j<n; ++j)
      //    for(int i=0; i<j; ++i)
      //       a[j*lda+i] = std::numeric_limits<T>::signaling_NaN();
      // Fill upper triangle with zeros
      // for(int j=0; j<n; ++j)
      //    for(int i=0; i<j; ++i)
      //       a[j*lda+i] = 0.0;
      // Symmetrize
      for(int j=0; j<n; ++j)
         for(int i=0; i<j; ++i)
            a[j*lda+i] = a[i*lda+j];

   }

   template<typename T>
   void gen_arrowhead(int m, int n, T* a, int lda, T min_a, T max_a) {

      std::cout << "[gen_arrowhead] sample entries in ["
                << min_a << "," << max_a << "]" << std::endl;

      // int k = m;
      // int k = (int)std::sqrt(m);
      // int k = 1000;
      // int k = 10;
      // int k = 1;
      // int k = 4;
      // int k = 8;
      // int k = 64;
      int k = 256;
      // int k = 512;
      
      std::default_random_engine generator;
      // std::uniform_int_distribution<long long int> distribution(1.0, std::pow(10.0,cond));
      // std::uniform_real_distribution<T> distribution(-1.0, 1.0);
      // std::uniform_real_distribution<T> distribution(0.0, 1e-3);
      // std::uniform_real_distribution<T> distribution(0.0, 1e-1);
      std::uniform_real_distribution<T> distribution(min_a, max_a);
      // std::uniform_real_distribution<T> distribution(1e-3, 1.0);
      // std::uniform_real_distribution<T> distribution(-T(k), T(k));
      // std::uniform_real_distribution<T> distribution(-1.0/T(k), 1.0/T(k));

      // std::bernoulli_distribution sign(0.5);
      
      for(int j=0; j<n; ++j) {
         for(int i=0; i<m; ++i) {
               a[j*lda+i] = static_cast<T>(0.0);
         }
      }
         
      int nc = 1 + ((n-1)/k);
      // std::cout << "[gen_tridiagdom] nc = " << nc << std::endl;
      for(int c=0;c<nc; ++c) {
         for(int j=c*k;j<std::min(n,(c+1)*k); ++j) {
            for(int i=c*k; i<std::min(m,(c+1)*k); ++i) {
               a[j*lda+i] = distribution(generator);
               a[i*lda+j] = distribution(generator);            

               // a[j*lda+j] += std::fabs(a[i*lda+j]);
            }
         }
      }

      for(int j=0; j<n; ++j) {
         for(int i=std::max(m-k,j); i<m; ++i) {
            // T op = (sign(generator)) ? 1.0 : -1.0;
            a[j*lda+i] = distribution(generator); 
            a[i*lda+j] = distribution(generator); 

            // a[j*lda+j] += std::fabs(a[i*lda+j]);
         }
         // a[j*lda+j] = std::sqrt(static_cast<T>(k));
         // a[j*lda+j] = 2*std::sqrt(static_cast<T>(k));
         // a[j*lda+j] = static_cast<T>(3*k);
         a[j*lda+j] = static_cast<T>(2*k);
         // a[j*lda+j] = static_cast<T>(k);
         // a[j*lda+j] += 1.0;
      }

   }

   
   // Generates a random dense positive definte matrix. Off
   // diagonal entries are Unif[-1,1]. Each diagonal entry a_ii =
   // Unif[0.1,1.1] + sum_{i!=j} |a_ij|.
   template<typename T>
   void gen_posdef(int n, T* a, int lda) {
      std::cout << "[gen_posdef]" << std::endl;
      /* Get general sym indef matrix */
      std::vector<T> b(lda*n);
      // gen_randmat_sym(n, b, lda);
      gen_randmat(n, n, (T*)&b[0], lda);

      remifa::host_gemm(
            remifa::operation::OP_N, remifa::operation::OP_T,
            n, n, n, (T)1.0,
            (T*)&b[0], lda,
            (T*)&b[0], lda,
            (T)0.0, a, lda);

      for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 1e1;

      // gen_randmat_sym(n, a, lda);
      // gen_sym_indef(n, a, lda);
      /* Make diagonally dominant */
      // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
      // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + (T)n;
      // for(int j=0; j<n; ++j)
      //    for(int i=j+1; i<n; ++i) {
      //       a[j*lda+j] += fabs(a[j*lda+i]);
      //       a[i*lda+i] += fabs(a[j*lda+i]);
      //    }
   }

   // Generates a random dense positive definte matrix. Off
   // diagonal entries are Unif[0,1]. Each diagonal entry a_ii =
   // Unif[0.1,1.1] + sum_{i!=j} |a_ij|.
   template<typename T>
   void genpos_posdef(int n, T* a, int lda) {
      std::cout << "[genpos_posdef]" << std::endl;

      std::vector<T> b(lda*n);

      // genpos_randmat(n, n, (T*)&b[0], lda);

      // remifa::host_gemm(
      //       remifa::operation::OP_T, remifa::operation::OP_N,
      //       n, n, n, (T)1.0,
      //       (T*)&b[0], lda,
      //       (T*)&b[0], lda,
      //       (T)0.0, a, lda);

      // for(int i=0; i<n; ++i) a[i*lda+i] += 1e-3;

      /* Get general sym indef matrix */
      genpos_randmat_sym(n, a, lda);
      // gen_sym_indef(n, a, lda);
      /* Make diagonally dominant */
      // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + (T)n;
      // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
      for(int j=0; j<n; ++j)
         for(int i=j+1; i<n; ++i) {
            a[j*lda+j] += fabs(a[j*lda+i]);
            a[i*lda+i] += fabs(a[j*lda+i]);
         }
   }

   /// @brief Generates a random, dense, positive-definte matrix with
   /// a condition specific condition number and eignen value
   /// distribution.
   template<typename T>
   void gen_posdef_cond(int n, T* a, int lda, T cond, T gamma) {

      // Error handling
      std::string context = "gen_posdef_cond";
      std::cout << "[" << context << "]" << " cond = " << cond << ", gamma = " << gamma << std::endl;
      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      
      std::default_random_engine generator;
      // std::uniform_int_distribution<long long int> distribution(1.0, std::pow(10.0, cond));
      // std::uniform_real_distribution<double> distribution(std::pow(10.0, -cond), 1.0);
 
      for(int j=0; j<n; ++j) {
         for(int i=0; i<n; ++i) {
            a[j*lda+i] = 0.0;
         }
      }
      // Generate diagonal matrix with eigen values 
      // T d = 1.0;
      // for(int i=0; i<n; ++i) a[i*lda+i] = (T)1.0;
      // for(int i=0; i<n; ++i) a[i*lda+i] = distribution(generator);
      // for(int i=0; i<n; ++i) a[i*lda+i] = std::pow(10.0, (T)-cond*std::pow( ((T)i)/((T)n-1), gamma ) );
      // for(int i=0; i<n; ++i) a[i*lda+i] = (T)1.0 - ((T)i/(T)(n - 1))*(1.0 - std::pow(10.0,-cond));

      // Mode #1
      // set d[0] = 1 and d[1:n - 1] = 1.0/cond
      // for(int i=0; i<n; ++i) {
      //    a[i*lda+i] = std::pow(10.0, -T(cond) );
      // }
      // a[0] = 1.0;

      // Mode #2
      // set d[0:n - 2] = 1 and d[n - 1] = 1.0/cond
      // for(int i=0; i<n; ++i) {
      //    a[i*lda+i] = 1.0;
      // }
      // a[(n-1)*lda+n-1] = std::pow(10.0, -T(cond) );

      // Mode #3
      // d[i] = cond^-i/(n - 1)
      for(int i=0; i<n; ++i) {
         a[i*lda+i] = std::pow(10.0, -T(cond)*(T(i))/T(n-1) );
      }
      
// #if defined(HAVE_CUTLASS)
//       for(int i=0; i<n; ++i) a[i*lda+i] = (T)cutlass::half_t(a[i*lda+i]);
// #endif

      double nelems_fp64 = (double)lda*n;
         
      // T *lambda = new T[lda*n];
      T *lambda = new T[(std::size_t)nelems_fp64];
      // Fill up lambda with random values
      gen_randmat(n, n, lambda, lda);

// #if defined(SPLDLT_USE_GPU)
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // cuBLAS status 

      // std::cout << "[" << context << "] allocate lambda" << std::endl;
      T *d_lambda = nullptr;
      cuerr = cudaMalloc((void**)&d_lambda, (std::size_t)n*lda*sizeof(T));      
      remifa::cuda_check_error(cuerr, context, "Failed to allocate lambda");
      // Send lambda to the GPU
      custat = cublasSetMatrix(n, n, sizeof(T), lambda, lda, d_lambda, lda);
      remifa::cublas_check_error(custat, context, "Failed to set lambda to GPU device");
      T *d_a = nullptr;
      cuerr = cudaMalloc((void**)&d_a, ((std::size_t)n*lda)*sizeof(T));
      remifa::cuda_check_error(cuerr, context, "Failed to allocate A");
      // Send A to the GPU
      custat = cublasSetMatrix(n, n, sizeof(T), a, lda, d_a, lda);
      remifa::cublas_check_error(custat, context, "Failed to set A to GPU device");

      cusolverStatus_t cusolstat;
      cusolverDnHandle_t cusolhandle;
      cusolstat = cusolverDnCreate(&cusolhandle);
      int lwork; // Workspace dimensions
      remifa::dev_geqrf_buffersize(cusolhandle, n, n, d_lambda, lda, &lwork);
      // std::cout << "[" << context << "]" <<" dev_geqrf lwork = " << lwork << std::endl;
      // Workspace on the device
      T *d_work;
      cuerr = cudaMalloc((void**)&d_work, lwork*sizeof(T));      
      remifa::cuda_check_error(cuerr, context, "Failed to allocate d_work");
      T *d_tau;
      cuerr = cudaMalloc((void**)&d_tau, n*sizeof(T));
      remifa::cuda_check_error(cuerr, context, "Failed to allocate d_tau");
      // Allocate info paramater on device
      int *dinfo;
      cuerr = cudaMalloc((void**)&dinfo, sizeof(int));
      remifa::cuda_check_error(cuerr, context, "Failed to allocate dinfo");

      start = std::chrono::high_resolution_clock::now();
      cusolstat = remifa::dev_geqrf(
            cusolhandle, n, n, d_lambda, lda,
            d_tau, d_work, lwork,
            dinfo);

      // A = Q * D
      cusolstat = remifa::dev_ormqr(
            cusolhandle,
            CUBLAS_SIDE_LEFT, CUBLAS_OP_N,
            n, n, n,
            d_lambda, lda, d_tau,
            d_a, lda,
            d_work, lwork,
            dinfo);

      // Generate a different Q for right multiplication
      // gen_randmat(n, n, lambda, lda, 2u);
      // // Send lambda to the GPU
      // custat = cublasSetMatrix(n, n, sizeof(T), lambda, lda, d_lambda, lda);
      // remifa::cublas_check_error(custat, context, "Failed to set lambda to GPU device");
      // cusolstat = remifa::dev_geqrf(
      //       cusolhandle, n, n, d_lambda, lda,
      //       d_tau, d_work, lwork,
      //       dinfo);
      
      // A = A * Q^T
      cusolstat = remifa::dev_ormqr(
            cusolhandle,
            CUBLAS_SIDE_RIGHT, CUBLAS_OP_T,
            n, n, n,
            d_lambda, lda, d_tau,
            d_a, lda,
            d_work, lwork,
            dinfo);
      end = std::chrono::high_resolution_clock::now();

      // Retrieve A on the host
      custat = cublasGetMatrix(n, n, sizeof(T), d_a, lda, a, lda);
      remifa::cublas_check_error(custat, context, "Failed to get A from GPU device");

      // Cleanup
      cusolstat = cusolverDnDestroy(cusolhandle);

      cuerr = cudaFree(dinfo);
      cuerr = cudaFree(d_tau);
      cuerr = cudaFree(d_work);
      cuerr = cudaFree(d_a); 
      cuerr = cudaFree(d_lambda);

// #else
      
//       T *tau = new T[n];
//       T worksz;
//       sylver::host_geqrf(n, n,
//                          lambda, lda,
//                          tau,
//                          &worksz, -1);
//       // std::cout << "geqrf worksz = " << worksz << std::endl;

//       int lwork = (int)worksz;
//       T *work = new T[lwork];

//       sylver::host_geqrf(n, n,
//                          lambda, lda,
//                          tau,
//                          work, lwork);
      
//       // A = Q * D
//       sylver::host_ormqr(
//             sylver::SIDE_LEFT, sylver::OP_N,
//             n, n, n,
//             lambda, lda, tau,
//             a, lda,
//             work, lwork);

//       // A = A * Q^T
//       sylver::host_ormqr(
//             sylver::SIDE_RIGHT, sylver::OP_T,
//             n, n, n,
//             lambda, lda, tau,
//             a, lda,
//             work, lwork);

//       delete[] work;
//       delete[] tau;
// #endif
      
      delete[] lambda;

//       for(int i=0; i<n; ++i) {
//          for(int j=0; j<n; ++j) {
//             // if (i != j) {
//             //    a[i*lda+i] += fabs(a[j*lda+i]);
//             // }
// #if defined(HAVE_CUTLASS)
//             a[j*lda+i] = (T)cutlass::half_t(a[j*lda+i]);
// #endif
//          }
//       }

      
      // Calculate walltime
      long ttotal =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();

      std::cout << "[" << context << "]" <<" matrix generation time (s) = " << ttotal*1e-9 << std::endl;

   }

   // template<typename T>
   // void genpos_posdef_cond(int n, T* a, int lda, T cond, T gamma) {

   //    gen_posdef_cond(n, a, lda, cond, gamma);

   //    for(int j=0; j<n; ++j) {
   //       for(int i=0; i<n; ++i) {
   //          a[j*lda+i] = fabs(a[j*lda+i]);
   //       }
   //    }
   // }

   // Generate a single rhs corresponding to solution x = 1.0
   /// @param m matrix order
   template<typename T>
   void unsym_gen_rhs(int m,  T* a, int lda, T* rhs, T* x) {
      memset(rhs, 0, m*sizeof(T));

      host_gemv(
            remifa::operation::OP_N, m, m, T(1.0), a, lda,
            x, 1, T(0.0), rhs, 1);
         
      // for (int i=0; i<m; i++) {
      //    for (int j=0; j<m; j++) {
      //       rhs[i] += a[j*lda+i] * x[j]; 
      //    }
      // }
   }

   /// @brief Generate one or more right-hand sides corresponding to
   /// soln x = 1.0.
   template<typename T>
   void gen_rhs(int n, T* a, int lda, T* rhs) {
      memset(rhs, 0, n*sizeof(T));
      for(int j=0; j<n; ++j) {
         rhs[j] += a[j*lda+j] * 1.0;
         for(int i=j+1; i<n; ++i) {
            rhs[j] += a[j*lda+i] * 1.0;
            rhs[i] += a[j*lda+i] * 1.0;
         }
      }
   }

   /// @brief Calculates forward error ||soln-x||_inf assuming x=1.0
   template<typename T>
   double forward_error(int n, int nrhs, T const* soln, int ldx) {
      /* Check scaled backwards error */
      double fwderr=0.0;
      for(int r=0; r<nrhs; ++r)
         for(int i=0; i<n; ++i) {
            double diff = std::fabs(static_cast<double>(soln[r*ldx+i] - 1.0));
            fwderr = std::max(fwderr, diff);
         }
      return fwderr;
   }


   /// @brief Caluculate componentwise error
   template<typename T>
   double compwise_error(int n, T const* a, int lda, T const* rhs, int nrhs, T const* soln, int ldsoln) {

      double cwerr = 0.0;

      // Residual resid=b-Ax
      std::vector<double> resid(n);
      // ax = |A||x|
      std::vector<double> ax(n);
      
      // Assume there is only one rhs
      int r = 0;

      for (int j=0; j<n; ++j) {
         resid[j] = (double)rhs[r*ldsoln+j];
         ax[j] = (double)0.0;
      }

      // for (int j=0; j<n; ++j) {
      //    for(int i=0; i<n; ++i) {
      //       resid[j] -= (double)a[i*lda+j] * soln[r*ldsoln+i];
      //       ax[j] += (double)fabs(a[i*lda+j]) * fabs(soln[r*ldsoln+i]);
      //    }
      // }
            
      for (int j=0; j<n; ++j) {
         resid[j] -= (double)a[j*lda+j] * (double)soln[r*ldsoln+j];
         ax[j] += (double)fabs(a[j*lda+j]) * (double)fabs(soln[r*ldsoln+j]);

         for(int i=j+1; i<n; ++i) {
            resid[j] -= (double)a[j*lda+i] * soln[r*ldsoln+i];
            resid[i] -= (double)a[j*lda+i] * soln[r*ldsoln+j];
            ax[j] += (double)fabs(a[j*lda+i]) * fabs(soln[r*ldsoln+i]);
            ax[i] += (double)fabs(a[j*lda+i]) * fabs(soln[r*ldsoln+j]);
            
         }
      }
      
      for(int i=0; i<n; ++i) {
         resid[i] /= ax[i];
         cwerr = std::max(cwerr, (double)std::fabs(resid[i])); 
      }

      return cwerr;
   }
   
   /// @brief Caluculate scaled backward error
   /// 
   /// The backward error is computed as follow:
   ///
   /// ||Ax-b||_inf / ( ||A||_1 ||x||_inf + ||b||_inf )
   ///
   template<typename T>
   double backward_error(int n, T const* a, int lda, T const* rhs, int nrhs, T const* soln, int ldsoln) {
      /* Allocate memory */
      double *resid = new double[n];
      double *rowsum = new double[n];

      /* Calculate residual vector and anorm*/
      double worstbwderr = 0.0;
      for(int r=0; r<nrhs; ++r) {
         // memcpy(resid, rhs, n*sizeof(T));
         for (int j=0; j<n; ++j)
            resid[j] = (double)rhs[r*ldsoln+j];
         memset(rowsum, 0, n*sizeof(double));
         for(int j=0; j<n; ++j) {
            resid[j] -= (double)a[j*lda+j] * soln[r*ldsoln+j];
            rowsum[j] += fabs((double)a[j*lda+j]);
            for(int i=j+1; i<n; ++i) {
               resid[j] -= (double)a[j*lda+i] * soln[r*ldsoln+i];
               resid[i] -= (double)a[j*lda+i] * soln[r*ldsoln+j];
               rowsum[j] += fabs((double)a[j*lda+i]);
               rowsum[i] += fabs((double)a[j*lda+i]);
            }
         }
         double anorm = 0.0;
         for(int i=0; i<n; ++i)
            anorm = std::max(anorm, rowsum[i]);

         /* Check scaled backwards error */
         double rhsnorm=0.0, residnorm=0.0, solnnorm=0.0;
         for(int i=0; i<n; ++i) {
            rhsnorm = std::max(rhsnorm, fabs((double)rhs[i]));
            residnorm = std::max(residnorm, fabs(resid[i]));
            if(std::isnan(resid[i])) residnorm = resid[i]; 
            solnnorm = std::max(solnnorm, fabs((double)soln[r*ldsoln+i]));
         }

         //printf("%e / %e %e %e\n", residnorm, anorm, solnnorm, rhsnorm);
         // worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm + rhsnorm));
         worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm));
         if(std::isnan(residnorm)) worstbwderr = residnorm;
      }

      /* Cleanup */
      delete[] resid;
      delete[] rowsum;

      /* Return result */
      //printf("worstbwderr = %e\n", worstbwderr);
      return worstbwderr;
   }

   template<typename T>
   double unsym_normwise_backward_error(
         int m, int n, T const* a, int lda, T const* rhs, int nrhs,
         T const* soln, int ldsoln, T const* lu, int ldlu) {

      double nwerr = static_cast<double>(0.0);

      int lda_f64 = m;
      std::vector<double> a_f64(lda_f64*n);
      std::vector<double> resid(m);
      std::vector<double> soln_f64(m);

      double zero_f64 = static_cast<double>(0.0);
      double one_f64 = static_cast<double>(1.0);
      double negone_f64 = static_cast<double>(-1.0);

      for (int j=0; j<n; ++j) {
         resid[j] = static_cast<double>(rhs[j]);
         soln_f64[j] = static_cast<double>(soln[j]);
         for (int i=0; i<m; ++i) {
            a_f64[j*lda_f64+i] = static_cast<double>(a[j*lda+i]);
         }
      }
      
      // Calculate r = Ax-b
      host_gemv(
            remifa::operation::OP_N, m, m, one_f64, &a_f64[0], lda_f64,
            &soln_f64[0], 1, negone_f64, &resid[0], 1);      

      // Calculate ||r||_inf
      // double rmax = 0.0;
      // for (int j=0; j<n; ++j) {
      //    rmax = std::max(rmax, fabs(resid[j]));
      // }
      double rnorm = 0.0;
      double anorm = 0.0;
      double xnorm = 0.0;
      
      rnorm = host_nrm2(m, &resid[0], 1);

      // Calculate ||A||_1
      anorm = host_lange(remifa::norm::NORM_ONE, m, n, &a_f64[0], lda_f64);

      // Calculate ||x||_inf
      xnorm = host_nrm2(m, &soln_f64[0], 1);

      double d = 1.0 / (anorm * xnorm);

      nwerr = rnorm;

      if (d > 0.0) {
         nwerr *=  d;
      }
      
      printf("[unsym_normwise_error] ||Ax-b||_2 = %e\n", rnorm);
      printf("[unsym_normwise_error] ||A||_1 = %e\n", anorm);
      printf("[unsym_normwise_error] ||x||_2 = %e\n", xnorm);
      
      return nwerr;
   }   
   
   template<typename T>
   void print_mat_unsym(char const* format, int n, T const* a, int lda,
                        int *rperm=nullptr) {
      for(int i=0; i<n; ++i) {
         printf("%d:", (rperm) ? rperm[i] : i);
         for(int j=0; j<n; ++j)
            printf(format, a[j*lda+i]);
         printf("\n");
      }
   }

}} // End of namespace remifa::tests
