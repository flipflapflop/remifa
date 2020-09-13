/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "wrappers.hxx"
#include "utils.hxx"
#include "common.hxx"
#include "errors.hxx"
#include "lu_nopiv.hxx"
#include "lu_nopiv_ll.hxx"
// #include "solve.hxx"

// STD
#include <vector>
#include <cstdint>

#include "convert.cuh"
#include <cuda_fp16.h>

namespace remifa {
namespace tests {

/// @param algo Algo to be tested
/// @param usetc Set to false in order to disable tensor cores
template<typename T, bool use_cutlass=false>
int lu_test(
      int m,
      enum algo algo,
      enum remifa::tests::prec work_prec, // Working precision
      enum remifa::compute_type in_upd, // Inner update compute type
      enum remifa::compute_type out_upd, // Outer update compute type
      enum remifa::tests::prec in_pnl, // Outer panel precision
      enum remifa::tests::prec out_pnl, // Outer panel precision
      enum remifa::tests::prec in_fac, // Outer factor precision
      enum remifa::tests::prec out_fac, // Inner factor precision
      // bool usetc,
      enum mat mat, T cond, T gamma,
      bool check,
      remifa::tests::TestOpts options) {

   std::string context = "lu_test";
   bool failed = false;

   int lda = remifa::align_lda<T>(m);
   // int lda = remifa::align_lda<double>(m);
   // int lda = m;
   
   // Number of entries in matrix A
   // std::size_t nelems = (std::size_t)lda*m;
   // double nelems_fp64 = (double)lda*m;
   
   int const nb = options.nb;
   std::cout << "[" << context << "] nb = " << nb << std::endl;

   bool split_k = options.split_k;
   int split_k_slices = options.split_k_slices;
   // GPU workspace for SplitK Gemm implementation
   uint8_t *workspace = nullptr;

   if (split_k) {
      
      std::cout << "[" << context << "] slices = " << split_k_slices << std::endl;
            
      if (split_k_slices <= 0) {
         throw std::runtime_error("Number of SplitK slices should be > 0");
      }
      else {
         // Allocate workspace for SplitK Gemm implementation

         // Find out worspace size depending on outer update 
         size_t workspace_size = 0;

#if defined(HAVE_CUTLASS)
         
         if (remifa::compute_type::FP32 == options.out_upd) {
            workspace_size = gemm_workspace_size
               <CutlassSplitKImpl, /*Op=*/remifa::FP,
                /*ElementOutput=*/float, /*ElementAccumulator=*/float>
               (m, nb, m, split_k_slices);
         }
         else if (remifa::compute_type::FP16 == options.out_upd) {
            workspace_size = gemm_workspace_size
               <CutlassSplitKImpl, /*Op=*/remifa::FP,
                /*ElementOutput=*/half, /*ElementAccumulator=*/half>
               (m, nb, m, split_k_slices);
         }
         else if (remifa::compute_type::TC16 == options.out_upd) {
            workspace_size = gemm_workspace_size
               <CutlassSplitKImpl, /*Op=*/remifa::TC,
                /*ElementOutput=*/half, /*ElementAccumulator=*/half>
               (m, nb, m, split_k_slices);
         }
         else if (remifa::compute_type::TC32 == options.out_upd) {
            if (options.out_pnl==tests::prec::FP32) {
               workspace_size = gemm_workspace_size
                  <CutlassSplitKImpl, /*Op=*/remifa::TC,
                   /*ElementOutput=*/float, /*ElementAccumulator=*/float>
                  (m, nb, m, split_k_slices);
            }
            else if (options.out_pnl==tests::prec::FP16) {
               workspace_size = gemm_workspace_size
                  <CutlassSplitKImpl, /*Op=*/remifa::TC,
                   /*ElementOutput=*/half, /*ElementAccumulator=*/float>
                  (m, nb, m, split_k_slices);
            }
         }
#endif
         
         if (workspace_size > 0) {
            
            std::cout << "[" << context << "] workspace_size (MB) = " << workspace_size / (1024.0*1024.0) << std::endl;

            cudaError_t cuda_error = cudaMalloc((void**)&workspace, workspace_size*sizeof(uint8_t));      
         }
         else {
            throw std::runtime_error("Workspace size is 0");
         }

      }
   }

   T *a = nullptr;
   T* l = nullptr;
   T* b = nullptr;

   // std::vector<T> a;
   // std::vector<T> l;
   // std::vector<T> b;

   // try {
   {
      // std::uint64_t nelems = lda*m;
      // std::uint64_t a_mem_b = nelems*sizeof(T);

      // printf("nelems = %u\n", nelems);

      // double d = (double)1.0 / ((double)1024.0*1024.0*1024.0);
      // double a_mem_gb = (static_cast<double>((std::uint64_t)nelems));
      // a_mem_gb *= sizeof(T);
      // std::size_t a_mem = (1024*1024*1024);
      // double a_mem_fp64 = (double)a_mem; 
         
      // printf("sizeof size_t = %d\n", sizeof(std::size_t));
      // printf("std::vector<T>::size_type = %d\n",
      // sizeof(typename std::vector<T>::size_type));
      // printf("a_mem_gb = %u\n", a_mem_gb);
      // printf("a_mem = %u\n", a_mem);
      // printf("a_mem_fp64 = %f\n", a_mem_fp64);

      // printf("nelems_fp64 = %f\n", nelems_fp64);

      // std::int64_t nelems_int64 = nelems;
      // std::int64_t nelems_int64 = lda*m;         
      // printf("nelems_int64 = %d\n", nelems_int64);

      // typename std::vector<T>::size_type nelems_size_type = lda*m; 
      // printf("nelems_size_type = %u\n", nelems_size_type);

      // long long nelems_long_long = lda*m;
      // printf("nelems_long_long = %lld\n", nelems_long_long);
         
      // std::cout << "[" << context << "]"
      // << " A memory footprint (GB) = "
      // << (nelems*sizeof(T)) / (1024*1024*1024) 
      // << std::endl;
      // a.resize(nelems, 0.0);
      // a.resize(nelems_size_type, 0.0);
 
      // a = new T[nelems];
      // a = new T[nelems_int64];
      // a = new T[(std::size_t)nelems_long_long];
      a = new T[(std::size_t)lda*(std::size_t)m];

      // char *c = new char[nelems];

   }
   // catch (std::bad_alloc& ba) {
   //    std::cout << context << " Unable to allocate A" << std::endl;
   //    return 1;
   // }
   // catch (const std::exception& e) { // reference to the base of a polymorphic object
   //    std::cout << "[" << context << "]"
   //              << " Error resizing A: " << e.what() << std::endl;
   //    return 1;
   // }      

   // Generate test matrix
   switch(mat) {
   case remifa::tests::mat::RAND:
      remifa::tests::gen_randmat(m, m, (T*)&a[0], lda);
      break;
   case remifa::tests::mat::SYMRAND:
      remifa::tests::gen_randmat_sym(m, (T*)&a[0], lda);
      // remifa::tests::gen_posdef(m, (T*)&a[0], lda);
      break;
   case remifa::tests::mat::DIAGDOM:
      remifa::tests::gen_diagdom(m, m, (T*)&a[0], lda);
      break;
   case remifa::tests::mat::DIAGDOM_POS:
      remifa::tests::gen_diagdom_pos(m, m, (T*)&a[0], lda, T(0.0), T(1.0));
      // remifa::tests::gen_diagdom_pos(m, m, (T*)&a[0], lda, T(0.5), T(1.0));
      break;
   case remifa::tests::mat::POSRAND:
      remifa::tests::genpos_randmat(m, m, (T*)&a[0], lda);
      break;
   case remifa::tests::mat::COND:
      remifa::tests::gen_posdef_cond(m, (T*)&a[0], lda, cond, gamma);
      break;
   case remifa::tests::mat::ARROWHEAD:
   case remifa::tests::mat::ARROWHEAD_POS:
      remifa::tests::gen_arrowhead(m, m, (T*)&a[0], lda, T(0.0), T(1.0));
      // remifa::tests::gen_arrowhead(m, m, (T*)&a[0], lda, T(0.0), T(0.001));
      break;
   case remifa::tests::mat::ARROWHEAD_POS_NEG:
      remifa::tests::gen_arrowhead(m, m, (T*)&a[0], lda, T(-1.0), T(1.0));
      break;
   case remifa::tests::mat::LATMR:
      remifa::tests::gen_latmr(m, m, (T*)&a[0], lda, options.latmr_mode);
      break;
   case remifa::tests::mat::LATMS:
      remifa::tests::gen_latms(m, m, (T*)&a[0], lda, options.latms_mode);
      break;
   case remifa::tests::mat::MTX:
      read_mtx(options.matfile, m, &a, lda);
      break;
   default:
      std::cout << "[" << context << "]" << " Matrix generator NOT available" << std::endl;
      exit(1);
      break;
   }

   std::cout << "[" << context << "] m = " << m << std::endl;
   std::cout << "[" << context << "] lda = " << lda << std::endl;

   if (options.cast_f16) {
      cast_fp16(m, m, a, lda);
   }

   try {
      // l.resize(lda*m, 0.0);
      // l = new T[nelems];
      l = new T[(std::size_t)lda*(std::size_t)m];
   }
   catch (std::bad_alloc& ba) {
      std::cout << "[" << context << "]"
                << " Unable to allocate L" << std::endl;
      return 1;
   }
   catch (const std::exception& e) { // reference to the base of a polymorphic object
      std::cout << "[" << context << "]"
                << "Error resizing L: " << e.what() << std::endl;
      return 1;
   }

   try {
      // b.resize(m, 0.0);
      b = new T[m];
   }
   catch (std::bad_alloc& ba) {
      std::cout << context << " Unable to allocate b" << std::endl;
      return 1;
   }
   catch (const std::exception& e) { // reference to the base of a polymorphic object
      std::cout << "Error resizing b: " << e.what() << std::endl;
      return 1;
   }
   
   // Pivoting sequence array
   std::vector<int> ipiv(m);

   // remifa::tests::print_mat_unsym("%8.2f", m, &a[0], lda);

   // T *atmp = new T[(std::size_t)nelems_fp64];
   // std::memcpy(atmp, a, ((std::size_t)nelems_fp64)*sizeof(T));
   // host_getrf(m, m, atmp, lda, &ipiv[0]);
   // lapmr(true, m, m, a, lda, &ipiv[0]);
   // delete[] atmp;

   // std::cout << "[" << context << "]" << " ipiv: ";
   // for(int i=0; i<m; ++i)
   //    std::cout << ipiv[i] << " ";        
   // std::cout << std::endl;
   // std::cout << "[" << context << "]" << " Reordered matrix:" << std::endl;
      
   // remifa::tests::print_mat_unsym("%6.2f", m, &a[0], lda);

   if (options.scale) {
      
      T *r = new T[m];
      T *c = new T[m];
      T rowcnd;
      T colcnd;
      T amax;
      
      // int info = geequ(m, m, (T*)&a[0], lda,
      //       r, c, &rowcnd, &colcnd, &amax);
      int info = geequb(m, m, (T*)&a[0], lda,
                        r, c, &rowcnd, &colcnd, &amax);
      std::cout << "[" << context << "]" << " geequ info = " << info << std::endl;
      std::cout << "[" << context << "]" << " geequ amax = " << amax << std::endl;

      char equed = 'B';      
      laqge(m, m, (T*)&a[0], lda, r, c, rowcnd, colcnd, amax, &equed);

      std::cout << "[" << context << "]" << " laqge equed = " << equed << std::endl;

      
      delete[] r;
      delete[] c;
   }
   
   // remifa::tests::print_mat_unsym("%6.2f", m, &a[0], lda);
      
   // std::vector<T> x_zero(m, 1.0);
   std::vector<T> x_zero(m, 0.0);
   // x_zero[m-1] = 1.0;
      
   std::default_random_engine generator;
   // // std::uniform_int_distribution<long long int> distribution(1.0, std::pow(10.0,cond));
   // std::uniform_real_distribution<T> distribution(-1.0 / double(m), 1.0 / double(m));
   std::uniform_real_distribution<T> distribution(-1.0, 1.0);
   // std::uniform_real_distribution<T> distribution(-T(m), T(m));

   for (int i=0;i<m;++i) {
      // if (i%2 == 0) {

      x_zero[i] = 1.0;
         
      // x_zero[i] = 1.0 / double(m-i);

      // }
      // else {
      // x_zero[i] = -1.0 / double(m-i);
      // }
      // x_zero[i] = distribution(generator);
         
      // double d = 1.0 / double(m);
      // x_zero[i] = d;
      // x_zero[i] = 1.0 / double(i+1);

   }
   // x_zero[0] = 1.0;
   // x_zero[m-1] = 1.0;

   if (options.cast_f16) {
      cast_fp16(m, 1, &x_zero[0], lda);
   }
      
   // Generate a RHS based on x=1, b=Ax
   remifa::tests::unsym_gen_rhs(m, &a[0], lda, &b[0], &x_zero[0]);

   // if (options.cast_f16) {
   // cast_fp16(m, 1, &b[0], lda);
   // }

   // std::cout << "rhs = ";
   // for (int i=0;i<m;++i) {
   //    std::cout << b[i] << " ";
   // }
   // std::cout << std::endl;
      
   // Copy a into l which is used to store the factors
   // l = a;
   std::memcpy(&l[0], &a[0], (std::size_t)lda*(std::size_t)m*sizeof(T));
      
   // Error managment
   cudaError_t cuerr;
   cublasStatus_t custat;

   // CUDA stream
   cudaStream_t stream;
   cuerr = cudaStreamCreate(&stream);
   remifa::cuda_check_error(cuerr, context);

   ////////
   // fp32 factor entries 
   T *d_l = nullptr;
   // fp16 factor entries 
   half *d_l_f16 = nullptr;

   // Select math mode
   // if (usetc) custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
   // else       custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);            

   // Timers
   std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      
   // cuSOLVER algo (full-precision)
   if (algo == remifa::tests::cuSOLVER) {
         
      // Initialize cuSOLVER
      cusolverStatus_t cusolstat;
      cusolverDnHandle_t cusolhandle;

      cusolstat = cusolverDnCreate(&cusolhandle);
      cusolverDnSetStream(cusolhandle, stream);

      int info;
      int *d_inform; // Device side status
      cuerr = cudaMalloc((void**)&d_inform, sizeof(int));
      remifa::cuda_check_error(cuerr, context, "d_inform allocatation failed");

      // Allocate memory on the device
      cuerr = cudaMalloc((void**)&d_l, (std::size_t)m*lda*sizeof(T));
      remifa::cuda_check_error(cuerr, context, "d_l allocatation failed");
      
      // Send matrix to device
      custat = cublasSetMatrix(m, m, sizeof(T), &l[0], lda, d_l, lda);
      // cudaMemcpy(d_l, l, lda*m*sizeof(T), cudaMemcpyHostToDevice);
      remifa::cublas_check_error(custat, context);

      // Setup workspace
      T *d_work = nullptr;
      int worksz = 0; // Workspace size
      cusolstat = remifa::dev_getrf_buffersize(cusolhandle, m, m, d_l, lda, &worksz);
      cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T)); 
      // Setup ipiv array
      int *d_ipiv = nullptr;
      cuerr = cudaMalloc((void**)&d_ipiv, m*sizeof(int)); 
      remifa::cuda_check_error(cuerr, context, "d_ipiv allocatation failed");
         
      start = std::chrono::high_resolution_clock::now();
      // Launch cuSOLVER getrf
      cusolstat = remifa::dev_getrf(
            cusolhandle, m, m, d_l, lda,
            d_work,
            // d_ipiv,
            NULL,
            d_inform);
      if (cusolstat != CUSOLVER_STATUS_SUCCESS) {
         std::cout << "[" << context << "][error] Failed to launch dev_getrf"
                   << std::endl;
         std::exit(1);         
      }
         
      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);

      end = std::chrono::high_resolution_clock::now();
      // exit(0);

      // Get matrix into host memory      
      custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, &l[0], lda);
      // cudaMemcpy(l, d_l, lda*m*sizeof(T), cudaMemcpyDeviceToHost);
      remifa::cublas_check_error(custat, context);
      // Get pivoting sequence into host memory
      // cuerr = cudaMemcpy(&ipiv[0], d_ipiv, m*sizeof(int), cudaMemcpyDeviceToHost);
      for(int i=0; i<m; ++i) ipiv[i] = i+1; // ipiv must be 1-indexed
      // cudaMemcpy(l, d_l, lda*m*sizeof(T), cudaMemcpyDeviceToHost);
      // remifa::cuda_check_error(cuerr, context);
      // Get info
      cuerr = cudaMemcpy(&info, d_inform, sizeof(int), cudaMemcpyDeviceToHost);
      // Cleanup
      cusolstat = cusolverDnDestroy(cusolhandle);
      cuerr = cudaFree(d_inform);
      remifa::cuda_check_error(cuerr, context);
      cuerr = cudaFree(d_work);
      remifa::cuda_check_error(cuerr, context);

      if (info != 0) {
         std::cout << "[" << context << "][error] Failed to execute dev_getrf "
                   << "(" << info << ")" << std::endl;
         std::exit(1);         
      }
         
   }

   else if (algo == remifa::tests::REMIFA_LL) {
      // Left-looking REMIFA algorithm

      cublasStatus_t custat; // CuBLAS status
      cublasHandle_t cuhandle;

      int info;
      int *d_inform; // Device side status
      cuerr = cudaMalloc((void**)&d_inform, sizeof(int));
      remifa::cuda_check_error(cuerr, context, "d_inform allocation failed");

      // Initialize factorization
      custat = cublasCreate(&cuhandle);
      remifa::cublas_check_error(custat, context, "Failed to create CUDA handle");
      custat = cublasSetStream(cuhandle, stream);
      remifa::cublas_check_error(custat, context, "Failed to associate CUDA handle with CUDA stream");

      // Set math mode to default: use tensor cores only if
      // explicitly requested
      custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);            
      remifa::cublas_check_error(custat, context);

      // Allocate memory on the device
      cuerr = cudaMalloc((void**)&d_l, (std::size_t)m*lda*sizeof(T));
      remifa::cuda_check_error(cuerr, context, "d_l allocation failed");
      
      // Send matrix to device
      custat = cublasSetMatrix(m, m, sizeof(T), &l[0], lda, d_l, lda);
      // cudaMemcpy(d_l, l, lda*m*sizeof(T), cudaMemcpyHostToDevice);
      remifa::cublas_check_error(custat, context);

      // If working precision is different from the one used to
      // generate the original matrix, then we need to allocate a
      // new buffer for converting the original matrix to the
      // working precision

      // W = fp16
      //
              
      // if (remifa::tests::prec::FP16 == work_prec) {
      // Allocate matrix with working precision
      cuerr = cudaMalloc((void**)&d_l_f16, (std::size_t)m*lda*sizeof(half));
      remifa::cuda_check_error(cuerr, context, "d_l_f16 allocation failed");
      // Convert original matrix to working precision
      remifa::convert(stream, m, m, d_l, lda, d_l_f16, lda);
      // remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);

      // }

      // Compute conversion error

      if (options.check_conv) {
         device_conv_compwise_error(m, a, lda, d_l_f16, lda);
      }
                  
      // Copy and convert back to original matrix
      // remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);

      // Start measuring the time to perform the factorization
      start = std::chrono::high_resolution_clock::now();
         
      //
      // Launch factorization on device
      //
         
      if (remifa::tests::prec::FP16 == work_prec) {
         // W=fp16

         assert(nullptr != d_l_f16);

         if(remifa::compute_type::FP16 == out_upd) {
            // W=fp16 and O=fp16
            if (use_cutlass) {
               if (split_k) {
                  lu_nopiv_ll_cutlass_splitk_f16(
                        cuhandle, m, m, nb, d_l_f16, lda, d_inform, split_k_slices, workspace);
               }
               else {
                  lu_nopiv_ll_cutlass_f16(cuhandle, m, m, nb, d_l_f16, lda, d_inform);
               }
            }
            else {
               lu_nopiv_ll_f16(cuhandle, m, m, nb, d_l_f16, lda, d_inform);
            }

         }
         else if(remifa::compute_type::TC16 == out_upd) {
            // W=fp16 and O=TC16

            if (remifa::compute_type::FP16 == in_upd) {
               // W=fp16 and O=TC16, OP=OF=fp16 and I=fp16, IP=IF=fp16
               if (use_cutlass) {
                  if (split_k) {
                     remifa::lu_nopiv_ll_cutlass_splitk_f16_f16xtc16(
                           cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                           split_k_slices, workspace);
                  }
                  else {
                     remifa::lu_nopiv_ll_cutlass_f16_f16xtc16(cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                  }
               }
               else {
                  remifa::lu_nopiv_ll_f16_f16xtc16(cuhandle, m, m, nb, d_l_f16, lda, d_inform);
               }
            }
            else if (remifa::compute_type::TC16 == in_upd) {
               // W=fp16 and O=TC16, OP=OF=fp16 and I=TC16, IP=IF=fp16
               if (use_cutlass) {
                  remifa::lu_nopiv_ll_cutlass_f16_tc16xtc16(
                        cuhandle, m, m, nb, d_l_f16, lda, d_inform);
               }
               else {
                  remifa::lu_nopiv_ll_f16_tc16xtc16(
                        cuhandle, m, m, nb, d_l_f16, lda, d_inform);
               }
            }
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }

         } // W=fp16 and O=TC16
         else if(remifa::compute_type::TC32 == out_upd) {
            // W=fp16 and O=TC32

            if (remifa::compute_type::FP16 == in_upd) {
               // W=fp16 and I=fp16, O=TC32

               if (remifa::tests::prec::FP32 == out_pnl) {

                  if (use_cutlass) {
                     if (split_k) {
                        lu_nopiv_ll_cutlass_splitk_f16_f16xtc32_fp32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                              split_k_slices, workspace);
                     }
                     else {
                        lu_nopiv_ll_cutlass_f16_f16xtc32_fp32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                     }
                  }
                  else {
                     lu_nopiv_ll_f16_f16xtc32_fp32(
                           cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                  }

                  // remifa::lu_nopiv_ll
                  //    <half, // Working prec
                  //     remifa::compute_type::FP16, // Inner update compute type
                  //     remifa::compute_type::TC32, // Outer update compute type
                  //     /*Inner blocking*/8,
                  //     /*OP*/float
                  //     >
                  //    (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
               }
               else {

                  if (use_cutlass) {
                     if (split_k) {
                        lu_nopiv_ll_cutlass_splitk_f16_f16xtc32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                              split_k_slices, workspace);
                     }
                     else {
                        lu_nopiv_ll_cutlass_f16_f16xtc32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                     }
                  }
                  else {
                     lu_nopiv_ll_f16_f16xtc32(
                           cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                  }
               }
            }
            else if (remifa::compute_type::FP32 == in_upd) {
               // W=fp16, O=TC32, I=fp32

               if ((remifa::tests::prec::FP32 == out_pnl) &&
                   (remifa::tests::prec::FP32 == out_fac) &&
                   (remifa::tests::prec::FP32 == in_fac)) {
                  // W=fp16, O=TC32, OP=OF=fp32 and I=IP=IF=fp32
                  if (use_cutlass) {
                     if (split_k) {
                        lu_nopiv_ll_cutlass_splitk_f16_f32xtc32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                              split_k_slices, workspace);
                     }
                     else {
                        lu_nopiv_ll_cutlass_f16_f32xtc32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                     }
                  }
                  else {
                     lu_nopiv_ll_f16_f32xt32(
                           cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                  }
                     
               }
            }
            else if (remifa::compute_type::TC16 == in_upd) {
               //
               // W=fp16 and I=TC16, O=TC32

               if (remifa::tests::prec::FP32 == out_pnl) {

                  //
                  // W=fp16 and I=TC16, O=TC32 and OP=W=FP32
                  remifa::lu_nopiv_ll
                     <half, // Working prec
                      remifa::compute_type::TC16, // Inner update compute type
                      remifa::compute_type::TC32, // Outer update compute type
                      /*Inner blocking*/8, ///*Outer block size*/256,
                      float // Outer panel prec
                      >
                     (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

               }
               else {
                  //
                  // W=fp16 and I=TC16, O=TC32 and OP=W=FP16

                  // if (remifa::tests::prec::FP32 == in_pnl) {
                  //    //
                  //    // W=fp16 and I=TC16, O=TC32 and OP=W=fp16, IP=fp32
                        
                  //    remifa::lu_nopiv_ll
                  //       <half, // Working prec
                  //        remifa::compute_type::TC16, // Inner update compute type
                  //        remifa::compute_type::TC32, // Outer update compute type
                  //        8, // Inner blocking
                  //        256, // outer block size
                  //        half, // Outer panel prec
                  //        float // Inner panel prec
                  //        >
                  //       (cuhandle, m, m, d_l_f16, lda, d_inform);
                  // }
                  // else {
                  //
                  // W=fp16 and I=TC16, O=TC32 and OP=W=fp16, IP=W=fp16
                        
                  remifa::lu_nopiv_ll
                     <half, // Working prec
                      remifa::compute_type::TC16, // Inner update compute type
                      remifa::compute_type::TC32, // Outer update compute type
                      /*Inner blocking*/8, ///*Outer block size*/256,
                      /*Outer panel prec*/half, /*Inner panel prec*/half,
                      /*Outer factor prec*/half, /*Inner factor prec*/half
                      >
                     (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

                  // }
                     
               }
            }
            else if(remifa::compute_type::TC32 == in_upd) {
               //
               // W=fp16 and I=TC32, O=TC32

               if (remifa::tests::prec::FP32 == out_pnl) {
                  //
                  // W=fp16 and I=TC32, O=TC32 and OP=W=FP32

                  if (remifa::tests::prec::FP32 == in_pnl) {
                     //
                     // W=fp16 and I=TC32, O=TC32 and OP=W=FP32, IP=fp32
                        
                     if (remifa::tests::prec::FP32 == out_fac) {
                        //
                        // W=fp16 and I=TC32, O=TC32 and OP=fp32, IP=fp32 and OF=fp32

                        // std::cout << "TUTUTUT" << std::endl;

                        if (use_cutlass) {
                           if (split_k) {
                              lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp32xfp32_fp32xfp32(
                                    cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                                    split_k_slices, workspace);
                           }
                           else {
                              lu_nopiv_ll_cutlass_f16_tc32xtc32_fp32xfp32_fp32xfp32(
                                    cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                           }
                        }
                        else {
                           lu_nopiv_ll_f16_tc32xtc32_fp32xfp32_fp32xfp32(
                                 cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                        }

                        // remifa::lu_nopiv_ll
                        //    <half, // Working prec
                        //     compute_type::TC32, // Inner update compute type
                        //     compute_type::TC32, // Outer update compute type
                        //     /*Inner blocking*/8, ///*Outer block size*/256,
                        //     /*Outer panel prec*/float, /*Inner panel prec*/float,
                        //     /*Outer factor prec*/float, /*Inner factor prec*/float 
                        //     >
                        //    (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

                     } // OF=fp32
                     else { // OF=fp16

                        if (remifa::tests::prec::FP32 == in_fac) {
                           //
                           // W=fp16 and I=TC32, O=TC32 and OP=fp32, OF=fp16 and IP=IF=fp32
                           
                           remifa::lu_nopiv_ll<
                              half, // Working prec
                              compute_type::TC32, // Inner update compute type
                              compute_type::TC32, // Outer update compute type
                              /*Inner blocking*/8, ///*Outer block size*/256, 
                              /*Outer panel*/ float, /*Inner panel*/ float, 
                              /*Outer factor*/ half, /*Inner factor*/ float>
                              (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

                        } // IF=fp32
                        else { // IF=fp16
                           //
                           // W=fp16 and I=TC32, O=TC32 and OP=fp32, OF=W=fp16 and IP=IF=fp16

                           remifa::lu_nopiv_ll
                              <half, // Working prec
                               compute_type::TC32, // Inner update comp. type
                               compute_type::TC32, // Outer update comp. type
                               /*Inner blocking*/8, ///*Outer block size*/256,
                               /*Outer panel prec*/float, /*Inner panel prec*/float,
                               /*Outer factor prec*/half, /*Inner factor prec*/half 
                               >
                              (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                        }
                     }
                        
                  } // IP=fp32
                  else { // IP=fp16
                     //
                     // W=fp16 and I=TC32, O=TC32 and OP=fp32, IP=fp16

                     remifa::lu_nopiv_ll
                        <half, // Working prec
                         compute_type::TC32, // Inner update compute type
                         compute_type::TC32, // Outer update compute type
                         /*Inner blocking*/8, ///*Outer block size*/256, 
                         /*Outer panel prec*/float, /*Inner panel prec*/half,
                         /*Outer factor*/ half, /*Inner factor*/half
                         >
                        (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

                  } // IP=fp16
               } // OP=fp32
               else { // OP=fp16
                  //
                  // W=fp16 and I=TC32, O=TC32 and OP=W=fp16

                  if (remifa::tests::prec::FP32 == in_pnl) {
                     //
                     // W=fp16 and I=TC32, O=TC32 and OP=fp16, IP=fp32

                     if (remifa::tests::prec::FP32 == in_fac) {
                        //
                        // W=fp16 and I=TC32, O=TC32 and OP=OF=fp16 and IP=IF=fp32

                        if (use_cutlass) {
                           if (split_k) {
                              lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32_fp16xfp32_fp16xfp32(
                                    cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                                    split_k_slices, workspace);
                           }
                           else {
                              lu_nopiv_ll_cutlass_f16_tc32xtc32_fp16xfp32_fp16xfp32(
                                    cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                           }
                        }
                        else {
                           lu_nopiv_ll_f16_tc32xtc32_fp16xfp32_fp16xfp32(
                                 cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                        }

                        
                     } // IF=fp32
                     else { // IF=fp16
                        //
                        // W=fp16 and I=TC32, O=TC32 and OP=OF=fp16 and IP=fp32, IF=fp16

                        remifa::lu_nopiv_ll
                           <half, // Working prec
                            /*Inner upd comp. type*/compute_type::TC32,
                            /*Outer upd comp. type*/compute_type::TC32,
                            /*Inner blocking*/8, ///*Outer block size*/256, 
                            /*Outer panel*/half, /*Inner panel*/float,
                            /*Outer factor*/half, /*Inner factor*/half>
                           (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

                     }
                        
                  } // IP=fp32
                  else { // IP=fp16
                     //
                     // W=fp16 and I=TC32, O=TC32 and OP=fp16, OF=fp16 and IP=fp16, IP=fp16

                     if (use_cutlass) {
                        if (split_k) {
                           lu_nopiv_ll_cutlass_splitk_f16_tc32xtc32(
                                 cuhandle, m, m, nb, d_l_f16, lda, d_inform,
                                 split_k_slices, workspace);
                        }
                        else {
                           lu_nopiv_ll_cutlass_f16_tc32xtc32(
                                 cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                        }
                     }
                     else {
                        lu_nopiv_ll_f16_tc32xtc32(
                              cuhandle, m, m, nb, d_l_f16, lda, d_inform);
                     }

                  } // IP=fp32
               } // OP=fp16
            } // I=TC32
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }
               
         }
         else {
            std::cout << "[" << context << "] Outer update compute type NOT supported " << std::endl;
            std::exit(0);
         }

      } // W == FP16
      else if (remifa::tests::prec::FP32 == work_prec) {
         // W=fp32

         assert(nullptr != d_l);

         // Copy and convert back to original matrix
         // remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);

         if(remifa::compute_type::FP32 == out_upd) {
            // W=fp32 and O=fp32
               
            if (remifa::compute_type::FP32 == in_upd) {
               // W=fp32 and I=O=fp32 and OP=OF=fp32 and IP=IF=fp32
               if (use_cutlass) {
                  if (split_k) {
                     lu_nopiv_ll_cutlass_splitk_f32(
                           cuhandle, m, m, nb, d_l, lda, d_inform,
                           split_k_slices, workspace);
                  }
                  else {
                     lu_nopiv_ll_cutlass_f32(
                           cuhandle, m, m, nb, d_l, lda, d_inform);
                  }
               }
               else {
                  lu_nopiv_ll_f32(cuhandle, m, m, nb, d_l, lda, d_inform);
               }

            } // W == fp32 && O=fp32 && I=fp32 
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }

         } // W == fp32 && O=fp32
         else if(remifa::compute_type::TC32 == out_upd) {
            // W == fp32 && O=tc32
               
            if (remifa::compute_type::FP32 == in_upd) {

               if (use_cutlass) {
                  if (split_k) {
                     lu_nopiv_ll_cutlass_splitk_f32_fp32xtc32(
                           cuhandle, m, m, nb, d_l, lda, d_inform,
                           split_k_slices, workspace, d_l_f16, lda);
                  }
                  else {
                     lu_nopiv_ll_cutlass_f32_fp32xtc32(
                           cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);
                  }
               }
               else {
                  lu_nopiv_ll_f32_fp32xtc32(
                        cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);
               }

               //
               // W=fp32 and I=fp32 and O=TC32
               // remifa::lu_nopiv_ll
               //    <float, // Working prec
               //     remifa::compute_type::FP32, // Inner update compute type
               //     remifa::compute_type::TC32, // Outer update compute type
               //     /*Inner block size*/8 
               //     >
               //    (cuhandle, m, m, nb, d_l, lda, d_inform,
               //     split_k_slices, workspace, d_l_f16, lda);
                  
            } // I=FP32 O=TC32
            else if (remifa::compute_type::TC32 == in_upd) {

               if (use_cutlass) {
                  if (split_k) {
                     lu_nopiv_ll_cutlass_splitk_f32_tc32xtc32(
                           cuhandle, m, m, nb, d_l, lda, d_inform,
                           split_k_slices, workspace, d_l_f16, lda);
                  }
                  else {
                     lu_nopiv_ll_cutlass_f32_tc32xtc32(
                           cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);
                  }
               }
               else {
                  lu_nopiv_ll_f32_tc32xtc32(
                        cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);
               }

               //
               // W=fp32 and I=O=TC32
               // remifa::lu_nopiv_ll
               //    <float, // Working prec
               //     remifa::compute_type::TC32, // Inner update compute type
               //     remifa::compute_type::TC32, // Outer update compute type
               //     /*Inner block size*/8 ///*Outer block size*/256 
               //     >
               //    (cuhandle, m, m, nb, d_l, lda, d_inform,
               //     split_k_slices, workspace, d_l_f16, lda);
                  
            } // I=O=TC32
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }

         } // O=TC32
         else {
            std::cout << "[" << context << "] Outer update compute type NOT supported " << std::endl;
            std::exit(0);
         }

      }
      else {
         std::cout << "[" << context << "] Working precision NOT supported " << std::endl;
         std::exit(0);
      }
         
      // Wait for completion of CUDA kernels on the device
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);
      
      end = std::chrono::high_resolution_clock::now();

      // If working precision (W) if different from the precision use
      // to generate the original matrix, then we need to convert
      // the computed factors back to original precision

      // W = fp16
      //
         
      // Half precision matrix copy 
         
      if (remifa::tests::prec::FP16 == work_prec) {
         // Get results back into full-prec buffer
         remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);
         cuerr = cudaStreamSynchronize(stream);
         remifa::cuda_check_error(cuerr, context);
      }

      // Generate ipiv array. As we don't use pivoting in this
      // testing, we set the ipiv array to (1, 2, 3, .., n).
      for(int i=0; i<m; ++i) ipiv[i] = i+1; // ipiv must be 1-indexed

      // Transfer matrix to host memory from the device
      custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, &l[0], lda);
      remifa::cublas_check_error(custat, context);
      // Get info
      cuerr = cudaMemcpy(&info, d_inform, sizeof(int), cudaMemcpyDeviceToHost);
      remifa::cuda_check_error(cuerr, context);
         
      // Cleanup
      custat = cublasDestroy(cuhandle);
      cuerr = cudaFree(d_inform);
      remifa::cuda_check_error(cuerr, context);
   }

   else if (algo == remifa::tests::REMIFA_RL) {
      // Right-looking REMIFA algorithm
         
      cublasStatus_t custat; // CuBLAS status
      cublasHandle_t cuhandle;
      // inform_t inform; // Host side status

      int info;
      int *d_inform; // Device side status
      cuerr = cudaMalloc((void**)&d_inform, sizeof(int));
      remifa::cuda_check_error(cuerr, context, "d_inform allocatation failed");
         
      // Initialize factorization
      custat = cublasCreate(&cuhandle);
      remifa::cublas_check_error(custat, context);
      custat = cublasSetStream(cuhandle, stream);
      remifa::cublas_check_error(custat, context);
      // Select math mode
      // if (usetc) custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
      // else       custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);            
      custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);            
      remifa::cublas_check_error(custat, context);

      // Allocate memory on the device
      cuerr = cudaMalloc((void**)&d_l, (std::size_t)m*lda*sizeof(T));
      remifa::cuda_check_error(cuerr, context, "d_l allocatation failed");
      
      // Send matrix to device
      custat = cublasSetMatrix(m, m, sizeof(T), &l[0], lda, d_l, lda);
      // cudaMemcpy(d_l, l, lda*m*sizeof(T), cudaMemcpyHostToDevice);
      remifa::cublas_check_error(custat, context);

      // if (remifa::tests::prec::FP16 == work_prec) {
      // Allocate matrix with working precision
      cuerr = cudaMalloc((void**)&d_l_f16, (std::size_t)m*lda*sizeof(half));
      remifa::cuda_check_error(cuerr, context, "d_l_f16 allocatation failed");

      if (remifa::tests::prec::FP16 == work_prec) {
         // Convert original matrix to working precision
         remifa::convert(stream, m, m, d_l, lda, d_l_f16, lda);
         cuerr = cudaStreamSynchronize(stream);
         remifa::cuda_check_error(cuerr, context);
      }
      // else if (remifa::tests::prec::FP32 == work_prec) {
      //    remifa::convert(stream, m, m, d_l, lda, d_l_f16, lda);
      //    remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);
      //    cuerr = cudaStreamSynchronize(stream);
      //    remifa::cuda_check_error(cuerr, context);
      // }
      
      
      start = std::chrono::high_resolution_clock::now();

      //
      // Launch factorization on device
      //
         
      if (remifa::tests::prec::FP16 == work_prec) {
         // fp16 working precision

         assert(nullptr != d_l_f16);

         if(remifa::compute_type::FP16 == out_upd) {
            //
            // W=fp16 and I=O=fp16
            remifa::lu_nopiv_rl
               <half, // Working prec
                remifa::compute_type::FP16, // Inner update compute type
                remifa::compute_type::FP16, // Outer update compute type
                8, // Inner blocking
                // 256, // outer block size
                use_cutlass
                >
               (cuhandle, m, m, nb, d_l_f16, lda, d_inform);

         }
         else if(remifa::compute_type::TC16 == out_upd) {

            if (remifa::compute_type::FP16 == in_upd) {

               //
               // W=fp16, I=fp16 and O=TC16
               remifa::lu_nopiv_rl
                  <half, // Working prec
                   remifa::compute_type::FP16, // Inner update compute type
                   remifa::compute_type::TC16, // Outer update compute type
                   8, // Inner blocking
                   // 256, // outer block size
                   use_cutlass
                   >
                  (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
            }
            else if(remifa::compute_type::TC16 == in_upd) {
               //
               // W=fp16, I=fp16 and O=TC16
               remifa::lu_nopiv_rl
                  <half, // Working prec
                   remifa::compute_type::TC16, // Inner update compute type
                   remifa::compute_type::TC16, // Outer update compute type
                   8, // Inner blocking
                   // 256, // outer block size
                   use_cutlass
                   >
                  (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
            }
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }
         }
         else if(remifa::compute_type::TC32 == out_upd) {

            if (remifa::compute_type::FP16 == in_upd) {

               //
               // W=fp16, I=fp16 and O=TC32
               remifa::lu_nopiv_rl
                  <half, // Working prec
                   remifa::compute_type::FP16, // Inner update compute type
                   remifa::compute_type::TC32, // Outer update compute type
                   8, // Inner blocking
                   // 256, // outer block size
                   use_cutlass
                   >
                  (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
            }
            else if(remifa::compute_type::TC32 == in_upd) {
               //
               // W=fp16, I=fp16 and O=TC16
               remifa::lu_nopiv_rl
                  <half, // Working prec
                   remifa::compute_type::TC32, // Inner update compute type
                   remifa::compute_type::TC32, // Outer update compute type
                   8, // Inner blocking
                   // 256, // outer block size
                   use_cutlass
                   >
                  (cuhandle, m, m, nb, d_l_f16, lda, d_inform);
            }
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }

         }
         else {
            std::cout << "[" << context << "] Outer update compute type NOT supported " << std::endl;
            std::exit(0);
         }

      }
      else if (remifa::tests::prec::FP32 == work_prec) {
         // fp32 working precision
            
         if (remifa::compute_type::FP32 == out_upd) {
            //
            // W=T=(fp32 or fp16) and I=O=fp32
            remifa::lu_nopiv_rl
               <T, // Working prec
                remifa::compute_type::FP32, // Inner update compute type
                remifa::compute_type::FP32, // Outer update compute type
                8, // Inner blocking
                // 256, // outer block size
                use_cutlass
                >
               (cuhandle, m, m, nb, d_l, lda, d_inform);

         }
         else if (remifa::compute_type::TC32 == out_upd) {

            if (remifa::compute_type::FP32 == in_upd) {

               //
               // W=T=fp32 and I=fp32 and O=TC32
               remifa::lu_nopiv_rl
                  <T, remifa::FP32, remifa::TC32, 8, /*256,*/ use_cutlass>
                  (cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);
            }
            else if (remifa::compute_type::TC32 == in_upd) {

               remifa::lu_nopiv_rl
                  <T, remifa::TC32, remifa::TC32, 8, /*256,*/ use_cutlass>
                  (cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);

            }
            else {
               std::cout << "[" << context << "] Inner update compute type NOT supported " << std::endl;
               std::exit(0);
            }

         }
         else {
            std::cout << "[" << context << "] Outer update compute type NOT supported " << std::endl;
            std::exit(0);
         }
      }
      else {
         std::cout << "[" << context << "] Working precision NOT supported " << std::endl;
         std::exit(0);
      }
         
      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      remifa::cuda_check_error(cuerr, context);
      
      end = std::chrono::high_resolution_clock::now();

      if (remifa::tests::prec::FP16 == work_prec) {
         // Get results back into full-prec buffer
         remifa::convert(stream, m, m, d_l_f16, lda, d_l, lda);
         cuerr = cudaStreamSynchronize(stream);
         remifa::cuda_check_error(cuerr, context);
      }
         
      for(int i=0; i<m; ++i) ipiv[i] = i+1; // ipiv must be 1-indexed
      // Get matrix into host memory      
      custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, &l[0], lda);
      // cudaMemcpy(l, d_l, lda*m*sizeof(T), cudaMemcpyDeviceToHost);
      remifa::cublas_check_error(custat, context);
      // Get info
      cuerr = cudaMemcpy(&info, d_inform, sizeof(int), cudaMemcpyDeviceToHost);
      remifa::cuda_check_error(cuerr, context);
         
      // Cleanup
      custat = cublasDestroy(cuhandle);
      cuerr = cudaFree(d_inform);
      remifa::cuda_check_error(cuerr, context);
   }
   else {
      std::cout << "[" << context << "] Algo NOT available " << std::endl;
      std::exit(0);
   }

   cudaDeviceSynchronize();

   // std::cout << "[" << context << "] A = " << std::endl;
   // remifa::tests::print_mat_unsym("%8.3f", m, &a[0], lda);
   // std::cout << "[" << context << "] LU = " << std::endl;
   // remifa::tests::print_mat_unsym("%8.3f", m, &l[0], lda);
      

   if (check) {

      // Compute componentwise backward error for the solution of Ax=b
         
      int info; 
      int nrhs = 1;
      int ldsoln = m;
      std::vector<T> soln(nrhs*ldsoln);
      // Setup permuted rhs
      // std::vector<T> pb(m);
      // soln = b;
      std::memcpy(&soln[0], &b[0], nrhs*ldsoln*sizeof(T));
         
      info = remifa::host_getrs(
            remifa::OP_N, m, nrhs,
            &l[0], lda,
            &ipiv[0],
            &soln[0], ldsoln);
      if (info != 0) {
         std::cout << "[" << context << "][error] Failed to execute host_getrs "
                   << "(" << info << ")" << std::endl;
         std::exit(1);         
      }

      ////////////////////////////////////////
      // GPU solve

      // if (d_l_f16 == nullptr) {
      //    // If fp16 factors are not present, allocate fp16 matrix
      //    // and convert fp32 factos into buffer.
      //    cuerr = cudaMalloc((void**)&d_l_f16, (std::size_t)m*lda*sizeof(half));
      //    remifa::cuda_check_error(cuerr, context, "d_l_f16 allocation failed");
      //    // Convert original matrix to working precision
      //    remifa::convert(stream, m, m, d_l, lda, d_l_f16, lda);
      //    cuerr = cudaStreamSynchronize(stream);
      //    remifa::cuda_check_error(cuerr, context);
      // }
         
      // // f16
      // half *d_soln_f16 = nullptr;
      // // f32
      // float *d_soln_f32 = nullptr;

      // // Allocate fp16 solution on GPU
      // cuerr = cudaMalloc((void**)&d_soln_f16, (std::size_t)nrhs*ldsoln*sizeof(half));
      // remifa::cuda_check_error(cuerr, context, "d_soln_f16 allocation failed");         
      // // Allocate fp32 solution on GPU
      // cuerr = cudaMalloc((void**)&d_soln_f32, (std::size_t)nrhs*ldsoln*sizeof(float));
         
      // // Send rhs to device
      // custat = cublasSetMatrix(m, nrhs, sizeof(T), &b[0], ldsoln, d_soln_f32, ldsoln);
      // // Convert rhs into x
      // remifa::convert(stream, m, nrhs, d_soln_f32, ldsoln, d_soln_f16, ldsoln);
      // cuerr = cudaStreamSynchronize(stream);

      // auto slv_sa = std::chrono::high_resolution_clock::now();

      // using LuSolveGpuType = remifa::LuSolveGpu<half>;
      // LuSolveGpuType lu_solve_gpu;
      // lu_solve_gpu.initialize(stream, m, m, d_l_f16, lda);
      // // Peform forward substitution
      // lu_solve_gpu.fwd(d_soln_f16);
      // // Peform backwardsubstitution
      // lu_solve_gpu.bwd(d_soln_f16);
      // cuerr = cudaStreamSynchronize(stream);
      // // Convert fp16 soln into fp32 soln
      // remifa::convert(stream, m, nrhs, d_soln_f16, ldsoln, d_soln_f32, ldsoln);
      // cuerr = cudaStreamSynchronize(stream);

      // // using LuSolveGpuType = remifa::LuSolveGpu<float>;
      // // LuSolveGpuType lu_solve_gpu;
      // // lu_solve_gpu.initialize(stream, m, m, d_l, lda);
      // // // Peform forward substitution
      // // lu_solve_gpu.fwd(d_soln_f32);
      // // // Peform backwardsubstitution
      // // lu_solve_gpu.bwd(d_soln_f32);
      // // cuerr = cudaStreamSynchronize(stream);

      // auto slv_en = std::chrono::high_resolution_clock::now();
      // long tslv =  
      //    std::chrono::duration_cast<std::chrono::nanoseconds>
      //    (slv_en-slv_sa).count();
         
      // std::cout << "solve time = " << 1e-9*tslv << std::endl; 

      // std::vector<T> soln_f32(nrhs*ldsoln, 0.0);
      // custat = cublasGetMatrix(m, nrhs, sizeof(T), d_soln_f32, ldsoln, &soln_f32[0], ldsoln);

      // // printf("rhs = %.6e %.6e\n", b[0], b[1]);
      // // printf("soln_f32 = %.6e %.6e\n", soln_f32[0], soln_f32[1]);

      // double slv_gpu_bwderr = unsym_compwise_error(
      //       m, m, &a[0], lda, &b[0], nrhs, &soln_f32[0], ldsoln, &l[0], lda);
      // printf("solve gpu, cwerr = %le\n", slv_gpu_bwderr);

      ////////////////////////////////////////
      // Calculate forward error
      // double fwderr = forward_error(m, nrhs, &soln[0], ldsoln);
      // printf("fwderr = %le\n", fwderr);         
      // Calculate normwise bwd error
      double nwerr = unsym_backward_error(
            m, m, &a[0], lda, &b[0], nrhs, &soln[0], ldsoln);
      // double bwderr = remifa::tests::backward_error(m, &a[0], lda, &b[0], nrhs, &soln[0], m);
      printf("nwerr = %le\n", nwerr);
      double bwderr = unsym_compwise_error(
            m, m, &a[0], lda, &b[0], nrhs, &soln[0], ldsoln, &l[0], lda);
      printf("cwerr = %le\n", bwderr);
         
   }

   ////////////////////////////////////////
   // Calculate walltime
      
   long ttotal =  
      std::chrono::duration_cast<std::chrono::nanoseconds>
      (end-start).count();
   double flops = ((double)2.0*m*m*m)/3.0;
   printf("factor time (s) = %e\n", 1e-9*ttotal);
   printf("GFlop/s = %.3f\n", flops/(double)ttotal);

   // Cleanup memory
   if (nullptr != d_l) {
      cuerr = cudaFree(d_l);
      remifa::cuda_check_error(cuerr, context);
   }
      
   if (nullptr != d_l_f16) {
      cuerr = cudaFree(d_l_f16);
      remifa::cuda_check_error(cuerr, context);
   }

   delete[] a;
   delete[] l;
   delete[] b;

   return 0;
}

}} // End of namespace remifa::tests
