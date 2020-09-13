/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// STD
#include <string>
#include <iostream>

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace remifa {

   // @brief Check CUDA error and exit if error is detected
   inline void cuda_check_error(
         cudaError_t cuerr, std::string fname, 
         std::string const& msg = std::string()) {
      if (cuerr != cudaSuccess) {
         // std::string msg = "[CUDA error] " + msg + " (" + cudaGetErrorString(cuerr) + ")";
         throw std::runtime_error("[CUDA error] " + msg + " (" + cudaGetErrorString(cuerr) + ")");
         // std::cout << "[" << fname << "][CUDA error] "
         //           << msg
         //           << " (" << cudaGetErrorString(cuerr) << ")" << std::endl;
         // std::exit(1);
      }
   }

   // @brief Check cuBLAS error and exit if error is detected
   inline void cublas_check_error(
         cublasStatus_t custat, std::string fname,
         std::string const& msg = std::string()) {
      if (custat != CUBLAS_STATUS_SUCCESS) {    
         throw std::runtime_error("[cuBLAS error] " + msg + " (" + std::to_string(custat) + ")");

         // std::cout << "[" << fname << "][cuBLAS error] "
         //           << msg
         //           << " (" << custat << ")" << std::endl;
         // std::exit(1);
      }
   }
}
