/// @file
/// @copyright 2019- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#include "utils.hxx"

#include <library_types.h>
#include <cuda_fp16.h>

namespace remifa {

   template<> cudaDataType_t get_cublas_type<half>() {return CUDA_R_16F;}
   template<> cudaDataType_t get_cublas_type<float>() {return CUDA_R_32F;}
   template<> cudaDataType_t get_cublas_type<double>() {return CUDA_R_64F;}

   template<> cudaDataType_t cublas_type<half>() {return CUDA_R_16F;}
   template<> cudaDataType_t cublas_type<float>() {return CUDA_R_32F;}
   template<> cudaDataType_t cublas_type<double>() {return CUDA_R_64F;}

   template<> std::string type_name<float>() { return "fp32"; }
   template<> std::string type_name<half>() { return "fp16"; }
   template<> std::string type_name<double>() { return "fp64"; }
   
   template<> std::string compute_type_name<remifa::compute_type::FP16>() { return "fp16"; }
   template<> std::string compute_type_name<remifa::compute_type::FP32>() { return "fp32"; }
   template<> std::string compute_type_name<remifa::compute_type::TC16>() { return "TC16"; }
   template<> std::string compute_type_name<remifa::compute_type::TC32>() { return "TC32"; }
}
