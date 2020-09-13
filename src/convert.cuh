/// @file
/// @copyright 2020- 
/// @author Florent Lopez
#pragma once

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace remifa {

   using half = __half;

   // @brief Convert matrix a of type TA into type TAO and put result
   // in aout
   template<typename TA, typename TAO>
   void convert(cudaStream_t const stream, int m, int n, TA *const a, int lda, TAO *const aout, int ldaout);

} // End of namespace gam
