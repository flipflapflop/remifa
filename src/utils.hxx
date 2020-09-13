/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include <cstddef>
#include <iostream>

#include <library_types.h>

// Magma macros for compatibility
#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_REAL(x)           (x)
#define MAGMA_D_IMAG(x)           (0.0)

namespace remifa { 

   enum compute_type
      {
       FP32,
       FP16,
       TC32,
       TC16
      };
   
   // =============================================================================
   // To use int64_t, link with mkl_intel_ilp64 or similar (instead of mkl_intel_lp64).
   // Similar to magma_int_t we declare magma_index_t used for row/column indices in sparse
   // #if defined(MAGMA_ILP64) || defined(MKL_ILP64)
   //    typedef long long int magma_int_t;  // MKL uses long long int, not int64_t
   // #else
   typedef int magma_int_t;
   // #endif

   template<typename T>
   cudaDataType_t get_cublas_type();

   template<typename T>
   cudaDataType_t cublas_type();

   template<typename T>
   std::string type_name();

   template<remifa::compute_type CT>
   std::string compute_type_name();

   /** Return nearest value greater than supplied lda that is multiple of alignment */
   template<typename T>
   size_t align_lda(size_t lda) {
      int const align = 64;
      // int const align = 256;
// #if defined(__AVX512F__)
//       int const align = 64;
// #elif defined(__AVX__)
//       int const align = 32;
// #else
//       int const align = 16;
// #endif
      static_assert(align % sizeof(T) == 0, "Can only align if T divides align");
      int const Talign = align / sizeof(T);
      return Talign*((lda-1)/Talign + 1);
   }

   template<typename T>
   void print_mat(char const* format, int m, int n, T const* a, int lda) {
      for(int i=0; i<m; ++i) {
         printf("%d:",  i);
         for(int j=0; j<n; ++j)
            printf(format, a[j*lda+i]);
         printf("\n");
      }
   }

}
