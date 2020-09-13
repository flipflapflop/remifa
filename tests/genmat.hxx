/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "utils.hxx"

// STD
#include <random>
#include <fstream>

namespace remifa { 
namespace tests {

template<typename T>
void read_mtx(std::string const& matfile, int& n, T** a, int& lda) {

   std::cout << "[read_mtx] matfile = " << matfile << std::endl;

   if (*a) delete[] *a;

   std::int64_t nrows, ncols, nnz;

   // Open matrix file:
   std::ifstream fin(matfile);
   
   // Ignore headers and comments:
   while (fin.peek() == '%') fin.ignore(2048, '\n');

   // Read matrix parameters
   fin >> nrows >> ncols >> nnz;

   std::cout << "[read_mtx] nrows = " << nrows
             << ", ncols = " << ncols
             << ", nnz = " << nnz << std::endl;

   n = ncols; 
   lda = remifa::align_lda<T>(n);

   size_t nelems = (std::size_t)lda*(std::size_t)n;
   *a = new T[nelems];

   std::fill(*a, *a + nelems, T(0));

   // Read matrix data
   for (int c = 0; c < nnz; c++) {
      int row, col;
      T val;
      // Symmetric matrix: Lower diagonal info
      fin >> row >> col >> val;

      (*a)[(std::size_t)(row-1) + (col-1)*((std::size_t)lda)] = val;
      // Fill upper diagonal part
      (*a)[(std::size_t)(col-1) + (row-1)*((std::size_t)lda)] = val;

      // if (row==col) {
      //    (*a)[(std::size_t)(row-1) + (col-1)*lda] += T(n);
      // }

   }

   fin.close();
}
   
// Generates a random digonally-dominant matrix. Off diagonal entries
// are Unif[-1,1]. Each diagonal entry a_ii = Unif[0.1,1.1] +
// sum_{i!=j} |a_ij|.
template<typename T>
void gen_diagdom(std::int64_t m, std::int64_t n, T* a, std::int64_t lda) {

   std::cout << "[gen_diagdom]" << std::endl;

   std::default_random_engine generator;
   std::uniform_real_distribution<T> distribution(-1.0, 1.0);
   // std::uniform_real_distribution<T> distribution(-1.0/T(std::sqrt(n)), 1.0/T(std::sqrt(n)));
   // std::uniform_real_distribution<T> distribution(-1.0/T(n), 1.0/T(n));
   // std::uniform_real_distribution<T> distribution(0.0, 1.0/T(n));
   // std::uniform_real_distribution<T> distribution(0.0, (T)1.0/std::sqrt(n));

   // gen_randmat(m, n, a, lda);

   // gen_randmat_sym(n, a, lda);
   // gen_sym_indef(n, a, lda);
   /* Make diagonally dominant */
   // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
   for(int i=0; i<m; ++i) {
      for(int j=0; j<n; ++j) {
         a[j*lda+i] = distribution(generator);

         // if (i != j) {
         //    a[i*lda+i] += fabs(a[j*lda+i]);
         // }
         // #if defined(HAVE_CUTLASS)
         //             a[j*lda+i] = (T)cutlass::half_t(a[j*lda+i]);
         // #endif
      }
      a[i*lda+i] = T(m);
      // a[i*lda+i] = T(m/2);
      // a[i*lda+i] = (T)std::sqrt(n);
   }

}
   
template<typename T>
void genpos_randmat(
      std::int64_t m, std::int64_t n, T* a, std::int64_t lda,
      T lbound=0.0, T ubound=1.0,
      typename std::default_random_engine::result_type seed=1u
      ) {

   // std::random_device rd;
   std::default_random_engine generator(seed);
   // generator.seed(seed);
   std::uniform_real_distribution<T> distribution(lbound, ubound);
   // std::uniform_real_distribution<T> distribution(0.0, 1e-3);
   // std::uniform_real_distribution<T> distribution(1e-3, 1.0);

   for(int j=0; j<n; ++j) {
      for(int i=0; i<m; ++i) {
         a[j*lda+i] = distribution(generator);
         // #if defined(HAVE_CUTLASS)
         //             a[j*lda+i] = (T)cutlass::half_t(a[j*lda+i]);
         // #endif

      }
   }
}

template<typename T>
void gen_diagdom_pos(
      std::int64_t m, std::int64_t n, T* a, std::int64_t lda,
      T lbound=0.0, T ubound=1.0,
      typename std::default_random_engine::result_type seed=1u) {

   genpos_randmat(m, n, a, lda, lbound, ubound, seed);

   for(int j=0; j<n; ++j) {
      a[j*lda+j] = T(m)*ubound;
      // a[j*lda+j] = T(m/2);
      // a[j*lda+j] = sqrt(T(m)*ubound);
      // a[j*lda+j] = T(m)*ubound*0.1;

      // for(int i=0; i<m; ++i) {
      //    a[j*lda+j] += std::fabs(a[j*lda+i]);
      // }
   }
}

}}
