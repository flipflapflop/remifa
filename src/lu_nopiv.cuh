/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#define BLOCKS 1 // Number of tiles per Thread blocks

#define BLOCK_SIZE 8 // Thread block size

namespace remifa {

   template<typename T, int ib=BLOCK_SIZE>
   void lu_nopiv_panel(
         const cudaStream_t stream,
         int m, int n,
         T const *const d, int ldd,
         T *l, int ldl,
         T *u, int ldu,
         int *stat);

} // End of namespace gam
