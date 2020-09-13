# get source files
file(
  GLOB 
  sources 
  *.cxx
  *.hxx
  )

# message(STATUS "sources: ${sources}")

add_library(remifa ${sources})
set_property(TARGET remifa PROPERTY LINKER_LANGUAGE CXX)

file(
  GLOB 
  cuda_src 
  *.cu
  *.cuh
  )

# cuda_add_library(remifa_cuda STATIC ${cuda_src})
add_library(remifa_cuda STATIC ${cuda_src})

set_target_properties(remifa_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(remifa_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(remifa_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
