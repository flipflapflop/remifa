cmake_minimum_required (VERSION 2.8)
project (REMIFA)
enable_language(CXX)
enable_language(CUDA)

# CMake version
message(STATUS "CMake version: ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")

# Include our own cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

################################################################################
# Linker options
set(LINKER_LANGUAGE CXX)

################################################################################
# Compiler options

########################################
# CXX flags
# STD C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Note: Not supported feature for Intel compilers in CMake version < 3.6
# set(CMAKE_CXX_STANDARD 11) # C++11...
# set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
# set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -march=native")
# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=leak")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=bounds -fbounds-check")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fbounds-check")

########################################
# Set default build type
set(default_build_type "RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Using default build type '${default_build_type}' because none was specified")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build (Debug, Release, MinSizeRel, RelWithDebInfo)" FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_MINSIZEREL})
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")  
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif()

################################################################################
# Third-party libraries

########################################
# std C++ (needed for linking with ssids)
set(LIBS ${LIBS} stdc++)

########################################
# CUDA

find_package(CUDA REQUIRED)

# list(APPEND CUDA_NVCC_FLAGS "-g")
# list(APPEND CUDA_NVCC_FLAGS "-g -O2")
# string(APPEND CMAKE_CUDA_FLAGS " -g")

set(REMIFA_NVCC_ARCHS "60;61;70;75" CACHE STRING "The SM architectures to build code for.")
  
# Set NVCC arguments
foreach(ARCH ${REMIFA_NVCC_ARCHS})
  # string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
endforeach()

MESSAGE( STATUS "CUDA found:         " ${CUDA_FOUND} )
MESSAGE( STATUS "CUDA host compiler: " ${CUDA_HOST_COMPILER} )
MESSAGE( STATUS "CUDA nvcc compiler: " ${CUDA_NVCC_EXECUTABLE})
MESSAGE( STATUS "NVCC FLAGS:         " ${CMAKE_CUDA_FLAGS} )

set(LIBS ${LIBS} ${CUDA_LIBRARIES})
set(LIBS ${LIBS} ${CUDA_CUBLAS_LIBRARIES})
set(LIBS ${LIBS} ${CUDA_cusolver_LIBRARY})

########################################
# BLAS
set(LBLAS "" CACHE STRING "BLAS library")

# If LBLAS not set, use environement variable
if(LBLAS)
  set(BLAS_LIBRARIES ${LBLAS})
elseif(DEFINED ENV{BLAS_LIB})
  set(BLAS_LIBRARIES $ENV{BLAS_LIB})
endif()

if(DEFINED BLAS_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT BLAS_FOUND)
    message(ERROR "User supplied BLAS is NOT working")
  endif()
else()
  find_package(BLAS)
endif()

if(BLAS_FOUND)
  set(LIBS ${LIBS} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "BLAS NOT found")
endif (BLAS_FOUND)

########################################
# LAPACK
set(LLAPACK "" CACHE STRING "LAPACK library")
# If LBLAS not set, use environement variable 
if(LLAPACK)
  set(LAPACK_LIBRARIES ${LLAPACK})
elseif(DEFINED ENV{LAPACK_LIB})
  set(LAPACK_LIBRARIES $ENV{LAPACK_LIB})
endif()
if(DEFINED LAPACK_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dpotrf" LAPACK_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT LAPACK_FOUND)
    message(ERROR "User supplied LAPACK is NOT working")
  endif()

else()
  find_package(LAPACK)
endif()

if (LAPACK_FOUND)
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "LAPACK library NOT found")
endif (LAPACK_FOUND)

################################################################################
# Sources

########################################
# src

# Include source directory
include_directories(${REMIFA_SOURCE_DIR}/src)

add_subdirectory(src)

########################################
# tests
add_subdirectory(tests)

################################################################################
# Print summary

message("--------------------------------------------------------------------------------")
message("Configuration of REMIFA done.")
message(" ")
message(
  "Build type: ${CMAKE_BUILD_TYPE}\n"
  "CXX compiler:  ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID})\n"
  "CXX compiler flags: ${CMAKE_CXX_FLAGS_ALL}\n"
  "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}\n"
  )
message(" ")
message("BLAS libraries: ${BLAS_LIBRARIES}")
message("LAPACK libraries: ${LAPACK_LIBRARIES}")
message(" ")
message(
  "CUDA host compiler: ${CUDA_HOST_COMPILER}\n"
  "CUDA libraries: ${CUDA_LIBRARIES}\n"
  "CUBLAS libraries: ${CUDA_CUBLAS_LIBRARIES}\n"
  "NVVC comiler: ${CUDA_NVCC_EXECUTABLE}\n"
  "NVCC FLAGS: ${CUDA_NVCC_FLAGS}\n"
  )
message(" ")
message("--------------------------------------------------------------------------------")
