set(tests_drivers)

# List tests drivers
list(APPEND tests_drivers "testing_lu_gpu.cxx")

file(
  GLOB
  tests_src
  *.cxx
  *.hxx)

# Remove tests drivers from tests sources
file(
  GLOB 
  to_remove
  ${tests_drivers})

list(REMOVE_ITEM tests_src ${to_remove})
# message(STATUS "tests_src: ${tests_src}")

# Add tests lib
add_library(remifa_tests STATIC ${tests_src})
set_property(TARGET remifa_tests PROPERTY LINKER_LANGUAGE CXX)
# set_property(TARGET remifa_tests PROPERTY LINKER_LANGUAGE CUDA)

# add_executable(testing_chol_gpu testing_chol_gpu.cxx)
# set_property(TARGET testing_chol_gpu PROPERTY LINKER_LANGUAGE Fortran)
# target_link_libraries(testing_chol_gpu remifa_tests)

message(STATUS "LIBS: ${LIBS}")

foreach(tests_driver ${tests_drivers})

  # message(STATUS "test driver: ${tests_driver}")
  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)

  add_executable(${tests_driver_name} ${tests_driver})

  # set_property(TARGET ${tests_driver_name} PROPERTY LINKER_LANGUAGE CXX)
  # set_property(TARGET ${tests_driver_name} PROPERTY LINKER_LANGUAGE CUDA)
  
  target_link_libraries(${tests_driver_name} remifa)
  target_link_libraries(${tests_driver_name} remifa_cuda)
  # target_link_libraries(${tests_driver_name} remifa_tests)
  target_link_libraries(${tests_driver_name} ${LIBS})

endforeach()
