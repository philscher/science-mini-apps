# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")


# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(bump-on-tail CXX)

# You need this for using Kokkos_ROOT variable
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

#find_package(fftw3 REQUIRED)
find_package(Kokkos REQUIRED)


find_package(Kokkos)
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

add_executable(bump-on-tail main.cpp BumpOnTail.cpp)

target_link_libraries(bump-on-tail fftw3 stdc++ m)
#m)



# This is the only thing required to set up compiler/linker flags
target_link_libraries(bump-on-tail Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
