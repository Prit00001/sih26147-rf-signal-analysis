cmake_minimum_required(VERSION 3.15)
project(sigscope_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

# SR-05: C++ hardening flags.
add_compile_options(-Wall -Wextra -fstack-protector-strong)
add_compile_definitions($<$<CONFIG:Release>:_FORTIFY_SOURCE=2>)

option(SIGSCOPE_SANITIZE "Build with ASan/UBSan instrumentation" OFF)
if(SIGSCOPE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O0)
  add_link_options(-fsanitize=address,undefined)
endif()

pybind11_add_module(_native src/hello.cpp src/viterbi.cpp src/ldpc.cpp src/correlation.cpp)

option(SIGSCOPE_BUILD_TESTS "Build the GoogleTest C++ kernel unit tests" OFF)
if(SIGSCOPE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(sigscope_kernel_tests tests/test_kernels.cpp src/viterbi.cpp src/ldpc.cpp src/correlation.cpp)
  target_include_directories(sigscope_kernel_tests PRIVATE src)
  target_link_libraries(sigscope_kernel_tests PRIVATE gtest_main)
endif()
