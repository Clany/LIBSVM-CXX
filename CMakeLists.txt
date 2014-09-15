cmake_minimum_required (VERSION 2.6)
project(LibSVM)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/openmp")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "-fopenmp")
endif()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(MY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${MY_INCLUDE_DIR})

add_subdirectory(src bin)