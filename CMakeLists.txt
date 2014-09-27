cmake_minimum_required (VERSION 2.8.12)
project(LibSVM)

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  $ENV{DEV_LIBS}/clany/cmake
)
include(Utilities)

find_compiler(
  GNU   4.7  "-std=c++11 -Wall"
  Clang 3.2  "-std=c++11 -Wall"
  MSVC  2013 "/W4"
)

option(USE_MY_PATH "Use my own default library path" OFF)
if(USE_MY_PATH)
  add_default_lib_path($ENV{DEV_LIBS} $ENV{DEV_PLAT})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/libsvm)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(LIBSVM_LIST
  libsvm/libsvm.h
  libsvm/libsvm.cpp
)
add_library(libsvm STATIC ${LIBSVM_LIST})

set(TEST_LIST
  libsvm/libsvm.cpp
  src/svm.cpp
  src/main.cpp
)
add_executable(svm_test ${TEST_LIST})
target_link_libraries(svm_test ${OpenCV_LIBS})
