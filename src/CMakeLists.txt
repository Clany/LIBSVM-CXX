set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(SRC_LIST libsvm.cpp svm.cpp)

add_library(cv_svm STATIC ${SRC_LIST})