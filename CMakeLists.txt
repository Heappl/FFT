cmake_minimum_required (VERSION 2.8)
project (FFT)

set(dir ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(CMAKE_CXX_FLAGS "--std=c++11 -O3")

include_directories(${PROJECT_SOURCE_DIR}/gtest-1.7.0/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory (gtest-1.7.0)

add_executable(
    test
    test/dft.cpp
    test/convolution.cpp
    test/fft.cpp
)
target_link_libraries(test gtest gtest_main)

