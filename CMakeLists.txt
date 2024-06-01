cmake_minimum_required(VERSION 3.0)

project(cuda_precess)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)          
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)

include_directories(${PROJECT_SOURCE_DIR})

# opencv 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/usr/include/arrch64-linux-gnu/)
#link_directories(/usr/lib/arrch64-linux-gnu/)

# 

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 


add_executable(process ${PROJECT_SOURCE_DIR}/test.cu propress.cpp)
#link_directories(/home/zeh/Desktop/cuda_yolo/build)
#target_link_libraries(trt_infer ${PROJECT_SOURCE_DIR}/libcuda_test.a)
target_link_libraries(process nvinfer)
target_link_libraries(process nvonnxparser)
target_link_libraries(process cudart)
target_link_libraries(process nvinfer_plugin)
target_link_libraries(process ${OpenCV_LIBS})