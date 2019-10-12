cmake_minimum_required(VERSION 3.5)
project(RetinaFace)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}/cmake)

# find cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_PATH: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIB_PATH: ${CUDA_LIBRARIES}")
# find opencv
set(OpenCV_DIR /usr/local/opencv3.4/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV_INLCUDE_PATH: ${OpenCV_INLCUDE_DIRS}")
message(STATUS "OpenCV_LIB_PATH: ${OpenCV_LIBRARIES}")
# set PKG_CONFIG_PATH for pkg-config find
find_package(PkgConfig)
# find protobuf3.x
pkg_search_module(PROTOBUF REQUIRED protobuf)
message(STATUS "Protobuf_HEADER: ${PROTOBUF_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIB: ${PROTOBUF_LIBRARIES}")
link_directories("/usr/local/protobuf3.7/lib") # find version 3 lib first
# set caffe path
set(CAFFE_INCLUDE_DIRS /xxx/caffe/build/src
                       /xxx/caffe/include)
set(CAFFE_LIB_DIRS     /xxx/caffe/build/lib)
include_directories(${CMAKE_SOURCE_DIR}/include
                    ${CUDA_INCLUDE_DIRS}
                    ${OpenCV_INLCUDE_DIRS}
                    ${CAFFE_INCLUDE_DIRS})
link_directories(${CAFFE_LIB_DIRS})
link_libraries(${CUDA_LIBRARIES}
               ${OpenCV_LIBRARIES}
               ${PROTOBUF_LIBRARIES}
               caffe
               glog
               boost_system)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)
add_executable(run_retinaface ${SRC})