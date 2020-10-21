cmake_minimum_required(VERSION 3.1)

#使用C或者C++语言
project(Caffe C CXX)

#使用C++11
set(CMAKE_CXX_STANDARD 11)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# include
SET(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}/include
                      )
include_directories(${Caffe_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DISR})


file(GLOB SO_SRC_LIST "${PROJECT_SOURCE_DIR}/tools/caffe.cpp")
file(GLOB_RECURSE COMPARE_SRC_LIST "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(Demo ${SO_SRC_LIST} ${COMPARE_SRC_LIST})
target_link_libraries(Demo ${Boost_LIBRARIES})


