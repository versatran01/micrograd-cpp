cmake_minimum_required(VERSION 3.15)
project(micrograd)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CC_TARGET_PREFIX mg)
include(CMakeHelpers)
add_library(mg_options INTERFACE)
enable_warnings(mg_options)
enable_sanitizers(mg_options)

find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(ArrayFire REQUIRED)

find_package(GTest REQUIRED)
find_package(GMock REQUIRED)
find_package(benchmark REQUIRED)

option(BUILD_SHARED_LIBS "Build shared libraries" On)
option(BUILD_TESTING "Build testing" On)

enable_testing()
add_subdirectory(mg)
