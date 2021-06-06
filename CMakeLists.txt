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
find_package(absl REQUIRED)
find_package(glog REQUIRED)
find_package(doctest REQUIRED)

option(BUILD_SHARED_LIBS "Build shared libraries" On)
option(BUILD_TESTING "Build testing" On)

cc_library(
  NAME base
  DEPS fmt::fmt glog::glog mg_options
  INCDIRS ${CMAKE_SOURCE_DIR}
  INTERFACE)

enable_testing()
add_subdirectory(mg)
