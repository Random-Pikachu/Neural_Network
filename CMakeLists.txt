cmake_minimum_required(VERSION 3.16)

project(neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
#    src/*.cpp
#)

#list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

#add_library(neural_core ${SOURCES})

add_executable(cpp_neural src/main.cpp)

#target_link_libraries(cpp_neural PRIVATE neural_core)