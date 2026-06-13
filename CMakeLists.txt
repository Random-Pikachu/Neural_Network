cmake_minimum_required(VERSION 3.16)

project(neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(cpp_neural
    src/main.cpp
    src/helpers/tensor.cpp
)

target_include_directories(cpp_neural PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/headers
)
