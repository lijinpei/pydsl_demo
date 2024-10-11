project(pydsl)
cmake_minimum_required(VERSION 3.30)

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
find_package(MLIR CONFIG REQUIRED)
find_package(CUDAToolkit MODULE REQUIRED)

add_subdirectory(src)
