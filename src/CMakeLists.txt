pybind11_add_module(_pydsl SHARED lib.cpp)
target_link_libraries(_pydsl PRIVATE /home/lijinpei/development/build_llvm/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.so.20.0git CUDA::cuda_driver)
target_include_directories(_pydsl PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
