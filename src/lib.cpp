#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda.h"

#include <iostream>

#define CHECK_CURESULT(x) checkCUresultImpl(x, __FILE__, __LINE__)

namespace py = pybind11;

using namespace mlir;

namespace {

void checkCUresultImpl(CUresult res, const char *file, int line) {
  if (res == CUDA_SUCCESS) {
    return;
  }
  std::cerr << "CUDA driver api call failed: " << file << " +" << line << '\n';
  const char *str;
  if (CUDA_SUCCESS == cuGetErrorName(res, &str)) {
    std::cerr << "error: " << str << '\n';
  } else {
    std::cerr << "unknown error-name\n";
  }
  if (CUDA_SUCCESS == cuGetErrorString(res, &str)) {
    std::cerr << "description: " << str << '\n';
  } else {
    std::cerr << "unkown error-description\n";
  }
}

void convertToLLVMIRDialect(Operation *modOp) {
  PassManager pm(modOp->getContext());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // Blanket-convert any remaining linalg ops to loops if any remain.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  // Blanket-convert any remaining affine ops if any remain.
  pm.addPass(createLowerAffinePass());
  // Convert SCF to CF (always needed).
  pm.addPass(createConvertSCFToCFPass());
  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Convert vector to LLVM (always needed).
  pm.addPass(createConvertVectorToLLVMPass());
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addPass(createLowerAffinePass());
  // Convert MemRef to LLVM (always needed).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  // Convert Func to LLVM (always needed).
  pm.addPass(createConvertFuncToLLVMPass());
  // Convert Index to LLVM (always needed).
  pm.addPass(createConvertIndexToLLVMPass());
  // Convert remaining unrealized_casts (always needed).
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (!pm.run(modOp).succeeded()) {
    assert(false);
  }
}

void attachDataLayout(const std::string &triple, const std::string &proc,
                      const std::string &features, llvm::Module &llvmMod) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "look up target failed:" << error << '\n';
    assert(target);
    return;
  }
  llvm::TargetOptions opt;
  // Target machine is only used to create the data layout.
  std::unique_ptr<llvm::TargetMachine> machine{
      target->createTargetMachine(triple, "", "", opt, llvm::Reloc::PIC_,
                                  std::nullopt, llvm::CodeGenOptLevel::None)};
  llvmMod.setTargetTriple(triple);
  // set data layout
  llvmMod.setDataLayout(machine->createDataLayout());
}
void init_mlir_helper(py::module &m) {
  m.def("translateMLIRModToLLVMIR",
        [](py::object capsule) {
          MlirOperation mod_op = mlirPythonCapsuleToOperation(capsule.ptr());
          Operation *moduleOp = unwrap(mod_op);
          // moduleOp->dump();
          // auto* mlir_ctx = moduleOp->getContext();
          // moduleOp->walk([mlir_ctx](func::FuncOp funcOp) {
          //   funcOp->setAttr("nvvm.kernel",
          //   IntegerAttr::get(IntegerType::get(mlir_ctx, 1), 1));
          // });
          convertToLLVMIRDialect(moduleOp);
          llvm::LLVMContext llvm_ctx;
          std::unique_ptr<llvm::Module> llvmMod =
              translateModuleToLLVMIR(moduleOp, llvm_ctx);
          assert(llvmMod);
          for (llvm::Function &f : llvmMod->functions()) {
            if (!f.isDeclaration()) {
              f.setCallingConv(llvm::CallingConv::PTX_Kernel);
            }
          }
          attachDataLayout("nvptx64-nvidia-cuda", "sm_89", "+ptx85",
                           *llvmMod.get());
          std::string buf;
          llvm::raw_string_ostream os(buf);
          llvmMod->print(os, nullptr);
          os.flush();
          return buf;
        })
      .def("translateLLVMIRToPtx",
           [](const std::string &ptx) {
             py::gil_scoped_release allow_threads;
             // create LLVM module from C++
             llvm::LLVMContext context;
             std::unique_ptr<llvm::MemoryBuffer> buffer =
                 llvm::MemoryBuffer::getMemBuffer(ptx);
             llvm::SMDiagnostic error;
             std::unique_ptr<llvm::Module> llvmMod =
                 llvm::parseIR(buffer->getMemBufferRef(), error, context);
             if (!llvmMod) {
               llvm::report_fatal_error(
                   "failed to parse IR: " + error.getMessage() +
                   "lineno: " + std::to_string(error.getLineNo()));
             }
             std::string triple = "nvptx64-nvidia-cuda";
             std::string error_str;
             auto target =
                 llvm::TargetRegistry::lookupTarget(triple, error_str);
             llvm::TargetOptions opt;
             std::unique_ptr<llvm::TargetMachine> machine{
                 target->createTargetMachine(triple, "", "", opt,
                                             llvm::Reloc::PIC_, std::nullopt,
                                             llvm::CodeGenOptLevel::None)};
             llvm::legacy::PassManager pass;
             std::string result;
             llvm::raw_string_ostream stream(result);
             llvm::buffer_ostream pstream(stream);
             machine->addPassesToEmitFile(pass, pstream, nullptr,
                                          llvm::CodeGenFileType::AssemblyFile);
             pass.run(*llvmMod);
             return result;
           })
      .def("load_cubin", [](py::bytes bytes, const std::string &func_name) {
        auto *obj = bytes.ptr();
        CUcontext cu_ctx;
        auto cu_res = cuCtxGetCurrent(&cu_ctx);
        if (cu_res == CUDA_ERROR_NOT_INITIALIZED) {
          CHECK_CURESULT(cuInit(0));
          CHECK_CURESULT(cuCtxGetCurrent(&cu_ctx));
        } else {
          CHECK_CURESULT(cu_res);
        }
        if (!cu_ctx) {
          CHECK_CURESULT(cuDevicePrimaryCtxRetain(&cu_ctx, 0));
          CHECK_CURESULT(cuCtxSetCurrent(cu_ctx));
        }
        CUmodule cu_mod;
        CHECK_CURESULT(
            cuModuleLoadData(&cu_mod, PyBytes_AS_STRING(bytes.ptr())));
        CUfunction cu_func;
        unsigned int count;
        CHECK_CURESULT(cuModuleGetFunctionCount(&count, cu_mod));
        CHECK_CURESULT(
            cuModuleGetFunction(&cu_func, cu_mod, func_name.c_str()));
        return reinterpret_cast<uintptr_t>(cu_func);
      });
}
} // namespace

PYBIND11_MODULE(_pydsl, m) {
  m.doc() = "Python binding for some MLIR API";
  init_mlir_helper(m);
}
