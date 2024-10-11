#from libpydsl import dump_mlir, parse_mlir
import functools
from dataclasses import dataclass
from .language import *
from .cache import CompilationCache
from .mlir_generator import MLIRGenerator
from .compiler_context import CompilerContext
from .compiled_func import CompiledFunction
from typing import Callable, Any
import torch
import inspect
import ast
import hashlib
import _pydsl
import tempfile
import subprocess
import os

@dataclass
class CompilerStage:
    suffix: str
    lower_fn: Callable[[Any, CompilerContext], Any]
    opt_fn: Callable[[Any, CompilerContext], Any]
    serialize_fn: Callable[[Any, CompilerContext], bytes]
    deserialize_fn: Callable[[bytes, CompilerContext], Any]


def ast_to_mlir(_, comp_ctx):
    fn_ast = comp_ctx.fn_ast
    gen = MLIRGenerator(comp_ctx)
    return gen.to_mlir()

def opt_mlir(mlir_mod, comp_ctx):
    return mlir_mod

def dump_mlir(mlir_mod, comp_ctx):
    return str(mlir_mod)

def parse_mlir(mlir_blob, comp_ctx):
    with comp_ctx.mlir_ctx:
        return mlir.ir.Module.parse(mlir_blob.decode())

def mlir_to_llvm(mlir_mod, comp_ctx):
    res = _pydsl.translateMLIRModToLLVMIR(mlir_mod.operation._CAPIPtr)
    print('mlir_to_llvm', res)
    return res

def opt_llvm(mlir_mod, comp_ctx):
    return mlir_mod

def dump_llvm(llvm_mod, comp_ctx):
    return llvm_mod

def parse_llvm(llvm_mod, comp_ctx):
    return llvm_mod

def llvm_to_ptx(llvm_mod, comp_ctx):
    return _pydsl.translateLLVMIRToPtx(llvm_mod)

def opt_ptx(ptx, comp_ctx):
    return ptx

def dump_ptx(ptx, comp_ctx):
    return ptx

def parse_ptx(ptx, comp_ctx):
    return ptx

def ptx_to_cubin(ptx, comp_ctx):
    if isinstance(ptx, str):
        ptx = ptx.encode('utf-8')
    with tempfile.NamedTemporaryFile(mode='w+b') as fin:
        fout_path = fin.name + ".cubin"
        fin.write(ptx)
        fin.flush()
        ptxas_cmd = ['ptxas', '-o', fout_path, fin.name, '--gpu-name=sm_89']
        subprocess.run(ptxas_cmd, check=True)
        with open(fout_path, 'rb') as fout:
            cubin = fout.read()
        if os.path.exists(fout_path):
            os.remove(fout_path)
    return cubin

def opt_cubin(cubin, comp_ctx):
    return cubin

def dump_cubin(cubin, comp_ctx):
    return cubin

def parse_cubin(cubin, comp_ctx):
    return cubin

mlir_stage = CompilerStage('mlir', ast_to_mlir, opt_mlir, dump_mlir, parse_mlir)
llvm_stage = CompilerStage('ll', mlir_to_llvm, opt_llvm, dump_llvm, parse_llvm)
ptx_stage = CompilerStage('ptx', llvm_to_ptx, opt_ptx, dump_ptx, parse_ptx)
cubin_stage = CompilerStage('cubin', ptx_to_cubin, opt_cubin, dump_cubin, parse_cubin)

all_stages = [mlir_stage, llvm_stage, ptx_stage, cubin_stage]


def do_compile(fn_info, *args, **kwargs):
    comp_ctx = CompilerContext(fn_info, *args, **kwargs)
    cache_mgr = CompilationCache(comp_ctx.name, comp_ctx.hash)
    curr_ir = None
    for stage in all_stages:
        def compile_stage():
            nonlocal next_ir
            next_ir = stage.opt_fn(stage.lower_fn(curr_ir, comp_ctx), comp_ctx)
            return stage.serialize_fn(next_ir, comp_ctx)
        next_ir = None
        next_blob = cache_mgr.get_or_create(stage.suffix, compile_stage)
        if next_ir is None:
            curr_ir = stage.deserialize_fn(next_blob, comp_ctx)
        else:
            curr_ir = next_ir
    return CompiledFunction(curr_ir, comp_ctx)

