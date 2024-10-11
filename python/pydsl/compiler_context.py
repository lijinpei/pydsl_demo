import functools
import inspect
import ast
import torch
import mlir.ir
from .language import *
import hashlib

def torch_dtype_to_ir_type(dtype):
    if dtype == torch.int32:
        return IntegerType(32, True)
    elif dtype == torch.float32:
        return FP32Type()
    assert False

def get_signature_from_arg(x):
    if isinstance(x, int):
        return IntegerType(32, True)
    if isinstance(x, float):
        return FP32Type()
    if isinstance(x, torch.Tensor):
        elem_ir_ty = torch_dtype_to_ir_type(x.dtype)
        return MemRefType(x.shape, elem_ir_ty, 1)
    assert False


class CompilerContext:
    def __init__(self, fn_info, *args, num_warps, **kwargs):
        fn, self.global_scope, self.file_name, self.line_no = fn_info
        self.name = fn.__name__
        self.fn_src = inspect.getsource(fn)
        self.fn_ast = ast.parse(self.fn_src)
        self.kwargs = kwargs
        self.signature = [get_signature_from_arg(x) for x in args]
        self.mlir_ctx = mlir.ir.Context()
        self.num_warps = num_warps

    @functools.cached_property
    def hash(self):
        res = hashlib.sha1()
        kwargs = [(key, value) for key, value in self.kwargs.items()]
        kwargs.sort(key=lambda x: x[0])
        res.update(self.fn_src.encode('utf-8'))
        for ty in self.signature:
            res.update(ty.to_str().encode('utf-8'))
        for kv in kwargs:
            res.update(kv[0].encode('utf-8'))
            res.update(str(kv[1]).encode('utf-8'))
        return res.hexdigest()

