import mlir.ir
import mlir.extras.types
from typing import List
from mlir.dialects import gpu


class Type:
    def __init__(self):
        pass

class FP32Type(Type):
    def __init__(self):
        super().__init__()

    def to_ir(self, ctx:mlir.ir.Context):
        return mlir.ir.F32Type.get(ctx)

    def to_str(self):
        return "fp32"

    def to_launcher_arg(self):
        return "float"

    def destruct_launcher_arg(self, prefix):
        return ""

    def construct_launcher_arg(self, prefix):
        return "(void*)&" + prefix


class IntegerType(Type):
    def __init__(self, width: int, is_signed: bool):
        self.is_signed = is_signed
        self.width = width

    def to_ir(self, ctx:mlir.ir.Context):
        return mlir.ir.IntegerType.get_signless(self.width, ctx)

    def to_str(self):
        return f"{'i' if self.is_signed else 'u'}{self.width}"

    def to_launcher_arg(self):
        return "int"

    def destruct_launcher_arg(self, index):
        return ""

    def construct_launcher_arg(self, prefix):
        return "(void*)&" + prefix

class MemRefType(Type):
    def __init__(self, shape: List[int], elem_ty: Type, addr_space: int):
        self.shape = shape
        self.elem_ty = elem_ty
        self.addr_space = addr_space

    def to_ir(self, ctx: mlir.ir.Context):
        elem_ty = self.elem_ty.to_ir(ctx)
        return mlir.extras.types.memref(*[x for x in self.shape], element_type=elem_ty, memory_space=self.addr_space)

    def to_str(self):
        shape_str = "x".join([str(x) for x in self.shape])
        return f"memref<{shape_str}, {self.elem_ty.to_str()}, addr_space {self.addr_space}>"

    def to_launcher_arg(self):
        return "at::Tensor"

    def destruct_launcher_arg(self, prefix):
        res = ""
        res += f"void * {prefix}_ptr = {prefix}.data_ptr();\n"
        for i, s in enumerate(self.shape):
            res += f"int64_t {prefix}_size{i} = {prefix}.size({i});\n"
        for i, s in enumerate(self.shape):
            res += f"int64_t {prefix}_stride{i} = {prefix}.stride({i});\n"
        return res

    def construct_launcher_arg(self, prefix):
        res = f"(void*)&{prefix}_ptr, (void*)&{prefix}_ptr"
        for i, s in enumerate(self.shape):
            res += f", (void*)&{prefix}_size{i}"
        for i, s in enumerate(self.shape):
            res += f", (void*)&{prefix}_stride{i}"
        return res


class BuiltinFunc:
    def __init__(self, handler):
        self.handler = handler

def builtin(handler):
    return BuiltinFunc(handler)

@builtin
def lane_id(*, builder):
    return gpu.lane_id()

@builtin
def thread_id(dim, *, builder):
    return gpu.thread_id(dim)

@builtin
def block_id(dim, *, builder):
    return gpu.block_id(dim)

@builtin
def block_dim(dim, *, builder):
    return gpu.block_dim(dim)

@builtin
def grid_dim(dim, *, builder):
    return gpu.grid_dim(dim)

@builtin
def printf(format_str, *args, builder):
    gpu.printf(mlir.ir.StringAttr.get(format_str), args)
