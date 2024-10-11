import ast
from typing import Dict, Any, List
from . import language
import mlir
from mlir.dialects import builtin, scf, func, arith
from mlir.dialects.arith import _is_float_type
import contextlib
from .compiler_context import CompilerContext

class MLIRGenerator(ast.NodeVisitor):

    comp_op_pred = {
            ast.Eq: arith.CmpIPredicate.eq,
            ast.Lt: arith.CmpIPredicate.slt,
            ast.LtE: arith.CmpIPredicate.sle,
            }

    def get_loc(self, node: ast.AST):
        line_no = self.line_no + node.lineno - 1
        return mlir.ir.Location.file(self.file_name, line_no, node.col_offset)

    def __init__(self, comp_ctx: CompilerContext):
        self.fn = comp_ctx.fn_ast
        self.signature = comp_ctx.signature
        self.constants = comp_ctx.kwargs
        self.global_scope = comp_ctx.global_scope
        self.file_name = comp_ctx.file_name
        self.line_no = comp_ctx.line_no
        self.ctx = comp_ctx.mlir_ctx

    def to_mlir(self):
        print(ast.dump(self.fn))
        self.met_func = None
        # TODO: SSA
        self.val_map = {}
        with self.ctx, mlir.ir.Location.file(self.file_name, self.line_no, 0):
            self.mod = builtin.ModuleOp()
            with mlir.ir.InsertionPoint.at_block_begin(self.mod.body):
                for x in self.fn.body:
                    self.visit(x)
        print('mod:', self.mod)
        return self.mod

    def assign_val(self, name, val):
        self.val_map[name] = val

    def get_val(self, name):
        return self.val_map[name]

    def visit(self, node):
        if hasattr(node, 'lineno'):
            ctx = self.get_loc(node)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return super().visit(node)

    def _visit_statements(self, stmts: List[ast.AST]):
        for stmt in stmts:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        print(ast.dump(node))
        if self.met_func:
            # no sub func def
            assert False
        self.met_func = node
        func_ty = func.FunctionType.get(inputs = [x.to_ir(self.ctx) for x in self.signature], results = [])
        func_op = func.FuncOp(node.name, func_ty)
        arg_nodes = node.args.posonlyargs + node.args.args
        arg_locs = [self.get_loc(arg) for arg in arg_nodes]
        func_op.add_entry_block(arg_locs)
        for arg_val, arg in zip(func_op.arguments, arg_nodes):
            self.assign_val(arg.arg, arg_val)
        with mlir.ir.InsertionPoint.at_block_begin(func_op.entry_block):
            self._visit_statements(node.body)
            func.return_([])

    def visit_If(self, node: ast.If):
        print('test:', ast.dump(node.test))
        cond = self.visit(node.test)
        cond = self.to_i1(cond)
        has_else = node.orelse is not None
        if_op = scf.IfOp(cond, hasElse=has_else)
        with mlir.ir.InsertionPoint.at_block_begin(if_op.then_block):
            self._visit_statements(node.body)
            scf.yield_([])
        if has_else:
            with mlir.ir.InsertionPoint.at_block_begin(if_op.else_block):
                self._visit_statements(node.orelse)
                scf.yield_([])

    def resolve_attr(self, attr):
        if isinstance(attr, ast.Attribute):
            obj = self.resolve_attr(attr.value)
            return getattr(obj, attr.attr)
        if isinstance(attr, ast.Name):
            return self.global_scope[attr.id]
        assert False

    def visit_For(self, node: ast.For):
        pass

    def visit_Call(self, node: ast.Call):
        callee = self.resolve_attr(node.func)
        def to_builtin_arg(x):
            if isinstance(x, ast.Constant) and isinstance(x.value, str):
                return x.value
            else:
                return self.visit(x)
        if isinstance(callee, language.BuiltinFunc):
            args = [to_builtin_arg(x) for x in node.args]
            kwargs = {**{keyword.arg: to_builtin_arg(keyword.value) for keyword in node.keywords}, 'builder': self}
            return callee.handler(*args, **kwargs)
        else:
            assert False
        print('call:', ast.dump(node))

    def to_i1(self, val):
        ty = val.type
        i1_ty = mlir.ir.IntegerType.get_signless(1)
        if ty != i1_ty:
            zero_val = arith.constant(ty, 0)
            return arith.cmpi(arith.CmpIPredicate.eq, val, zero_val)
        else:
            return val

    def to_fp_cast(self, val):
        ty = val.type
        if isinstance(ty, mlir.ir.F32Type):
            return val
        if isinstance(ty, mlir.ir.IndexType):
            return self.to_fp_cast(self.to_uint_cast(val))
        if isinstance(ty, mlir.ir.IntegerType):
            if ty.is_signed:
                return arith.sitofp(mlir.ir.F32Type.get(), val)
            else:
                return arith.uitofp(mlir.ir.F32Type.get(), val)
        assert False

    def to_uint_cast(self, val):
        ty = val.type
        if isinstance(ty, mlir.ir.F32Type):
            return arith.fptoui(mlir.ir.IntegerType.get_unsigned(32))
        if isinstance(ty, mlir.ir.IndexType):
            return arith.index_castui(mlir.ir.IntegerType.get_unsigned(32), val)
        if isinstance(ty, mlir.ir.IntegerType):
            if ty.is_signed:
                return arith.bitcast(mlir.ir.IntegerType.get_unsigned(32), val)
            else:
                return val
        assert False

    def to_sint_cast(self, val):
        ty = val.type
        if isinstance(ty, mlir.ir.F32Type):
            return arith.fptoui(mlir.ir.IntegerType.get_signed(32))
        if isinstance(ty, mlir.ir.IndexType):
            return arith.index_castis(mlir.ir.IntegerType.get_signed(32), val)
        if isinstance(ty, mlir.ir.IntegerType):
            if ty.is_signed:
                return arith.bitcast(mlir.ir.IntegerType.get_signed(32), val)
            else:
                return val
        assert False

    def to_index_cast(self, val):
        ty = val.type
        if isinstance(ty, mlir.ir.IndexType):
            return val
        if isinstance(ty, mlir.ir.IntegerType):
            if ty.is_signed:
                return arith.index_castsi(mlir.ir.IndexType.get(), val)
            else:
                return arith.index_castui(mlir.ir.IndexType.get(), val)

    def visit_Assign(self, node: ast.Assign):
        target = node.targets
        assert len(target) == 1
        target = target[0]
        assert isinstance(target, ast.Name)
        val = self.visit(node.value)
        self.assign_val(target.id, val)


    def promote_to_common_type(self, val1, val2):
        print(val1, val2)
        ty1 = val1.type
        ty2 = val2.type
        if ty1 == ty2:
            return val1, val2
        if _is_float_type(ty1):
            return val1, self.to_fp_cast(val2)
        if _is_float_type(ty2):
            return self.to_fp_cast(val1), val2
        if isinstance(ty1, mlir.ir.IndexType):
            return val1, self.to_index_cast(val2)
        if isinstance(ty2, mlir.ir.IndexType):
            return self.to_index_cast(val1), val2
        if isinstance(ty1, mlir.ir.IntegerType) and isinstance(ty2, mlir.ir.IntegerType):
            if ty1.is_unsigned or ty2.is_unsigned:
                return self.to_uint_cast(val1), self.to_uint_cast(val2)
            if ty1.is_signed or ty2.is_signed:
                return self.to_sint_cast(val1), self.to_sint_cast(val2)
        assert False

    def visit_Name(self, node: ast.Name):
        return self.val_map[node.id]

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        if isinstance(val, float):
            return arith.constant(mlir.ir.F32Type, val)
        elif isinstance(val, int):
            assert 3000 >= val >= -3000
            return arith.constant(mlir.ir.IntegerType.get_signless(32), val)
        assert False


    def visit_Compare(self, node: ast.Compare):
        print('comp: ',  ast.dump(node))
        lhs = node.left
        op = node.ops
        assert len(op) == 1
        op = op[0]
        rhs = node.comparators
        assert len(rhs) == 1
        rhs = rhs[0]
        lhs_val = self.visit(lhs)
        rhs_val = self.visit(rhs)
        lhs_val, rhs_val = self.promote_to_common_type(lhs_val, rhs_val)
        if _is_float_type(lhs_val.type):
            op_ctor = arith.cmpf
        else:
            op_ctor = arith.cmpi
        pred = self.comp_op_pred[type(op)]
        return op_ctor(pred, lhs_val, rhs_val)

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        lhs, rhs = self.promote_to_common_type(lhs, rhs)
        ty = lhs.type
