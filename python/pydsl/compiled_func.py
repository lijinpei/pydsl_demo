from .cache import CompilationCache
import importlib
from importlib import abc as importlib_abc
import torch
from .make_launcher import render_launcher
import tempfile
import _pydsl

class CompiledFunction:
    def decompose_dim3(self, dim3, args):
        if callable(dim3):
            dim3 = dim3(*args, **self.kwargs)

        if isinstance(dim3, tuple):
            len_dim3 = len(dim3)
            if len_dim3 == 3:
                return dim3[0], dim[1], dim[2]
            elif len_dim3 == 2:
                return dim3[0], dim[1], 1
            elif len_dim3 == 1:
                return dim3[0], 1, 1
            else:
                assert False
        return dim3, 1, 1

    def __init__(self, cubin_blob, comp_ctx):
        cache_mgr = CompilationCache(comp_ctx.name, comp_ctx.hash)
        def build_shared_lib():
            def render_this_launcher():
                res = render_launcher(comp_ctx)
                return res.encode('utf-8')
            cache_mgr.get_or_create('cpp', render_this_launcher)
            cpp_path = cache_mgr.get_file_path('cpp')
            from torch.utils.cpp_extension import load, _write_ninja_file_and_build_library
            with tempfile.TemporaryDirectory() as tmpdir_path:
                _write_ninja_file_and_build_library(name=comp_ctx.name, sources=[cpp_path], with_cuda=True, extra_cflags=None, extra_cuda_cflags=None, extra_include_paths=None, build_directory=tmpdir_path, extra_ldflags=["-lcuda"], verbose=True)
                with open(tmpdir_path + f'/{comp_ctx.name}.so', 'rb') as so_file:
                    return so_file.read()

        cache_mgr.get_or_create('so', build_shared_lib)
        mod_so_path = cache_mgr.get_file_path('so')
        spec = importlib.util.spec_from_file_location(comp_ctx.name, mod_so_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib_abc.Loader)
        spec.loader.exec_module(module)
        self.mod = module
        self.launch = self.mod.launch
        self.func_ptr = _pydsl.load_cubin(cubin_blob, comp_ctx.name)
        self.kwargs = comp_ctx.kwargs
        self.num_warps = comp_ctx.num_warps

    def run(self, /, *args, grid, stream = None):
        if stream is None:
            stream = torch.cuda.current_stream() 
        grid_x, grid_y, grid_z = self.decompose_dim3(grid, args)
        shared_memory_bytes = 0 # TODO
        self.launch(grid_x, grid_y, grid_z, self.num_warps, shared_memory_bytes, stream, self.func_ptr, *args)
