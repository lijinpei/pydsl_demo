from .compiler import do_compile
import inspect

def jit(fn):
    call_stack = inspect.stack()
    caller_frame = call_stack[1]
    return JitFunc(fn, caller_frame.frame.f_globals, caller_frame.filename, caller_frame.lineno)

class JitFunc:
    def __init__(self, fn, global_scope, filename, lineno):
        self.fn = fn
        self.global_scope = global_scope
        self.file_name = filename
        self.lineno = lineno
        self.compiled_fn = None

    def __call__(self, *args, grid, **kwargs):
        if not self.compiled_fn:
            self.compiled_fn = do_compile((self.fn, self.global_scope, self.file_name, self.lineno), *args, **kwargs)
        self.compiled_fn.run(*args, grid=grid)
