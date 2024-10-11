import pydsl
import pydsl.language as pl
import torch

@pydsl.jit
def my_kernel(a, b, /, c, *, c0, c1):
    if b < 1:
        pass
    elif c == 0:
        pass
    pl.thread_id('x')
    pl.block_id('y')

a = torch.ones((5, 2), dtype=torch.float32)
my_kernel(a, 1, 1., grid=(1,), num_warps=1)
