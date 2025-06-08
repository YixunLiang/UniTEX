from functools import wraps
from time import perf_counter
from typing import TypeVar
import torch

F = TypeVar('F')
def decorator(cls, func: F) -> F:
    @wraps(func)
    def decorated_func(*args, **kwargs):
        with cls:
            return func(*args, **kwargs)
    return decorated_func

class CPUTimer:
    def __init__(self, prefix='', synchronize=False):
        self.prefix = prefix
        self.synchronize = synchronize

    def __enter__(self):
        if self.synchronize:
            torch.cuda.synchronize()
        self.t = perf_counter()

    def __exit__(self, _type, _value, _traceback):
        if self.synchronize:
            torch.cuda.synchronize()
        t = perf_counter() - self.t
        print('>>>', self.prefix, t, '>>>')

    def __call__(self, func: F) -> F:
        return decorator(self, func)
