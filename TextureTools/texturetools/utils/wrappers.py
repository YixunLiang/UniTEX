from typing import Tuple
import torch


def torch_wrapper(func_np, *args):
    rets_np = func_np(*[arg.detach().cpu().numpy() for arg in args])
    if isinstance(rets_np, Tuple):
        return [torch.as_tensor(ret, device=args[0].device, dtype=args[0].dtype) for ret in rets_np]
    else:
        return torch.as_tensor(rets_np, device=args[0].device, dtype=args[0].dtype)

