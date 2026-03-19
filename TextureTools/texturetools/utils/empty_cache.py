import gc
import torch


def empty_cache():
    mem = 24 * 1024 * 1024 * 1024
    while True:
        mem_cur = torch.cuda.memory_allocated()
        if mem_cur == mem:
            break
        mem = mem_cur
        torch.cuda.empty_cache()
    gc.collect()

