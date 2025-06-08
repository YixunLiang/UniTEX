'''
fast knn based on faiss
'''

from typing import Optional, Tuple
import torch

# NOTE: pip install faiss-gpu-cu12
import faiss
# NOTE: TypeError: can't convert cuda:0 device type tensor to numpy.
import faiss.contrib.torch_utils
# faiss.omp_set_num_threads(16)


class KDTree:
    def __init__(self, index_factory:str='Flat', reserve:Optional[int]=None):
        '''
        index_factory: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        reserve: the number of elements to reserve memory for before re-allocating (GPU-only).
        '''
        self.index_factory = index_factory
        self.index:Optional[faiss.Index] = None
        self.reserve = reserve

    def add_emb(self, emb:torch.Tensor):
        if self.index is None:
            self.index = faiss.index_factory(emb.shape[1], self.index_factory)
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                emb.device.index,
                self.index,
            )
            if self.reserve is not None:
                if hasattr(self.index, 'reserveMemory'):
                    self.index.reserveMemory(self.reserve)
            self.index.train(emb.detach().contiguous())
        self.index.add(emb.detach().contiguous())
        return self

    def get_emb(self) -> Optional[torch.Tensor]:
        if self.index is not None:
            return self.index.reconstruct_n(0, self.index.ntotal)

    def search_emb(self, emb:torch.Tensor, k:int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.index is not None
        score, index = self.index.search(emb.detach().contiguous(), k)
        return score, index

