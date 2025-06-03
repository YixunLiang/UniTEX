'''
KNN of different implements for pytorch
'''
from tqdm import tqdm
import math
from time import perf_counter
from typing import Tuple
import numpy as np
import torch
import cupy as cp


class KNN:
    def __init__(self, src:torch.Tensor, backend='torch_kdtree'):
        '''
        src: [N, C]
        backend: scipy, cupy, faiss, pytorch
        '''
        N, C = src.shape
        t = perf_counter()
        if backend in ['scipy']:
            from scipy.spatial import KDTree
            # NOTE: https://github.com/scipy/scipy/issues/18467
            self.kdtree = KDTree(src.detach().cpu().numpy().astype(np.float64), leafsize=16)
        elif backend in ['cupy']:
            # https://docs.cupy.dev/en/stable/install.html
            # pip install cupy-cuda12x
            from .knn_cupy import KDTree as KDTreeCUDA
            self.kdtree = KDTreeCUDA(cp.asarray(src, dtype=cp.float32), leafsize=10)
        elif backend in ['faiss']:
            # pip install faiss-gpu-cu12
            from .knn_faiss import KDTree
            self.kdtree = KDTree().add_emb(src)
        elif backend in ['torch', 'pytorch']:
            self.kdtree = src
        elif backend in ['torch_kdtree']:
            # https://github.com/thomgrand/torch_kdtree
            from torch_kdtree import build_kd_tree
            self.kdtree = build_kd_tree(src)
        else:
            raise NotImplementedError(f'backend {backend} is not supported')
        dt1 = perf_counter() - t
        self.backend = backend
        self.N = N
        self.C = C
        self.dt1 = dt1
    
    def __call__(self, dst:torch.Tensor, k:int=1, batch_size=None, return_dt=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        dst: [M, C]
        distance: [N, M]
        score: [M, K]
        index: [M, K]
        '''
        M, C = dst.shape
        assert C == self.C,f'expected num of channel is {self.C} but {C}'
        device = dst.device
        t = perf_counter()
        if self.backend in ['scipy']:
            if batch_size is None:
                score, index = self.kdtree.query(dst.detach().cpu().numpy().astype(np.float64), k=k, workers=8, distance_upper_bound=2 * math.sqrt(3))
            else:
                score = np.zeros((M, k), dtype=np.float64)
                index = np.zeros((M, k), dtype=np.int64)
                dst = dst.detach().cpu().numpy()
                for idx in tqdm(range(0, M, batch_size), 'knn'):
                    score[idx:idx+batch_size, :], index[idx:idx+batch_size, :] = self.kdtree.query(dst[idx:idx+batch_size, :], k=k, workers=8, distance_upper_bound=2 * math.sqrt(3))
            score = torch.as_tensor(score, dtype=torch.float32, device=device)
            index = torch.as_tensor(index, dtype=torch.int64, device=device)
        elif self.backend in ['cupy']:
            if batch_size is None:
                score, index = self.kdtree.query(cp.asarray(dst, dtype=cp.float32), k=k)
            else:
                score = cp.zeros((M, k), dtype=cp.float32)
                index = cp.zeros((M, k), dtype=cp.int64)
                dst = cp.asarray(dst, dtype=cp.float32)
                for idx in tqdm(range(0, M, batch_size), 'knn'):
                    score[idx:idx+batch_size, :], index[idx:idx+batch_size, :] = self.kdtree.query(dst[idx:idx+batch_size, :], k=k, workers=8, distance_upper_bound=2 * math.sqrt(3))
            score = torch.as_tensor(score, dtype=torch.float32, device=device)
            index = torch.as_tensor(index, dtype=torch.int64, device=device)
            dt2 = perf_counter() - t
        elif self.backend in ['faiss']:
            if batch_size is None:
                score, index = self.kdtree.search_emb(dst, k=k)
            else:
                score = torch.zeros((M, k), dtype=torch.float32, device=device)
                index = torch.zeros((M, k), dtype=torch.int64, device=device)
                for idx in tqdm(range(0, M, batch_size), 'knn'):
                    score[idx:idx+batch_size, :], index[idx:idx+batch_size, :] = self.kdtree.query(dst[idx:idx+batch_size, :], k=k, workers=8, distance_upper_bound=2 * math.sqrt(3))
        elif self.backend in ['torch', 'pytorch']:
            distance = torch.cdist(self.kdtree, dst)
            score, index = distance.topk(k=k, dim=-2, sorted=True)
        elif self.backend in ['torch_kdtree']:
            score, index = self.kdtree.query(dst, nr_nns_searches=k)
            index = index.to(dtype=torch.int64)
        else:
            raise NotImplementedError(f'backend {self.backend} is not supported')
        dt2 = perf_counter() - t
        if return_dt:
            return score, index, (self.dt1, dt2)
        return score, index


def knn(src:torch.Tensor, dst:torch.Tensor, k:int=1, backend='torch_kdtree', batch_size=None, return_dt=False) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    src: [N, C]
    dst: [M, C]
    backend: scipy, cupy, faiss, pytorch

    distance: [N, M]
    score: [M, K]
    index: [M, K]
    '''
    return KNN(src, backend=backend)(dst, k=k, batch_size=batch_size, return_dt=return_dt)

