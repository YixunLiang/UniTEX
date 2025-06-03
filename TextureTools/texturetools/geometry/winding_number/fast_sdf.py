
'''
Convert mesh to sdf field with ray tracing and 
extract iso field to mesh with sparse grid marching cubes
'''

from functools import partial
from glob import glob
from itertools import product
import json
import math
import os
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm
import trimesh
import open3d as o3d
import pygltflib
from scipy.sparse import coo_matrix, csgraph
from timeout_decorator import timeout

# https://github.com/nv-tlabs/FlexiCubes.git
# from flexicubes import FlexiCubes

# https://github.com/libigl/libigl.git
# import igl

# https://github.com/ashawkey/cubvh.git
# https://github.com/opencv/opencv/issues/14868
from cubvh import cuBVH

# https://github.com/lcp29/trimesh-ray-optix
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# export OptiX_INSTALL_DIR=${HOME}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
from triro.ray.ray_optix import RayMeshIntersector

# https://pytorch-geometric.readthedocs.io/en/1.3.1/index.html
# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
import torch_cluster

# https://github.com/YuliangXiu/bvh-distance-queries.git
# https://github.com/NVIDIA/cuda-samples.git
# export CUDA_SAMPLES_INC=${HOME}/cuda-samples-12.4.1/Common
# https://github.com/LCH1238/bevdet-tensorrt-cpp/issues/18
# remove nvcc compile args '-DNUM_THREADS=256' in setup.py
# from bvh_distance_queries import BVH

# https://docs.cupy.dev/en/stable/install.html
# pip install cupy-cuda12x
import cupy as cp
# https://docs.rapids.ai/api/cugraph/stable/basics
# pip install cugraph-cu12
import cugraph

# https://kaolin.readthedocs.io/en/stable/index.html
# pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
# import kaolin

from ...raytracing import RayTracing










def sdf_rt_optix(
    vertices:torch.Tensor, 
    faces:torch.Tensor, 
    points:torch.Tensor, 
    n_rays=8,
    n_rays_threshold=8,
    max_iter=10,
    epsilon=1e-5,
    visible_check=True,
    chunk_size:Optional[int]=None,
) -> torch.Tensor:
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    points: [N, 3], float32
    n_rays: total num of rays per point
    n_rays_threshold: inner points satisfy num of inside rays >= threshold
    max_iter: max num of iterations
    epsilon: offset of origins along normal on the reflected side
    visible_check: enable visible check for the first hit
    chunk_size: chunk size of points

    sdf: [N,], float32
    '''
    assert n_rays_threshold <= n_rays
    chunk_size = chunk_size or points.shape[0]
    bvh = RayMeshIntersector(vertices=vertices, faces=faces)
    bvh_invalid_id = -1
    normals = torch.linalg.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]], dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    generator = torch.Generator(device='cuda').manual_seed(666)

    points_cnt = torch.zeros((points.shape[0],), dtype=torch.int64, device='cuda')
    points_index = torch.arange(points.shape[0], dtype=torch.int64, device='cuda')
    for idx in tqdm(range(0, points.shape[0], chunk_size), desc='sdf_rt_optix'):
        _points = points[idx:idx+chunk_size]
        _points_index = points_index[idx:idx+chunk_size]

        rays_o = _points.repeat_interleave(n_rays, dim=0)
        rays_d = torch.randn(rays_o.shape, dtype=torch.float32, device='cuda', generator=generator)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        rays_mask, _, tid, rays_h, _ = bvh.intersects_closest(rays_o, rays_d, stream_compaction=False)
        if visible_check:
            rays_mask = rays_mask.reshape(_points.shape[0], n_rays).all(dim=-1)
            rays_idx = _points_index[rays_mask]
            if max_iter > 1:
                rays_mask = rays_mask.repeat_interleave(n_rays, dim=0)
                rays_idx = rays_idx.repeat_interleave(n_rays, dim=0)
            else:
                points_cnt.scatter_(0, rays_idx, n_rays, reduce='add')
        else:
            rays_idx = _points_index.repeat_interleave(n_rays, dim=0)[rays_mask]
        for iter in range(max_iter - 1):
            if rays_idx.shape[0] == 0:
                break
            rays_n = normals[tid[rays_mask], :]
            rays_d = rays_d[rays_mask, :]
            rays_h = rays_h[rays_mask, :]
            rays_cos = torch.linalg.vecdot(rays_d, rays_n, dim=-1)
            rays_d = rays_d - 2.0 * rays_cos.unsqueeze(-1) * rays_n
            rays_o = rays_h - epsilon * torch.sign(rays_cos).unsqueeze(-1) * rays_n
            rays_mask, _, tid, rays_h, _ = bvh.intersects_closest(rays_o, rays_d, stream_compaction=False)
            rays_idx = rays_idx[rays_mask]
        if max_iter > 1:
            points_cnt.scatter_(0, rays_idx, 1, reduce='add')

    bvh = cuBVH(vertices=vertices, triangles=faces)
    sdf, _, _ = bvh.unsigned_distance(points, return_uvw=False)
    sdf.scatter_(0, torch.where(points_cnt >= n_rays_threshold)[0], -1.0, reduce='multiply')
    return sdf









