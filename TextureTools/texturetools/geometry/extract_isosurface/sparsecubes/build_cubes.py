from itertools import product
from typing import Optional
import torch


def build_cubes(
    surface_points:torch.Tensor, 
    depth=8, 
    depth_refine:Optional[int]=10, 
    expand:Optional[int]=1, 
    expand_refine:Optional[int]=2, 
):
    '''
    Assign surface points to cubes

    surface_points: [N, 3], float32
    cube_vertices: [V, 3], float32
    cube_faces: [C, 8], int64
    '''
    b = (2 ** depth)
    bx, by, bz = b ** 2, b, 1
    indices = torch.floor((torch.clamp(surface_points, -1.0, 1.0) * 0.5 + 0.5) * b).to(dtype=torch.int64)
    indices = torch.unique(indices, dim=0)
    if expand is not None or depth_refine is not None:
        # NOTE: abandon conv3d/max_pool because of OOM
        bool_grid = torch.zeros((b, b, b), dtype=torch.bool, device='cuda')
        indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
        expand_basis = list(product(*([range(-expand, expand+1)] * 3)))
        bool_grid = bool_grid.reshape(-1)
        for i, basis in enumerate(expand_basis):
            bool_grid.scatter_(0, torch.clamp(indices_spatial + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
        bool_grid = bool_grid.reshape(b, b, b)
        if depth_refine is not None:
            depth_delta = depth_refine - depth
            b = (2 ** depth_refine)
            bx, by, bz = b ** 2, b, 1
            bool_grid = bool_grid[:, None, :, None, :, None].repeat(1, 2 ** depth_delta, 1, 2 ** depth_delta, 1, 2 ** depth_delta).reshape(b, b, b)
            if expand_refine is not None:
                indices = torch.nonzero(bool_grid)
                indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
                expand_basis = list(product(*([range(-expand_refine, expand_refine+1)] * 3)))
                for i, basis in enumerate(expand_basis):
                    bool_grid.reshape(-1).scatter_(0, torch.clamp(indices_spatial + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
        indices = torch.nonzero(bool_grid)
    b = b + 1
    bx, by, bz = b ** 2, b, 1
    indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
    expand_basis = list(product(*([[0, 1]] * 3)))
    indices_spatial = torch.tile(indices_spatial[:, None], (1, len(expand_basis)))
    for i, basis in enumerate(expand_basis):
        indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
    indices_spatial_unique, indices_spatial_inverse = torch.unique(indices_spatial.reshape(-1), dim=0, return_inverse=True)
    indices_unique = torch.stack([indices_spatial_unique // basis % b for basis in [bx, by, bz]], dim=-1)
    cube_vertices = (indices_unique.to(dtype=torch.float32) / (b - 1)) * 2.0 - 1.0
    cube_faces = indices_spatial_inverse.reshape(-1, 8)
    return cube_vertices, cube_faces

