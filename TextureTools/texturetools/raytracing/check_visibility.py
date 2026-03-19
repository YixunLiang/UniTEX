import math
from typing import Optional
import torch
from tqdm import tqdm

from . import RayTracing


def self_rt(
    bvh:RayTracing, 
    points:torch.Tensor, 
    n_rays=32, 
    chunk_size=1_000_000, 
    seed:Optional[int]=666, 
) -> torch.Tensor:
    '''
    scatter rays from points to anywhere and do self ray tracing

    points: [N, 3]
    n_rays: int
    seed: int
    points_mask: [N,], inner point mask
    '''
    generator = None if seed is None else torch.Generator(device='cuda').manual_seed(seed)

    def _self_rt(_points):
        batch_shape = _points.shape[:-1]
        rays_o = _points.reshape(-1, 3).repeat_interleave(n_rays, dim=0)
        rays_d = torch.randn(rays_o.shape, dtype=torch.float32, device='cuda', generator=generator)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        rays_mask, _, _, _, _ = bvh.intersects_closest(rays_o, rays_d, stream_compaction=False)
        _points_mask = rays_mask.reshape(-1, n_rays).all(dim=-1).reshape(*batch_shape)
        return _points_mask

    batch_shape = points.shape[:-1]
    points = points.reshape(-1, 3)
    points_mask = torch.zeros((points.shape[0],), dtype=torch.bool, device='cuda')
    for i in tqdm(range(0, points.shape[0], chunk_size)):
        _points = points[i:i+chunk_size, :]
        _points_mask = _self_rt(_points)
        points_mask[i:i+chunk_size] = _points_mask
    points_mask = points_mask.reshape(*batch_shape)
    return points_mask


def cross_rt(
    bvh:RayTracing, 
    points:torch.Tensor, 
    outer_points:torch.Tensor, 
    chunk_size=32, 
    exhaustive_mode=True, 
) -> torch.Tensor:
    '''
    scatter rays from outer points to points and do cross ray tracing

    points, outer_points: 
        exhaustive mode: [N, 3], [M, 3]
        non-exhaustive mode: [N, M, 3], [N, M, 3]
    chunk_size: int, split outer points into chunks
    exhaustive_mode: bool
    points_mask: [N,], inner point mask
    '''
    def _cross_rt(_points:torch.Tensor, _outer_points:torch.Tensor) -> torch.Tensor:
        '''
        _points: [..., N, 3]
        _outer_points: [..., N, 3]
        _points_mask: [...,], inner point mask
        '''
        _points, _outer_points = torch.broadcast_tensors(_points, _outer_points)
        rays_d = (_points - _outer_points).reshape(-1, 3)
        rays_o = _outer_points.reshape(-1, 3)
        rays_t = rays_d.norm(p=2.0, dim=-1)
        rays_d = rays_d / rays_t.unsqueeze(-1).clamp_min_(1e-12)
        _, _, _, rays_h, _ = bvh.intersects_closest(rays_o, rays_d, stream_compaction=False)
        rays_mask = ((rays_h - rays_o).norm(p=2.0, dim=-1) < rays_t)
        _points_mask = rays_mask.reshape(*_points.shape[:-1]).all(dim=-1)
        return _points_mask

    if exhaustive_mode:
        batch_shape = points.shape[:-1]
        points = points.reshape(-1, 3).unsqueeze(-2)
        outer_points = outer_points.reshape(-1, 3).unsqueeze(-3)
    else:
        batch_shape = points.shape[:-2]
        points = points.reshape(-1, points.shape[-2], 3)
        outer_points = outer_points.reshape(-1, outer_points.shape[-2], 3)
    points, outer_points = torch.broadcast_tensors(points, outer_points)
    points_idx = torch.arange(points.shape[0], dtype=torch.int64, device='cuda')
    for i in tqdm(range(0, outer_points.shape[1], chunk_size)):
        _points = points[:, i:i+chunk_size, :]
        _outer_points = outer_points[:, i:i+chunk_size, :]
        _points_mask = _cross_rt(_points, _outer_points)
        points = points[_points_mask, :, :]
        points_idx = points_idx[_points_mask]
    points_mask = torch.zeros(batch_shape, dtype=torch.bool, device='cuda')
    points_mask.reshape(-1).scatter_(0, points_idx, True)
    return points_mask


def sphere_rt(
    bvh: RayTracing, 
    n_rays=1_000, 
    sample_offset=0.0, 
) -> torch.Tensor:
    '''
    scatter rays from sphere points to origin and do sphere ray tracing

    n_rays: num of rays
    rays_tid: [N,], int64
    '''
    radius = math.sqrt(3) * (1.0 + sample_offset)
    rays_d = torch.randn((n_rays, 3), dtype=torch.float32, device='cuda', generator=torch.Generator(device='cuda').manual_seed(666))
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = radius * rays_d
    rays_d = rays_d.neg()
    rays_mask, _, rays_tid, _, _ = bvh.intersects_closest(rays_o, rays_d)
    rays_tid = torch.masked_select(rays_tid, rays_mask)
    return rays_tid


def sphere_rt_nvdiffrast(
    bvh: RayTracing,
    n_cameras=6,
    sample_offset=0.0, 
) -> torch.Tensor:
    '''
    scatter rays from sphere points to origin and do sphere ray tracing

    n_cameras: num of cameras. if n_cameras is 4 or 6, using standard 4 views or 6 views.
    rays_tid: [N, H, W], int64, H and W are from nvdiffrt

    NOTE: when depth=10, using ~ 3 GB more memory than optix due to memory leak,
        refer to https://github.com/NVlabs/nvdiffrast/issues/30.
    '''
    radius = math.sqrt(3) * (1.0 + sample_offset)
    if n_cameras == 4:
        rays_o = torch.as_tensor([
            [radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, -radius, 0.0],
        ], dtype=torch.float32, device='cuda')
        rays_d = torch.as_tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32, device='cuda')
    elif n_cameras == 6:
        rays_o = torch.as_tensor([
            [radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ], dtype=torch.float32, device='cuda')
        rays_d = torch.as_tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device='cuda')
    else:
        rays_d = torch.randn((n_cameras, 3), dtype=torch.float32, device='cuda', generator=torch.Generator(device='cuda').manual_seed(666))
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        rays_o = radius * rays_d
        rays_d = rays_d.neg()
    rays_mask, _, rays_tid, _, _ = bvh.intersects_closest(rays_o, rays_d)
    rays_tid = torch.masked_select(rays_tid, rays_mask)
    return rays_tid


