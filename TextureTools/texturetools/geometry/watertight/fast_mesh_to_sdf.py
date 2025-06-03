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
from typing import List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
import trimesh
import open3d as o3d
from timeout_decorator import timeout

# https://github.com/nv-tlabs/FlexiCubes.git
from flexicubes import FlexiCubes

# https://github.com/libigl/libigl.git
import igl

# https://github.com/ashawkey/cubvh.git
# https://github.com/opencv/opencv/issues/14868
import cubvh

# https://github.com/lcp29/trimesh-ray-optix
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# export OptiX_INSTALL_DIR=${HOME}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
from triro.ray.ray_optix import RayMeshIntersector

# https://github.com/SarahWeiii/diso.git
from diso import DiffMC, DiffDMC


class FlexiCubesPlus(FlexiCubes):
    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        return (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)


def f_traverse(node, node_info, origins:List, sizes:List, leaf_only=True):
    early_stop = False
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if not leaf_only:
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                n = 0
                for child in node.children:
                    if child is not None:
                        n += 1
                origins.append(node_info.origin)
                sizes.append(node_info.size)
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            origins.append(node_info.origin)
            sizes.append(node_info.size)
    else:
        raise NotImplementedError(f'node type {type(node)} is not supported')
    return early_stop


def mesh_to_sdf_dense(
    input_mesh_path, 
    output_mesh_path, 
    scale:Optional[float]=0.95,
    depth=8,
    enable_diso=True,
    use_udf=True, 
    use_wn=False, 
    sdf_offset=1.0,
):
    '''
    input_mesh_path: normalized mesh path, range(-1.0, 1.0)
    output_mesh_path: reconstructed mesh path
    scale: apply scale on input mesh
    depth: depth of octree
    enable_diso: whether use diso as backend instead of flexicubes
    use_wn: whether use winding number to compute sdf, prior: udf > wn
    use_udf: whether use udf, prior: udf > wn
    sdf_offset: whether dilate sdf_offset * size_of_cube for sdf
    '''
    ## load whole mesh
    mesh_trimesh:trimesh.Trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')
    if scale is not None:
        mesh_trimesh = mesh_trimesh.apply_scale([scale] * 3)
    if use_wn:
        mesh_o3d = None
    else:
        mesh_o3d = o3d.io.read_triangle_mesh(input_mesh_path, enable_post_processing=False, print_progress=False)
        if scale is not None:
            mesh_o3d = mesh_o3d.scale(scale=scale, center=[0.0] * 3)

    if not use_wn:
        ## build kdtree
        mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d, vertex_dtype=o3d.core.float32, triangle_dtype=o3d.core.int64, device=o3d.core.Device('CPU:0'))
        kdtree_o3d = o3d.t.geometry.RaycastingScene()
        kdtree_o3d.add_triangles(mesh_o3d_t)

    ## run flexicubes plus or diso
    if enable_diso:
        x_nx3 = torch.nonzero(torch.ones([2 ** depth] * 3, device='cuda')).float() / (2 ** depth)
        x_nx3 = x_nx3.reshape(-1, 3) * 2.0 - 1.0
        cube_fx8 = None
        fc = DiffMC(dtype=torch.float32).cuda()
    else:
        fc = FlexiCubesPlus()
        x_nx3, cube_fx8 = fc.construct_voxel_grid(2 ** depth)    
        x_nx3 = x_nx3 * 2.0
    # sdf = torch.rand_like(x_nx3[:,0]) - 0.1  # randomly initialize sdf
    if use_wn:
        # NOTE: https://github.com/libigl/libigl/blob/main/include/igl/signed_distance.h#L31
        sdf, _, _ = igl.signed_distance(x_nx3.detach().cpu().numpy(), mesh_trimesh.vertices, mesh_trimesh.faces, return_normals=False, sign_type=4)
        sdf = torch.as_tensor(sdf, dtype=torch.float32, device='cuda')
    elif use_udf:
        sdf = kdtree_o3d.compute_distance(o3d.core.Tensor(x_nx3.detach().cpu().numpy(), dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))).numpy()
        sdf = torch.as_tensor(sdf, dtype=torch.float32, device='cuda')
    else:
        sdf = kdtree_o3d.compute_signed_distance(query_points=o3d.core.Tensor(x_nx3.detach().cpu().numpy(), dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))).numpy()
        sdf = torch.as_tensor(sdf, dtype=torch.float32, device='cuda')
    if sdf_offset != 0.0:
        sdf = sdf - sdf_offset * (2 / (2 ** depth))
    if enable_diso:
        t = perf_counter()
        sdf = sdf.reshape(2 ** depth, 2 ** depth, 2 ** depth)
        vertices, faces = fc(sdf, deform=None, normalize=True)
        vertices = vertices * 2.0 - 1.0
        print('diso dense waste', perf_counter() - t, 'sec')  # ~ 0.00153 sec for depth = 8
    else:
        t = perf_counter()
        vertices, faces, _ = fc(x_nx3, sdf, cube_fx8, 2 ** depth)
        print('flexicubes dense waste', perf_counter() - t, 'sec')  # ~ 0.0744 sec for depth = 8
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    ## export reconstructed mesh
    mesh_reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if use_udf:
        mesh_reconstructed = mesh_reconstructed.split(only_watertight=False)
        mesh_reconstructed = mesh_reconstructed[np.argmax([m.faces.shape[0] for m in mesh_reconstructed])]
    if scale is not None:
        mesh_reconstructed = mesh_reconstructed.apply_scale([1.0 / scale] * 3)
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    mesh_reconstructed.export(output_mesh_path)


def to_tensor_f(vertices:Union[torch.Tensor, np.ndarray], device='cuda'):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.to(dtype=torch.float32, device=device)
    else:
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    return vertices

def to_tensor_i(faces:Union[torch.Tensor, np.ndarray], device='cuda'):
    if isinstance(faces, torch.Tensor):
        faces = faces.to(dtype=torch.int64, device=device)
    else:
        faces = torch.as_tensor(faces, dtype=torch.int64, device=device)
    return faces

def to_array_f(vertices:Union[torch.Tensor, np.ndarray]):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy().astype(np.float32)
    elif isinstance(vertices, np.ndarray):
        vertices = vertices.astype(np.float32)
    else:
        vertices = np.asarray(vertices, dtype=np.float32)
    return vertices

def to_array_i(faces:Union[torch.Tensor, np.ndarray]):
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy().astype(np.int64)
    elif isinstance(faces, np.ndarray):
        faces = faces.astype(np.int64)
    else:
        faces = np.asarray(faces, dtype=np.int64)
    return faces


def sdf_rt(
    vertices:Union[torch.Tensor, np.ndarray], 
    faces:Union[torch.Tensor, np.ndarray], 
    points:Union[torch.Tensor, np.ndarray], 
    n_rays=64,
    n_rays_threshold=32,
    max_iter=100,
    epsilon=1e-5,
    visible_check=True,
    chunk_size:Optional[int]=1_000_000,
    backend=3,
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
    backend: cubvh(0), open3d(1), trimesh(2), optix(3)

    sdf: [N,], float32
    '''
    assert n_rays_threshold <= n_rays
    kwargs = dict(
        n_rays=n_rays,
        n_rays_threshold=n_rays_threshold,
        max_iter=max_iter,
        epsilon=epsilon,
        visible_check=visible_check,
        chunk_size=chunk_size,
    )
    if backend == 0:
        return sdf_rt_cubvh(
            vertices=to_tensor_f(vertices, device='cuda'),
            faces=to_tensor_i(faces, device='cuda'),
            points=to_tensor_f(points, device='cuda'),
            **kwargs,
        )
    elif backend == 1:
        return to_tensor_f(sdf_rt_o3d(
            vertices=to_array_f(vertices),
            faces=to_array_i(faces),
            points=to_array_f(points),
            **kwargs,
        ), device='cuda')
    elif backend == 2:
        return to_tensor_f(sdf_rt_trimesh(
            vertices=to_array_f(vertices),
            faces=to_array_i(faces),
            points=to_array_f(points),
            **kwargs,
        ), device='cuda')
    elif backend == 3:
        return sdf_rt_optix(
            vertices=to_tensor_f(vertices, device='cuda'),
            faces=to_tensor_i(faces, device='cuda'),
            points=to_tensor_f(points, device='cuda'),
            **kwargs,
        )
    else:
        raise NotImplementedError(f'ray tracing backend {backend} is not supported')

def sdf_rt_o3d(
    vertices:np.ndarray, 
    faces:np.ndarray, 
    points:np.ndarray, 
    n_rays=8,
    n_rays_threshold=8,
    max_iter=10,
    epsilon=1e-5,
    visible_check=True,
    chunk_size:Optional[int]=None,
) -> np.ndarray:
    # NOTE: NO CHECK
    chunk_size = chunk_size or points.shape[0]
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices), 
        triangles=o3d.utility.Vector3iVector(faces), 
    )
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        mesh, 
        vertex_dtype=o3d.core.float32, 
        triangle_dtype=o3d.core.int64, 
        device=o3d.core.Device('CPU:0'), 
    )
    bvh = o3d.t.geometry.RaycastingScene()
    bvh.add_triangles(mesh)
    bvh_invalid_id = bvh.INVALID_ID
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]], axis=-1)
    normals = normals / np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
    generator = np.random.default_rng(666)

    points_cnt = np.zeros((points.shape[0],), dtype=np.int64)
    points_index = np.arange(points.shape[0], dtype=np.int64)
    for idx in tqdm(range(0, points.shape[0], chunk_size), desc='sdf_rt_o3d'):
        _points = points[idx:idx+chunk_size]
        _points_index = points_index[idx:idx+chunk_size]

        rays_o = _points.repeat(n_rays, axis=0)
        rays_d = generator.normal(size=rays_o.shape).astype(rays_o.dtype)
        rays_d = rays_d / np.clip(np.linalg.norm(rays_d, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
        rays_o3d = o3d.core.Tensor(
            np.concatenate([rays_o, rays_d], axis=-1), 
            dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'),
        )
        cast_result = bvh.cast_rays(rays_o3d)
        tid = cast_result['primitive_ids'].numpy().astype(np.int64)
        if visible_check:
            rays_mask = (tid != bvh_invalid_id).reshape(_points.shape[0], n_rays).all(axis=-1)
            rays_idx = _points_index[rays_mask]
            if max_iter > 1:
                rays_mask = rays_mask.repeat(n_rays, axis=0)
                rays_idx = rays_idx.repeat(n_rays, axis=0)
            else:
                np.add.at(points_cnt, rays_idx, n_rays)
        else:
            rays_idx = _points_index.repeat(n_rays, axis=0)[rays_mask]
        for iter in range(max_iter - 1):
            if rays_idx.shape[0] == 0:
                break
            rays_t = cast_result['t_hit'].numpy().astype(np.float32)
            rays_n = normals[tid[rays_mask], :]
            rays_d = rays_d[rays_mask, :]
            rays_h = rays_o[rays_mask, :] + rays_t[rays_mask][:, None] * rays_d
            rays_cos = np.sum(rays_d * rays_n, axis=-1)
            rays_d = rays_d - 2.0 * rays_cos[:, None] * rays_n
            rays_o = rays_h - epsilon * np.sign(rays_cos)[:, None] * rays_n

            rays_o3d = o3d.core.Tensor(
                np.concatenate([rays_o, rays_d], axis=-1), 
                dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'),
            )
            cast_result = bvh.cast_rays(rays_o3d)
            tid = cast_result['primitive_ids'].numpy().astype(np.int64)
            rays_mask = (tid != bvh_invalid_id)
            rays_idx = rays_idx[rays_mask]
        if max_iter > 1:
            np.add.at(points_cnt, rays_idx, 1)

    sdf = bvh.compute_distance(o3d.core.Tensor(points, dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))).numpy()
    np.multiply.at(sdf, np.where(points_cnt >= n_rays_threshold)[0], -1.0)
    return sdf

def sdf_rt_cubvh(
    vertices:torch.Tensor, 
    faces:torch.Tensor, 
    points:torch.Tensor, 
    n_rays=8,
    n_rays_threshold=8,
    max_iter=10,
    epsilon=1e-5,
    chunk_size:Optional[int]=None,
    visible_check=True,
) -> torch.Tensor:
    # NOTE: NO CHECK
    chunk_size = chunk_size or points.shape[0]
    bvh = cubvh.cuBVH(vertices=vertices, triangles=faces)
    bvh_invalid_id = -1
    normals = torch.linalg.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]], dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    generator = torch.Generator(device='cuda').manual_seed(666)

    points_cnt = torch.zeros((points.shape[0],), dtype=torch.int64, device='cuda')
    points_index = torch.arange(points.shape[0], dtype=torch.int64, device='cuda')
    for idx in tqdm(range(0, points.shape[0], chunk_size), desc='sdf_rt_cubvh'):
        _points = points[idx:idx+chunk_size]
        _points_index = points_index[idx:idx+chunk_size]

        rays_o = _points.repeat_interleave(n_rays, dim=0)
        rays_d = torch.randn(rays_o.shape, dtype=torch.float32, device='cuda', generator=generator)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        rays_h, tid, _ = bvh.ray_trace(rays_o, rays_d)
        if visible_check:
            rays_mask = (tid != bvh_invalid_id).reshape(_points.shape[0], n_rays).all(dim=-1)
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
            rays_h, tid, _ = bvh.ray_trace(rays_o, rays_d)
            rays_mask = (tid != bvh_invalid_id)
            rays_idx = rays_idx[rays_mask]
        if max_iter > 1:
            points_cnt.scatter_(0, rays_idx, 1, reduce='add')

    sdf, _, _ = bvh.unsigned_distance(points, return_uvw=False)
    sdf.scatter_(0, torch.where(points_cnt >= n_rays_threshold)[0], -1.0, reduce='multiply')
    return sdf

def sdf_rt_trimesh(
    vertices:np.ndarray, 
    faces:np.ndarray, 
    points:np.ndarray, 
    n_rays=8,
    n_rays_threshold=8,
    max_iter=10,
    epsilon=1e-5,
    visible_check=True,
    chunk_size:Optional[int]=None,
) -> np.ndarray:
    # NOTE: NO CHECK
    chunk_size = chunk_size or points.shape[0]
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]], axis=-1)
    normals = normals / np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        process=False,
    )
    bvh = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    bvh_invalid_id = -1
    generator = np.random.default_rng(666)

    points_cnt = np.zeros((points.shape[0],), dtype=np.int64)
    for idx in tqdm(range(0, points.shape[0], chunk_size), desc='sdf_rt_trimesh'):
        _points = points[idx:idx+chunk_size]

        rays_o = _points.repeat(n_rays, axis=0)
        rays_d = generator.normal(size=rays_o.shape).astype(rays_o.dtype)
        rays_d = rays_d / np.clip(np.linalg.norm(rays_d, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
        tid, rays_idx = bvh.intersects_id(
            ray_origins=rays_o.reshape(-1, 3), 
            ray_directions=rays_d.reshape(-1, 3),
            max_hits=max_iter,
        )
        np.add.at(points_cnt, rays_idx // n_rays + idx, 1)

    sdf, _, _ = igl.point_mesh_squared_distance(points, vertices, faces)
    np.multiply.at(sdf, np.where(points_cnt >= n_rays_threshold)[0], -1.0)
    return sdf

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
    # NOTE: NO CHECK
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

    bvh = cubvh.cuBVH(vertices=vertices, triangles=faces)
    sdf, _, _ = bvh.unsigned_distance(points, return_uvw=False)
    sdf.scatter_(0, torch.where(points_cnt >= n_rays_threshold)[0], -1.0, reduce='multiply')
    return sdf


@timeout(60)
def mesh_to_sdf_octree(
    input_mesh_path, 
    output_mesh_path, 
    fix_nonmanifold=False, 
    scale:Optional[float]=0.95,
    N=10_000_000, 
    depth=10, 
    depth_coarse:Optional[int]=8, 
    use_octree=False, 
    use_bool_grid=True, 
    low_mem=True, 
    expand=2, 
    expand_coarse:Optional[int]=1, 
    use_wn=True, 
    use_rt=False, 
    use_rtv=False, 
    use_bvh=False, 
    use_udf=False,
    use_cuda_stage_1=True, 
    use_cuda_stage_2=True, 
    sdf_offset=0.5, 
):
    '''
    input_mesh_path: normalized mesh path, range(-1.0, 1.0)
    output_mesh_path: reconstructed mesh path
    fix_nonmanifold: remove nonmanifold faces on input mesh
    scale: apply scale on input mesh
    N: num of samples on mesh
    depth: depth of octree
    use_octree: whether use octree to build leaf cubes
    use_bool_grid: if not use octree, whether use bool grid to build leaf cubes
    low_mem: low cuda memory for high resolition/depth
    expand: expand of octree
    use_wn: whether use winding number to compute sdf, prior: winding numver > ray tracing > cubvh > udf
    use_rt: whether use ray tracing to compute sdf, prior: winding numver > ray tracing > cubvh > udf
    use_rtv: if use ray tracing to compute sdf, whether use visible checking only with ray tracing to compute sdf
    use_bvh: whether use cubvh to compute sdf, prior: winding numver > ray tracing > cubvh > udf
    use_cuda: whether use cuda to unique/sort
    sdf_offset: whether dilate sdf_offset * size_of_cube for sdf
    '''
    assert expand >= 0
    if expand_coarse is not None:
        depth_delta = depth - depth_coarse
        assert depth_delta > 0
    else:
        depth_delta = None
    time_log = dict()

    ### load whole mesh
    t0 = perf_counter()
    if not fix_nonmanifold:
        mesh_trimesh:trimesh.Trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')
        if scale is not None:
            mesh_trimesh = mesh_trimesh.apply_scale([scale] * 3)
    else:
        mesh_trimesh = None
    if not (use_wn or use_rt or use_bvh) or fix_nonmanifold:
        mesh_o3d = o3d.io.read_triangle_mesh(input_mesh_path, enable_post_processing=False, print_progress=False)
        if scale is not None:
            mesh_o3d = mesh_o3d.scale(scale=scale, center=[0.0] * 3)
    else:
        mesh_o3d = None
    if fix_nonmanifold:
        mesh_o3d = mesh_o3d.remove_non_manifold_edges()
        mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices, dtype=np.float32),
            faces=np.asarray(mesh_o3d.triangles, dtype=np.int64),
            process=False,
        )
    # NOTE: debug with sphere
    # mesh_trimesh = trimesh.creation.uv_sphere(0.85)
    time_log['load whole mesh'] = perf_counter() - t0

    ### sample surface points
    t0 = perf_counter()
    surface_points, surface_points_face_index = trimesh.sample.sample_surface(mesh=mesh_trimesh, count=N, face_weight=None, seed=666)
    time_log['sample surface points'] = perf_counter() - t0

    ### build cubes
    t0 = perf_counter()
    ## build leaf cubes
    if use_octree:
        ## build octree and leaf cubes
        dummy_points = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        surface_points = np.concatenate([surface_points, dummy_points])
        points_o3d = o3d.utility.Vector3dVector(surface_points)
        point_cloud_o3d = o3d.geometry.PointCloud(points=points_o3d)
        octree_o3d = o3d.geometry.Octree(max_depth=depth)
        octree_o3d.convert_from_point_cloud(point_cloud_o3d, size_expand=0.0)
        origins = []
        sizes = []
        octree_o3d.traverse(partial(f_traverse, origins=origins, sizes=sizes))
        origins = np.array(origins, dtype=np.float32)
        # NOTE: open3d octree voxels are cubes
        sizes = np.array(sizes, dtype=np.float32)
        sizes = np.tile(sizes[:, None], (1, 3))
        if not use_cuda_stage_1:
            dx, dy, dz = np.array_split(sizes, 3, axis=-1)
            zeros = np.zeros_like(dx)
            sizes_octree = np.stack([
                np.concatenate([zeros, zeros, zeros], axis=-1),
                np.concatenate([dx, zeros, zeros], axis=-1),
                np.concatenate([zeros, dy, zeros], axis=-1),
                np.concatenate([dx, dy, zeros], axis=-1),
                np.concatenate([zeros, zeros, dz], axis=-1),
                np.concatenate([dx, zeros, dz], axis=-1),
                np.concatenate([zeros, dy, dz], axis=-1),
                np.concatenate([dx, dy, dz], axis=-1),
            ], axis=1)
            octree_o3d_max = np.asarray(octree_o3d.get_max_bound(), dtype=np.float32)
            octree_o3d_min = np.asarray(octree_o3d.get_min_bound(), dtype=np.float32)
            indices = np.round((origins - octree_o3d_min) / (octree_o3d_max - octree_o3d_min) * (2 ** depth)).astype(np.int64)
            b = (2 ** depth)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
        else:
            origins = torch.as_tensor(origins, dtype=torch.float32, device='cuda')
            sizes = torch.as_tensor(sizes, dtype=torch.float32, device='cuda')
            dx, dy, dz = torch.chunk(sizes, 3, dim=-1)
            zeros = torch.zeros_like(dx)
            sizes_octree = torch.stack([
                torch.cat([zeros, zeros, zeros], dim=-1),
                torch.cat([dx, zeros, zeros], dim=-1),
                torch.cat([zeros, dy, zeros], dim=-1),
                torch.cat([dx, dy, zeros], dim=-1),
                torch.cat([zeros, zeros, dz], dim=-1),
                torch.cat([dx, zeros, dz], dim=-1),
                torch.cat([zeros, dy, dz], dim=-1),
                torch.cat([dx, dy, dz], dim=-1),
            ], dim=1)
            octree_o3d_max = torch.as_tensor(octree_o3d.get_max_bound(), dtype=torch.float32)
            octree_o3d_min = torch.as_tensor(octree_o3d.get_min_bound(), dtype=torch.float32)
            indices = torch.round((origins - octree_o3d_min) / (octree_o3d_max - octree_o3d_min) * (2 ** depth)).to(dtype=torch.int64)
            b = (2 ** depth)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
        use_bool_grid = False
        bool_grid = None
    else:
        ## build leaf cubes from surface points
        if not use_cuda_stage_1:
            if expand_coarse is None:
                indices = np.floor((np.clip(surface_points, -1.0, 1.0) * 0.5 + 0.5) * (2 ** depth)).astype(dtype=np.int64)
                b = (2 ** depth)
                expand_fine = None
            else:
                indices = np.floor((np.clip(surface_points, -1.0, 1.0) * 0.5 + 0.5) * (2 ** (depth - depth_delta))).astype(dtype=np.int64)
                b = (2 ** (depth - depth_delta))
                expand_fine = expand
                expand = expand_coarse
            indices = np.unique(indices, axis=0)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
            if not use_bool_grid:
                bool_grid = None
                origins = (indices.astype(np.float32) / (2 ** depth)) * 2.0 - 1.0
                sizes = np.full_like(origins, fill_value=2.0 / (2 ** depth))
                dx, dy, dz = np.array_split(sizes, 3, axis=-1)
                zeros = np.zeros_like(dx)
                sizes_octree = np.stack([
                    np.concatenate([zeros, zeros, zeros], axis=-1),
                    np.concatenate([dx, zeros, zeros], axis=-1),
                    np.concatenate([zeros, dy, zeros], axis=-1),
                    np.concatenate([dx, dy, zeros], axis=-1),
                    np.concatenate([zeros, zeros, dz], axis=-1),
                    np.concatenate([dx, zeros, dz], axis=-1),
                    np.concatenate([zeros, dy, dz], axis=-1),
                    np.concatenate([dx, dy, dz], axis=-1),
                ], axis=1)
            else:
                bool_grid = np.zeros((2 ** depth, 2 ** depth, 2 ** depth), dtype=bool)
                origins = None
                sizes = None
                sizes_octree = None
        else:
            surface_points = torch.as_tensor(surface_points, dtype=torch.float32, device='cuda')
            if expand_coarse is None:
                indices = torch.floor((torch.clamp(surface_points, -1.0, 1.0) * 0.5 + 0.5) * (2 ** depth)).to(dtype=torch.int64)
                b = (2 ** depth)
                expand_fine = None
            else:
                indices = torch.floor((torch.clamp(surface_points, -1.0, 1.0) * 0.5 + 0.5) * (2 ** (depth - depth_delta))).to(dtype=torch.int64)
                b = (2 ** (depth - depth_delta))
                expand_fine = expand
                expand = expand_coarse
            indices = torch.unique(indices, dim=0)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
            if not use_bool_grid:
                bool_grid = None
                origins = (indices.to(dtype=torch.float32) / (2 ** depth)) * 2.0 - 1.0
                sizes = torch.full_like(origins, fill_value=2.0 / (2 ** depth))
                dx, dy, dz = torch.chunk(sizes, 3, dim=-1)
                zeros = torch.zeros_like(dx)
                sizes_octree = torch.stack([
                    torch.cat([zeros, zeros, zeros], dim=-1),
                    torch.cat([dx, zeros, zeros], dim=-1),
                    torch.cat([zeros, dy, zeros], dim=-1),
                    torch.cat([dx, dy, zeros], dim=-1),
                    torch.cat([zeros, zeros, dz], dim=-1),
                    torch.cat([dx, zeros, dz], dim=-1),
                    torch.cat([zeros, dy, dz], dim=-1),
                    torch.cat([dx, dy, dz], dim=-1),
                ], dim=1)
            else:
                bool_grid = torch.zeros((2 ** depth, 2 ** depth, 2 ** depth), dtype=torch.bool, device='cuda')
                origins = None
                sizes = None
                sizes_octree = None
    del surface_points, indices

    ## expand leaf cubes
    expand_basis = list(product(*([range(-expand, expand+1)] * 3)))
    if not use_cuda_stage_1:
        if not use_bool_grid:
            sizes_octree = np.tile(sizes_octree[:, None, :, :], (1, len(expand_basis), 1, 1))
            indices_spatial = np.tile(indices_spatial[:, None], (1, len(expand_basis)))
            for i, basis in enumerate(expand_basis):
                for j, b in enumerate(basis):
                    if b != 0:
                        sizes_octree[:, i, :, j] = sizes_octree[:, i, :, j] + b * sizes[:, j, None]
                indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
            points_octree = origins[:, None, None, :] + sizes_octree
        else:
            indices_spatial = np.tile(indices_spatial[:, None], (1, len(expand_basis)))
            for i, basis in enumerate(expand_basis):
                indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
            np.logical_or.at(bool_grid.reshape(-1), indices_spatial.reshape(-1), True)
    else:
        if not use_bool_grid:
            sizes_octree = torch.tile(sizes_octree[:, None, :, :], (1, len(expand_basis), 1, 1))
            indices_spatial = torch.tile(indices_spatial[:, None], (1, len(expand_basis)))
            for i, basis in enumerate(expand_basis):
                for j, base in enumerate(basis):
                    if base != 0:
                        sizes_octree[:, i, :, j] = sizes_octree[:, i, :, j] + base * sizes[:, j, None]
                indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
            points_octree = origins[:, None, None, :] + sizes_octree
        else:
            if not low_mem:
                indices_spatial = torch.tile(indices_spatial[:, None], (1, len(expand_basis)))
                for i, basis in enumerate(expand_basis):
                    indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
                bool_grid.reshape(-1).scatter_(0, indices_spatial.reshape(-1), True)
            else:
                if expand_coarse is None:
                    for i, basis in enumerate(expand_basis):
                        bool_grid.reshape(-1).scatter_(0, torch.clamp(indices_spatial + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
                else:
                    bool_grid = bool_grid.reshape(b, 2 ** depth_delta, b, 2 ** depth_delta, b, 2 ** depth_delta).permute(0, 2, 4, 1, 3, 5).reshape(-1, (2 ** depth_delta) ** 3)
                    for i, basis in enumerate(expand_basis):
                        bool_grid.scatter_(0, torch.clamp(indices_spatial.unsqueeze(-1) + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
                    bool_grid = bool_grid.reshape(b, b, b, 2 ** depth_delta, 2 ** depth_delta, 2 ** depth_delta).permute(0, 3, 1, 4, 2, 5).reshape((2 ** depth_delta) * b, (2 ** depth_delta) * b, (2 ** depth_delta) * b)
                    indices = torch.stack(torch.where(bool_grid), dim=-1)
                    b = (2 ** depth)
                    bx, by, bz = b ** 2, b, 1
                    expand = expand_fine
                    indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
                    expand_basis = list(product(*([range(-expand, expand+1)] * 3)))
                    for i, basis in enumerate(expand_basis):
                        bool_grid.reshape(-1).scatter_(0, torch.clamp(indices_spatial + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
            del indices_spatial
    del origins, sizes, sizes_octree

    ## remove reduplicated cubes and vertices
    if use_cuda_stage_2:
        if not use_bool_grid:
            if not use_cuda_stage_1:
                indices_spatial = torch.as_tensor(indices_spatial, dtype=torch.int64, device='cuda')
                points_octree = torch.as_tensor(points_octree, dtype=torch.float32, device='cuda')
            # NOTE: https://github.com/pytorch/pytorch/issues/36748
            indices_spatial_unique, indices_spatial_inverse, indices_spatial_counts = torch.unique(indices_spatial.reshape(-1), return_inverse=True, return_counts=True)
            indices_spatial_index = indices_spatial_inverse.argsort(stable=True)[torch.cat([indices_spatial_counts.new_zeros(1), indices_spatial_counts.cumsum(dim=0)])[:-1]]
            indices_spatial_unique_sorted = torch.argsort(indices_spatial_unique, stable=True)
            points_octree = points_octree.reshape(-1, 8, 3)[indices_spatial_index, :, :][indices_spatial_unique_sorted, :, :]
            points_octree_unique, points_octree_unique_inverse = torch.unique(points_octree.reshape(-1, 3), dim=0, return_inverse=True)
            points_octree_unique_inverse = points_octree_unique_inverse.reshape(-1, 8)
            del points_octree, indices_spatial_unique_sorted, indices_spatial_index, indices_spatial_unique
        else:
            if not use_cuda_stage_1:
                bool_grid = torch.as_tensor(bool_grid, dtype=torch.bool, device='cuda')
            indices = torch.stack(torch.where(bool_grid), dim=-1)
            b = (2 ** depth + 1)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
            expand_basis = list(product(*([[0,1]] * 3)))
            indices_spatial = torch.tile(indices_spatial[:, None], (1, len(expand_basis)))
            for i, basis in enumerate(expand_basis):
                indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
            indices_spatial_unique, indices_spatial_inverse = torch.unique(indices_spatial.reshape(-1), dim=0, return_inverse=True)
            indices_unique = torch.stack([
                indices_spatial_unique // bx % b,
                indices_spatial_unique // by % b,
                indices_spatial_unique // bz % b,
            ], dim=-1)
            points_octree_unique = (indices_unique.to(dtype=torch.float32) / (2 ** depth)) * 2.0 - 1.0
            points_octree_unique_inverse = indices_spatial_inverse.reshape(-1, 8)
            del bool_grid, indices, indices_spatial, indices_spatial_unique, indices_spatial_inverse, indices_unique
    else:
        if not use_bool_grid:
            if use_cuda_stage_1:
                indices_spatial = indices_spatial.detach().cpu().numpy()
                points_octree = points_octree.detach().cpu().numpy()
            indices_spatial_unique, indices_spatial_index = np.unique(indices_spatial.reshape(-1), return_index=True)
            indices_spatial_unique_sorted = np.argsort(indices_spatial_unique)
            points_octree = points_octree.reshape(-1, 8, 3)[indices_spatial_index, :, :][indices_spatial_unique_sorted, :, :]
            points_octree_unique, points_octree_unique_inverse = np.unique(points_octree.reshape(-1, 3), axis=0, return_inverse=True)
            points_octree_unique_inverse = points_octree_unique_inverse.reshape(-1, 8)
            del points_octree, indices_spatial_unique_sorted, indices_spatial_index, indices_spatial_unique
        else:
            if not use_cuda_stage_1:
                bool_grid = bool_grid.detach().cpu().numpy()
            indices = np.stack(np.where(bool_grid), axis=-1)
            b = (2 ** depth + 1)
            bx, by, bz = b ** 2, b, 1
            indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
            expand_basis = list(product(*([[0,1]] * 3)))
            indices_spatial = np.tile(indices_spatial[:, None], (1, len(expand_basis)))
            for i, basis in enumerate(expand_basis):
                indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
            indices_spatial_unique, indices_spatial_inverse = np.unique(indices_spatial.reshape(-1), axis=0, return_inverse=True)
            indices_unique = np.stack([
                indices_spatial_unique // bx % b,
                indices_spatial_unique // by % b,
                indices_spatial_unique // bz % b,
            ], axis=-1)
            points_octree_unique = (indices_unique.astype(np.float32) / (2 ** depth)) * 2.0 - 1.0
            points_octree_unique_inverse = indices_spatial_inverse.reshape(-1, 8)
            del bool_grid, indices, indices_spatial, indices_spatial_unique, indices_spatial_inverse, indices_unique
    time_log['build cubes'] = perf_counter() - t0

    ### compute sdf
    t0 = perf_counter()
    if use_wn:
        ## compute udf with winding number
        # NOTE: https://github.com/libigl/libigl/blob/main/include/igl/signed_distance.h#L31
        if use_cuda_stage_2:
            points_octree_unique_sdf, _, _ = igl.signed_distance(points_octree_unique.detach().cpu().numpy(), mesh_trimesh.vertices, mesh_trimesh.faces, return_normals=False, sign_type=4)
            points_octree_unique_sdf = torch.as_tensor(points_octree_unique_sdf, dtype=torch.float32, device='cuda')
        else:
            points_octree_unique_sdf, _, _ = igl.signed_distance(points_octree_unique, mesh_trimesh.vertices, mesh_trimesh.faces, return_normals=False, sign_type=4)
            points_octree_unique = torch.as_tensor(points_octree_unique, dtype=torch.float32, device='cuda')
            points_octree_unique_inverse = torch.as_tensor(points_octree_unique_inverse, dtype=torch.int64, device='cuda')
            points_octree_unique_sdf = torch.as_tensor(points_octree_unique_sdf, dtype=torch.float32, device='cuda')
    elif use_bvh:
        ## build bvh, compute sdf via ray tracing
        # NOTE: https://github.com/ashawkey/cubvh.git
        bvh = cubvh.cuBVH(
            vertices=torch.as_tensor(mesh_trimesh.vertices, dtype=torch.float32, device='cuda'), 
            triangles=torch.as_tensor(mesh_trimesh.faces, dtype=torch.int64, device='cuda'), 
        )
        if not use_cuda_stage_2:
            points_octree_unique = torch.as_tensor(points_octree_unique, dtype=torch.float32, device='cuda')
        points_octree_unique_sdf, _, _ = bvh.signed_distance(points_octree_unique, return_uvw=False, mode='raystab')
        if not use_cuda_stage_2:
            points_octree_unique_inverse = torch.as_tensor(points_octree_unique_inverse, dtype=torch.int64, device='cuda')
        del bvh
    elif use_rt:
        if not use_cuda_stage_2:
            points_octree_unique = torch.as_tensor(points_octree_unique, dtype=torch.float32, device='cuda')
        if use_rtv:
            points_octree_unique_sdf = sdf_rt(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces, points=points_octree_unique, max_iter=1)
        else:
            points_octree_unique_sdf = sdf_rt(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces, points=points_octree_unique)
        if not use_cuda_stage_2:
            points_octree_unique_inverse = torch.as_tensor(points_octree_unique_inverse, dtype=torch.int64, device='cuda')
    elif use_udf:
        ## build bvh, compute sdf via ray tracing
        # NOTE: https://github.com/ashawkey/cubvh.git
        bvh = cubvh.cuBVH(
            vertices=torch.as_tensor(mesh_trimesh.vertices, dtype=torch.float32, device='cuda'), 
            triangles=torch.as_tensor(mesh_trimesh.faces, dtype=torch.int64, device='cuda'), 
        )
        if not use_cuda_stage_2:
            points_octree_unique = torch.as_tensor(points_octree_unique, dtype=torch.float32, device='cuda')
        points_octree_unique_sdf, _, _ = bvh.unsigned_distance(points_octree_unique, return_uvw=False)
        if not use_cuda_stage_2:
            points_octree_unique_inverse = torch.as_tensor(points_octree_unique_inverse, dtype=torch.int64, device='cuda')
        del bvh
    else:
        ## build kdtree, compute sdf for watertight mesh
        mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d, vertex_dtype=o3d.core.float32, triangle_dtype=o3d.core.int64, device=o3d.core.Device('CPU:0'))
        kdtree_o3d = o3d.t.geometry.RaycastingScene()
        kdtree_o3d.add_triangles(mesh_o3d_t)
        if use_cuda_stage_2:
            points_octree_unique_sdf = kdtree_o3d.compute_signed_distance(query_points=o3d.core.Tensor(points_octree_unique.detach().cpu().numpy(), dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))).numpy()
            points_octree_unique_sdf = torch.as_tensor(points_octree_unique_sdf, dtype=torch.float32, device='cuda')
        else:
            points_octree_unique_sdf = kdtree_o3d.compute_signed_distance(query_points=o3d.core.Tensor(points_octree_unique, dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))).numpy()
            points_octree_unique = torch.as_tensor(points_octree_unique, dtype=torch.float32, device='cuda')
            points_octree_unique_inverse = torch.as_tensor(points_octree_unique_inverse, dtype=torch.int64, device='cuda')
            points_octree_unique_sdf = torch.as_tensor(points_octree_unique_sdf, dtype=torch.float32, device='cuda')
    time_log['compute sdf'] = perf_counter() - t0

    ### run flexicubes plus
    t0 = perf_counter()
    fc = FlexiCubesPlus()
    x_nx3 = points_octree_unique
    cube_fx8 = points_octree_unique_inverse
    sdf = points_octree_unique_sdf
    if sdf_offset != 0.0:
        sdf = sdf - sdf_offset * (2 / (2 ** depth))
    t = perf_counter()
    vertices, faces, _ = fc(x_nx3, sdf, cube_fx8, 2 ** depth)
    print('flexicubes octree waste', perf_counter() - t, 'sec')  # ~ 0.0164 sec for depth = 8
    time_log['run flexicubes plus'] = perf_counter() - t0

    ## export reconstructed mesh
    t0 = perf_counter()
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    mesh_reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if use_udf:
        mesh_reconstructed = mesh_reconstructed.split(only_watertight=False)
        mesh_reconstructed = mesh_reconstructed[np.argmax([m.faces.shape[0] for m in mesh_reconstructed])]
    if scale is not None:
        mesh_reconstructed = mesh_reconstructed.apply_scale([1.0 / scale] * 3)
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    mesh_reconstructed.export(output_mesh_path)
    time_log['export reconstructed mesh'] = perf_counter() - t0

    print('time log:\n\t' + '\n\t'.join(f'{k:36s}:{v:.6f}' for k, v in time_log.items()))
    return time_log


def small_test():
    test_examples = [
        'cute_wolf/textured_mesh.glb',
        'car/867ceb1e7dc245539e0301ef0ece74f4.glb',
        'car2/88a596cf876c4175a59a3323510d49f0.glb',
        'car3/368873bdc2634959b020d58a88797158.glb',
        'watch/9057e49289d742eb9663edea9aadf3e8.glb',
    ]
    for test_example in test_examples:
        input_mesh_path = os.path.join('gradio_examples_mesh', test_example)
        output_mesh_path_dense = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_dense.glb')
        output_mesh_path_octree = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_octree.glb')

        t = perf_counter()
        mesh_to_sdf_dense(input_mesh_path, output_mesh_path_dense)
        print('>> mesh_to_sdf_dense', perf_counter() - t)

        t = perf_counter()
        mesh_to_sdf_octree(input_mesh_path, output_mesh_path_octree)
        print('>> mesh_to_sdf_octree', perf_counter() - t)


def large_test():
    src = '/mnt/nas-algo/chenxiao/dataset/vae_example_data/objaverse_data/*/*.glb'
    dst = '/mnt/nas-algo/chenxiao/dataset/reconstruct_results'
    config_list = {
        'wn_sdf': {
            'use_wn': True, 
            'use_rt': False, 
            'use_rtv': False, 
            'use_bvh': False, 
        },
        'rtv_sdf': {
            'use_wn': False, 
            'use_rt': False, 
            'use_rtv': True, 
            'use_bvh': False, 
        },
        'bvh_sdf': {
            'use_wn': False, 
            'use_rt': False, 
            'use_rtv': False, 
            'use_bvh': True, 
        },
    }
    for key, config in config_list.items():
        for input_path in tqdm(glob(src), desc=key):
            uid = os.path.join(os.path.basename(os.path.dirname(input_path)), os.path.splitext(os.path.basename(input_path))[0])
            output_path = os.path.join(dst, key, uid + '.glb')
            log_path = os.path.join(dst, key, uid + '.json')
            if os.path.isfile(output_path):
                continue
            try:
                time_log = mesh_to_sdf_octree(input_path, output_path, **config)
            except Exception as e:
                time_log = 'failed'
                print(e)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({uid: time_log}, f, indent=4)


if __name__ == '__main__':
    small_test()
    # large_test()

