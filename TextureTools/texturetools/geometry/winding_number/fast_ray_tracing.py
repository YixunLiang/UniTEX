from math import prod
import os
from time import perf_counter
from typing import Optional
import numpy as np
import open3d as o3d
import torch
import trimesh
import igl  # pip install libigl
# https://github.com/ashawkey/cubvh.git
# https://github.com/opencv/opencv/issues/14868
import cubvh


def dot(x: np.ndarray, y: np.ndarray, axis: int = -1, keepdims: bool = False):
    return np.sum(x * y, axis=axis, keepdims=keepdims)

def dot_torch(x: torch.Tensor, y: torch.Tensor, dim: int = -1, keepdim: bool = False):
    return torch.sum(x * y, dim=dim, keepdim=keepdim)

def normalize(x: np.ndarray, p: float = 2.0, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.clip(np.linalg.norm(x, ord=p, axis=axis, keepdims=True), a_min=eps, a_max=None)
    return np.divide(x, denom)

def reflect(x: np.ndarray, n: np.ndarray, axis: int = -1, n_normalized: bool = False) -> np.ndarray:
    if not n_normalized:
        n = normalize(n, axis=axis)
    return x - 2.0 * dot(x, n, axis=axis, keepdims=True) * n

def reflect_torch(x: torch.Tensor, n: torch.Tensor, dim: int = -1, n_normalized: bool = False) -> torch.Tensor:
    if not n_normalized:
        n = torch.nn.functional.normalize(n, dim=dim)
    return x - 2.0 * dot_torch(x, n, dim=dim, keepdim=True) * n


def ray_tracing_o3d(
    mesh: o3d.geometry.TriangleMesh, 
    rays_o: np.ndarray, 
    rays_d: Optional[np.ndarray] = None,
    max_iter: int = 1000, 
    epsilon: float = 1e-5,
    compute_mode: int = 0,
) -> np.ndarray:
    '''
    mesh: open3d mesh
    rays_o: [..., 3], origin of rays
    rays_d: [..., 3], directions of rays
    max_iter: maximum num of iterations for ray tracing
    epsilon: small offset for origin of reflected rays
    compute_mode: 0 or 1

    rays_mask: [...,], mask of inner rays
    '''
    if rays_d is None:
        generator = np.random.default_rng(seed=666)
        rays_d = generator.normal(size=rays_d.shape)
    rays_o, rays_d = np.broadcast_arrays(rays_o.astype(np.float32), rays_d.astype(np.float32))
    rays_mask = np.zeros(rays_o.shape[:-1], dtype=bool)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rays_idx = np.arange(rays_o.shape[0], dtype=np.int64)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        mesh, vertex_dtype=o3d.core.float32, triangle_dtype=o3d.core.int64, 
        device=o3d.core.Device('CPU:0'),
    )
    v = mesh.vertex['positions'].numpy()
    t = mesh.triangle['indices'].numpy()
    kdtree = o3d.t.geometry.RaycastingScene()
    kdtree.add_triangles(mesh)

    if compute_mode == 0:
        for iter in range(max_iter):
            rays_o3d = o3d.core.Tensor(
                np.concatenate([rays_o, rays_d], axis=-1), 
                dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'),
            )
            cast_result = kdtree.cast_rays(rays_o3d)
            tid = cast_result['primitive_ids'].numpy()
            mask = (tid != kdtree.INVALID_ID)
            tid = tid[mask]
            rays_idx = rays_idx[mask]
            rays_d = rays_d[mask]
            uv = cast_result['primitive_uvs'].numpy()[mask]
            rays_n = cast_result['primitive_normals'].numpy()[mask]
            rays_r = reflect(rays_d, rays_n, n_normalized=True)

            # NOTE: make sure origin is on the refleted side of face
            rays_o = v[t[tid, 1], :] * uv[:, [0]] + \
                v[t[tid, 2], :] * uv[:, [1]] + \
                v[t[tid, 0], :] * (1 - uv.sum(axis=1, keepdims=True)) + \
                epsilon * np.sign(dot(rays_r, rays_n, keepdims=True)) * rays_n
            rays_d = rays_r
    elif compute_mode == 1:
        for iter in range(max_iter):
            rays_o3d = o3d.core.Tensor(
                np.concatenate([rays_o, rays_d], axis=-1), 
                dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'),
            )
            cast_result = kdtree.cast_rays(rays_o3d)
            tid = cast_result['primitive_ids'].numpy()
            mask = (tid != kdtree.INVALID_ID)
            tid = tid[mask]
            rays_idx = rays_idx[mask]
            rays_o = rays_o[mask]
            rays_d = rays_d[mask]
            rays_t = cast_result['t_hit'].numpy()[mask][:, None]
            rays_n = cast_result['primitive_normals'].numpy()[mask]
            rays_r = reflect(rays_d, rays_n, n_normalized=True)

            rays_o = rays_o + rays_t * rays_d + epsilon * np.sign(dot(rays_r, rays_n, keepdims=True)) * rays_n
            rays_d = rays_r
    else:
        raise NotImplementedError(f'compute_mode {compute_mode} is not supported')

    np.logical_or.at(rays_mask.reshape(-1), rays_idx, True)
    return rays_mask


def ray_tracing_bvh(
    mesh: trimesh.Trimesh,
    rays_o: np.ndarray, 
    rays_d: Optional[np.ndarray] = None,
    max_iter: int = 2000, 
    epsilon: float = 1e-5,
) -> np.ndarray:
    '''
    mesh: trimesh mesh
    rays_o: [..., 3], origin of rays
    rays_d: [..., 3], directions of rays
    max_iter: maximum num of iterations for ray tracing
    epsilon: small offset for origin of reflected rays

    rays_mask: [...,], mask of inner rays
    '''
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device='cuda')
    faces_normal = torch.as_tensor(mesh.face_normals.copy(), dtype=torch.float32, device='cuda')
    bvh = cubvh.cuBVH(vertices, faces)

    rays_o = torch.as_tensor(rays_o, dtype=torch.float32, device='cuda')
    if rays_d is None:
        generator = torch.Generator(device='cuda').manual_seed(666)
        rays_d = torch.randn_like(rays_o, dtype=torch.float32, device='cuda', generator=generator)
    else:
        rays_d = torch.as_tensor(rays_d, dtype=torch.float32, device='cuda')
    rays_o, rays_d = torch.broadcast_tensors(rays_o, rays_d)
    rays_mask = torch.zeros(rays_o.shape[:-1], dtype=torch.bool, device='cuda')
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rays_idx = torch.arange(rays_o.shape[0], dtype=torch.int64, device='cuda')

    for iter in range(max_iter):
        rays_h, tid, rays_t = bvh.ray_trace(rays_o, rays_d)
        mask = (tid != -1)
        tid = tid[mask]
        rays_idx = rays_idx[mask]
        rays_n = faces_normal[tid, :]
        rays_d = rays_d[mask]
        rays_h = rays_h[mask]
        rays_r = reflect_torch(rays_d, rays_n, n_normalized=True)

        # NOTE: make sure origin is on the refleted side of face
        rays_o = rays_h + epsilon * torch.sign(dot_torch(rays_r, rays_n, keepdim=True)) * rays_n
        rays_d = rays_r

    rays_mask.scatter_(0, rays_idx, True)
    rays_mask = rays_mask.detach().cpu().numpy()
    return rays_mask


def test_ray_tracing(
    input_mesh_path, output_rays_path, N=10_000_000,
    test_o3d=False, test_bvh=True,
):
    '''
    input_mesh_path: all formats
    output_rays_path: ply format
    N: num of rays
    '''
    # load mesh
    if test_o3d:
        mesh_o3d = o3d.io.read_triangle_mesh(input_mesh_path, enable_post_processing=False, print_progress=False)
    if test_bvh:
        mesh_trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')

    # initialize rays
    generator = np.random.default_rng(seed=666)
    v = generator.normal(size=(N, 3))
    vn = generator.normal(size=(N, 3))

    # ray tracing
    if test_o3d:
        t = perf_counter()
        vm = ray_tracing_o3d(mesh_o3d, v, vn)
        print('ray tracing (o3d)', perf_counter() - t, 'sec')

    if test_bvh:
        t = perf_counter()
        vm = ray_tracing_bvh(mesh_trimesh, v, vn)
        print('ray tracing (bvh)', perf_counter() - t, 'sec')

    print('all points', prod(vm.shape), 'inner points', vm.sum())

    # export rays
    os.makedirs(os.path.dirname(output_rays_path), exist_ok=True)
    trimesh.Trimesh(vertices=v, vertex_normals=vn, vertex_colors=vm.astype(np.float32)[:, None], process=False).export(output_rays_path)
    trimesh.Trimesh(vertices=v[vm, :], vertex_normals=vn[vm, :], process=False).export(os.path.splitext(output_rays_path)[0] + '_inner.ply')
    trimesh.Trimesh(vertices=v[~vm, :], vertex_normals=vn[~vm, :], process=False).export(os.path.splitext(output_rays_path)[0] + '_outer.ply')


def winding_number_igl(vertices:np.ndarray, faces:np.ndarray, points:np.ndarray, fast=True):
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    points: [N, 3], float32

    wns: [N,], float32, range(0.0, 1.0)
    '''
    if fast:
        wns = igl.fast_winding_number_for_meshes(vertices, faces, points)
    else:
        wns = igl.winding_number(vertices, faces, points)
    return wns


def sdf_bvh(vertices:np.ndarray, faces:np.ndarray, points:np.ndarray)-> np.ndarray:
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    points: [N, 3], float32

    sdf: [N,], float32
    '''
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(faces, dtype=torch.int64, device='cuda')
    points = torch.as_tensor(points, dtype=torch.float32, device='cuda')
    bvh = cubvh.cuBVH(vertices, faces)
    sdf, _, _ = bvh.signed_distance(points, return_uvw=False, mode='raystab')
    sdf = sdf.detach().cpu().numpy()
    return sdf


def test_winding_number(
    input_mesh_path, output_points_path, N=10_000_000,
    test_igl=True, test_bvh=True,
):
    '''
    input_mesh_path: all formats
    output_points_path: ply format
    N: num of points
    '''
    # load mesh
    mesh = trimesh.load(input_mesh_path, process=False, force='mesh')

    # initialize points
    generator = np.random.default_rng(seed=666)
    v = generator.normal(size=(N, 3))

    # compute winding number or sdf
    if test_igl:
        t = perf_counter()
        wn_igl = winding_number_igl(mesh.vertices, mesh.faces, v)
        print('compute winding number (igl)', perf_counter() - t, 'sec')
    
    if test_bvh:
        t = perf_counter()
        sdf_bvh = winding_number_igl(mesh.vertices, mesh.faces, v)
        print('compute sdf (bvh)', perf_counter() - t, 'sec')

    # export points
    wn = wn_igl
    vm = (wn > 0.5)
    os.makedirs(os.path.dirname(output_points_path), exist_ok=True)
    trimesh.Trimesh(vertices=v, vertex_colors=np.clip(wn, 0.0, 1.0)[:, None], process=False).export(output_points_path)
    trimesh.Trimesh(vertices=v[vm, :], process=False).export(os.path.splitext(output_points_path)[0] + '_inner.ply')
    trimesh.Trimesh(vertices=v[~vm, :], process=False).export(os.path.splitext(output_points_path)[0] + '_outer.ply')


if __name__ == '__main__':
    test_examples = [
        'cute_wolf/textured_mesh.glb',
        'car/867ceb1e7dc245539e0301ef0ece74f4.glb',
        'car2/88a596cf876c4175a59a3323510d49f0.glb',
        'car3/368873bdc2634959b020d58a88797158.glb',
        'watch/9057e49289d742eb9663edea9aadf3e8.glb',
    ]
    for test_example in test_examples:
        input_mesh_path = os.path.join('gradio_examples_mesh', test_example)
        output_rays_path = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_rays.ply')
        output_points_path = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_points.ply')

        test_ray_tracing(input_mesh_path, output_rays_path)
        test_winding_number(input_mesh_path, output_points_path)

