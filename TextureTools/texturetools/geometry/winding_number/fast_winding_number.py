from math import prod
import os
from time import perf_counter
from typing import Optional
import numpy as np
import open3d as o3d
import torch
from torch.library import custom_op
from tqdm import tqdm
import trimesh
import igl  # pip install libigl
from meshlib import mrmeshpy, mrcudapy, mrmeshnumpy  # pip install meshlib


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


# Create a custom op that works on cpu
@custom_op("mylib::winding_number_torch_wrapper", mutates_args=(), device_types="cpu")
def winding_number_torch_wrapper(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    points_np = points.numpy()
    wns_np = igl.winding_number(vertices_np, faces_np, points_np)
    wns = torch.from_numpy(wns_np)
    return wns

# Add implementations for the cuda device
@torch.library.register_kernel("mylib::winding_number_torch_wrapper", "cuda")
def _(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    points_np = points.cpu().numpy()
    wns_np = igl.winding_number(vertices_np, faces_np, points_np)
    wns = torch.from_numpy(wns_np).to(device=vertices.device)
    return wns


def winding_number_torch(vertices:np.ndarray, faces:np.ndarray, points:np.ndarray):
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    points: [N, 3], float32

    points: [N,], float32, range(0.0, 1.0)
    '''
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(faces, dtype=torch.int64, device='cuda')
    points = torch.as_tensor(points, dtype=torch.float32, device='cuda')
    if vertices.shape[1] == 2:
        wns = winding_number_2d_torch(vertices, faces, points)
    elif vertices.shape[1] == 3:
        wns = winding_number_3d_torch(vertices, faces, points)
        # wns = winding_number_torch_wrapper(vertices, faces, points)
    else:
        raise NotImplementedError(f'vertices dimension should be 2 or 3, but {vertices.shape[1]}')
    wns = wns.detach().cpu().numpy().astype(np.float32)
    return wns

def winding_number_2d_torch(vertices:torch.Tensor, faces:torch.Tensor, points:torch.Tensor):
    # https://github.com/libigl/libigl/blob/main/include/igl/winding_number.cpp
    wns = signed_angle(vertices[faces[:, 0]], vertices[faces[:, 1]], points)
    return wns

def winding_number_3d_torch(vertices:torch.Tensor, faces:torch.Tensor, points:torch.Tensor):
    # https://github.com/libigl/libigl/blob/main/include/igl/winding_number.cpp
    wns = solid_angle(vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]], points)
    return wns

def signed_angle(v1:torch.Tensor, v2:torch.Tensor, points:torch.Tensor, bzv=-1, bzp=1024):
    ...  # TODO

def solid_angle(v1:torch.Tensor, v2:torch.Tensor, v3:torch.Tensor, points:torch.Tensor, bzv=-1, bzp=1024):
    '''
    Compute the signed solid angle subtended by the oriented 3d triangle (A,B,C) at some point P
    Reference: https://github.com/libigl/libigl/blob/main/include/igl/solid_angle.cpp

    v1, v2, v3: [V, 3]
    points: [N, 3]
    bzv, bzp: batch size of vertices, points
    angles: [N, V]
    winding_numbers: [N,]
    '''
    vertices = torch.stack([v1, v2, v3], dim=-2).unsqueeze(-4)  # [1, V, 3, 3]
    points = points.unsqueeze(-2).unsqueeze(-2)  # [N, 1, 1, 3]

    def _solid_angle(_matrix:torch.Tensor):
        '''
        matrix: [N, V, 3, 3]
        angles: [N, V]
        '''
        det = torch.linalg.det(_matrix)
        vl = torch.linalg.norm(_matrix, dim=-1)
        dp_0 = _matrix[..., 1,0] * _matrix[..., 2,0] + _matrix[..., 1,1] * _matrix[..., 2,1] + _matrix[..., 1,2] * _matrix[..., 2,2]
        dp_1 = _matrix[..., 2,0] * _matrix[..., 0,0] + _matrix[..., 2,1] * _matrix[..., 0,1] + _matrix[..., 2,2] * _matrix[..., 0,2]
        dp_2 = _matrix[..., 0,0] * _matrix[..., 1,0] + _matrix[..., 0,1] * _matrix[..., 1,1] + _matrix[..., 0,2] * _matrix[..., 1,2]
        u = vl[..., 0] * vl[..., 1] * vl[..., 2] + dp_0 * vl[..., 0] + dp_1 * vl[..., 1] + dp_2 * vl[..., 2]
        # NOTE: Only divide by 2 * pi instead of 4 * pi because there was a 2 out front
        _angles = torch.atan2(det, u) / (2 * torch.pi)
        return _angles

    shape = torch.broadcast_shapes(vertices.shape, points.shape)
    winding_numbers = torch.zeros(shape[:-3], device=vertices.device, dtype=vertices.dtype)
    for ip in tqdm(range(0, points.shape[-4], bzp)):
        if bzv == -1:
            matrix = vertices - points[..., ip:ip+bzp, :, :, :]  # [N, V, 3, 3]
            angles = _solid_angle(matrix)
        else:
            angles = torch.zeros(shape[:-4] + (bzp, shape[-3]), device=vertices.device, dtype=vertices.dtype)
            for iv in range(0, vertices.shape[-3], bzv):
                matrix = vertices[..., iv:iv+bzv, :, :] - points[..., ip:ip+bzp, :, :, :]
                angles[..., ip:ip+bzp, iv:iv+bzv] = _solid_angle(matrix)
        winding_numbers[..., ip:ip+bzp] = angles.sum(dim=-1)
    return winding_numbers


def winding_number_meshlib(vertices:np.ndarray, faces:np.ndarray, points:np.ndarray, use_cuda=False):
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    points: [N, 3], float32

    wns: [N,], float32, range(0.0, 1.0)
    '''
    # mesh_ml = mrmeshpy.loadMesh(input_mesh_path)
    mesh_ml  = mrmeshnumpy.meshFromFacesVerts(faces.astype(np.int64), vertices.astype(np.float32))
    if use_cuda:
        fwn_ml = mrcudapy.FastWindingNumber(mesh_ml)
    else:
        fwn_ml = mrmeshpy.FastWindingNumber(mesh_ml)
    points_ml = mrmeshnumpy.fromNumpyArray(points.astype(np.float32))
    wns_ml = mrmeshpy.std_vector_float()
    fwn_ml.calcFromVector(wns_ml, points_ml, beta=0.5)  # ~ 0.255 sec / 0.2 B, ~ 0.162 sec / 0.02 B

    # TODO: most of time is wasted here
    wns = np.asarray(wns_ml, dtype=np.float32)  # ~ 29.212 sec / 0.2 B, ~ 2.946 sec / 0.02 B
    # wns = mrmeshnumpy.toNumpyArray(wns_ml)  # NOTE: support std_vector_Vector3_float only
    return wns


def test_winding_number(
    input_mesh_path, output_points_path, N=200_000_000,
    test_igl=True, test_torch=False, test_meshlib=True,
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

    # compute winding number (igl), ~ 3.008 sec / 0.2 B, ~ 0.400 sec / 0.02 B
    if test_igl:
        t = perf_counter()
        wn_igl = winding_number_igl(mesh.vertices, mesh.faces, v)
        print('compute winding number (igl)', perf_counter() - t, 'sec')
    else:
        wn_igl = None

    # compute winding number (torch), too slow
    if test_torch:
        t = perf_counter()
        wn_torch = winding_number_torch(mesh.vertices, mesh.faces, v)
        print('compute winding number (torch)', perf_counter() - t, 'sec')
    else:
        wn_torch = None

    # compute winding number (meshlib), ~ 30.702 sec / 0.2 B, ~ 3.123 sec / 0.02 B
    if test_meshlib:
        t = perf_counter()
        wn_meshlib = winding_number_meshlib(mesh.vertices, mesh.faces, v)
        print('compute winding number (meshlib)', perf_counter() - t, 'sec')
    else:
        wn_meshlib = None

    # export points
    wn = wn_igl
    vm = (wn > 0.5)
    os.makedirs(os.path.dirname(output_points_path), exist_ok=True)
    trimesh.Trimesh(vertices=v, vertex_colors=np.clip(wn, 0.0, 1.0)[:, None], process=False).export(output_points_path)
    trimesh.Trimesh(vertices=v[vm, :], process=False).export(os.path.splitext(output_points_path)[0] + '_inner.ply')
    trimesh.Trimesh(vertices=v[~vm, :], process=False).export(os.path.splitext(output_points_path)[0] + '_outer.ply')


if __name__ == '__main__':
    input_mesh_path = 'gradio_examples_mesh/cute_wolf/textured_mesh.glb'
    output_points_path = 'gradio_examples_mesh_results/cute_wolf/textured_mesh_points.ply'

    test_winding_number(input_mesh_path, output_points_path)


