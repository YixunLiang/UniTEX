from typing import Optional
import math
import numpy as np
import torch
from .rotation import euler_angles_to_matrix


def lookat_to_matrix(lookat:torch.Tensor) -> torch.Tensor:
    '''
    lookat: [..., 3], camera locations looking at origin
    c2ws: [..., 4, 4]
        * world: x forward, y right, z up, need to transform xyz to zxy
        * camera: z forward, x right, y up
    '''
    batch_shape = lookat.shape[:-1]
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    xyz_to_zxy = torch.as_tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=lookat.dtype, device=lookat.device)
    # NOTE: camera locations are opposite from ray directions
    z_axis = torch.nn.functional.normalize(lookat, dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis_mask = (x_axis == 0).all(dim=-1, keepdim=True)
    if x_axis_mask.sum() > 0:
        # NOTE: top and down is not well defined, hard code here
        x_axis = torch.where(x_axis_mask, e2, x_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    c2ws = torch.cat([
        torch.cat([rots, lookat.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    # NOTE: world f/r/u is x/y/z, camera f/r/u is z/x/y, so we need to transform x/y/z to z/x/y for c2ws
    c2ws = torch.matmul(xyz_to_zxy, c2ws)
    return c2ws

def make_orthonormals(N:torch.Tensor):
    # https://projects.blender.org/blender/cycles/src/branch/main/src/util/math.h
    Nx, Ny, Nz = N.split([1, 1, 1], dim=-1)
    T = torch.where(
        torch.logical_or(Nx != Ny, Nx != Nz), 
        torch.cat([Nz - Ny, Nx - Nz, Ny - Nx], dim=-1),  # (1,1,1)x N
        torch.cat([Nz - Ny, Nx + Nz, -Ny - Nx], dim=-1),  # (-1,1,1)x N
    )
    T = torch.nn.functional.normalize(T, dim=-1)
    B = torch.linalg.cross(N, T)
    return T, B

def sample_uniform_disk(shape, generator=None):
    # https://projects.blender.org/blender/cycles/src/branch/main/src/kernel/sample/mapping.h
    a = 2.0 * torch.rand(shape, dtype=torch.float32, generator=generator) - 1.0
    b = 2.0 * torch.rand(shape, dtype=torch.float32, generator=generator) - 1.0
    c = a ** 2 > b ** 2
    r = torch.where(c, a, b)
    phi = torch.where(c, (torch.pi / 4) * (b / a), (torch.pi / 2) - (torch.pi / 4) * (a / b))
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return x, y

def sample_gaussian_disk(shape, generator=None):
    x = torch.randn(shape, dtype=torch.float32, generator=generator)
    y = torch.randn(shape, dtype=torch.float32, generator=generator)
    return x, y

def sample_uniform_hemisphere(N:torch.Tensor, generator=None, semi=True):
    # https://projects.blender.org/blender/cycles/src/branch/main/src/kernel/sample/mapping.h
    x, y = sample_uniform_disk(N.shape, generator=generator)
    z = 1 - (x ** 2 + y ** 2)
    x = x * torch.sqrt(z + 1.0)
    y = y * torch.sqrt(z + 1.0)
    T, B = make_orthonormals(N)
    if not semi:
        z = z * torch.where(torch.randn(N.shape, generator=generator) > 0.0, 1.0, -1.0)
    wo = x * T + y * B + z * N
    # pdf = 2 * torch.pi
    return wo

def sample_near_vector(N:torch.Tensor, scale_x=0.1, scale_y=0.1, generator=None):
    x, y = sample_gaussian_disk(N.shape, generator=generator)
    r = torch.sqrt((scale_x ** 2) * (x ** 2) + (scale_y ** 2) * (y ** 2) + 1)
    x = scale_x * x / r
    y = scale_y * y / r
    z = 1 / r
    T, B = make_orthonormals(N)
    wo = x * T + y * B + z * N
    return wo

def generate_intrinsics(f_x: float, f_y: float, fov=True, degree=False):
    '''
    f_x, f_y: 
        * focal length divide width/height for perspective camera
        * fov degree or radians for perspective camera
        * scale for orthogonal camera
    intrinsics: [3, 3], normalized
    '''
    if fov:
        if degree:
            f_x = math.radians(f_x)
            f_y = math.radians(f_y)
        f_x_div_W = 1 / (2 * math.tan(f_x / 2))
        f_y_div_H = 1 / (2 * math.tan(f_y / 2))
    else:
        f_x_div_W = f_x
        f_y_div_H = f_y
    return torch.as_tensor([
        [f_x_div_W, 0.0, 0.5],
        [0.0, f_y_div_H, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def generate_orbit_views_c2ws(num_views: int, radius: float = 1.0, height: float = 0.0, theta_0: float = 0.0, degree=False):
    if degree:
        theta_0 = math.radians(theta_0)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    theta = torch.linspace(theta_0, 2.0 * math.pi + theta_0, num_views, dtype=torch.float32)
    x = projected_radius * torch.cos(theta)
    y = projected_radius * torch.sin(theta)
    z = torch.full((num_views,), fill_value=height, dtype=torch.float32)
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix(xyz)
    return c2ws

def generate_hemisphere_views_c2ws(num_views:int, radius: float = 1.0, seed: Optional[int] = None, semi: bool = True):
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0).repeat(num_views, 1)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    xyz = radius * sample_uniform_hemisphere(e3, generator=generator, semi=semi)
    c2ws = lookat_to_matrix(xyz)
    return c2ws

def generate_semisphere_views_c2ws(num_views:int, radius: float = 1.0, seed: Optional[int] = None, hemi: bool = False):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    random_vector = torch.randn((num_views, 3), dtype=torch.float32, generator=generator)
    random_vector = torch.nn.functional.normalize(random_vector, dim=-1)
    if hemi:
        random_vector[:, 2] = torch.abs(random_vector[:, 2])
    xyz = radius * random_vector
    c2ws = lookat_to_matrix(xyz)
    return c2ws

def generate_near_front_views_c2ws(num_views:int, radius: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0, seed: Optional[int] = None):
    e1 = torch.as_tensor([1.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0).repeat(num_views, 1)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    xyz = radius * sample_near_vector(e1, scale_x=scale_x, scale_y=scale_y, generator=generator)
    c2ws = lookat_to_matrix(xyz)
    return c2ws

def generate_box_views_c2ws(radius=2.8):
    # NOTE: top and down is not well defined, hard code here
    return torch.tensor([
        [[ 1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [-1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[-1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [-0.0000, -0.0000, -1.0000, -radius],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.0000,  0.0000, -1.0000, -radius],
        [ 0.0000,  1.0000,  0.0000, -0.0000],
        [ 1.0000, -0.0000,  0.0000, -0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[-1.0000,  0.0000, -0.0000, -0.0000],
        [-0.0000, -0.0000, -1.0000, -radius],
        [-0.0000, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    ], dtype=torch.float32)

def generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8)):
    eulers = [
        [yaw, pitch, roll]
        for roll in np.linspace(0.0, 360.0, steps[2], endpoint=False)
        for pitch in np.linspace(0.0, 360.0, steps[1], endpoint=False)
        for yaw in np.linspace(0.0, 360.0, steps[0], endpoint=False)
    ]
    eulers = torch.as_tensor(eulers, dtype=torch.float32)
    trans = torch.as_tensor([[0.0, 0.0, radius]], dtype=torch.float32)
    rots = euler_angles_to_matrix(torch.deg2rad(eulers), convention='XYZ')
    c2ws = torch.eye(4).repeat(rots.shape[0], 1, 1)
    c2ws[:, :3, :3] = rots
    c2ws[:, :3, [3]] = torch.matmul(trans, rots.permute(0, 2, 1)).permute(0, 2, 1)
    return c2ws

