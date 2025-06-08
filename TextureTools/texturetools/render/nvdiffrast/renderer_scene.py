'''
Geometry Renderer for Panorama/Cubemap
'''
from glob import glob
import math
import os
from time import perf_counter
from typing import List, Optional, Dict, Tuple, Union
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from scipy.interpolate import (
    NearestNDInterpolator,
    LinearNDInterpolator,
    CloughTocher2DInterpolator,
)
import imageio

# https://docs.cupy.dev/en/stable/install.html
# pip install cupy-cuda12x
# import cupy as cp
# from cupyx.scipy.interpolate import (
#     NearestNDInterpolator,
#     LinearNDInterpolator,
#     CloughTocher2DInterpolator,
# )

# https://github.com/NVlabs/nvdiffrast.git
import nvdiffrast.torch as dr

from ...camera.generator import generate_orbit_views_c2ws, generate_intrinsics
from ...camera.conversion import (
    inverse_transform,
    project,
    c2ws_to_ray_matrices,
)
from .renderer_utils import image_to_tensor, tensor_to_image, tensor_to_video


def grid_to_cube(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    '''
    x, y: [H, W]
    cube: [6, H, W, 3]
    '''
    x, y = torch.broadcast_tensors(x, y)
    H, W = x.shape
    ones = torch.ones_like(x)
    cube = torch.stack([
        ones, -y, -x,  # right
        -ones, -y, x,  # left
        x, ones, y,    # top
        x, -ones, -y,  # down
        x, -y, ones,   # back
        -x, -y, -ones, # front
    ], dim=0).reshape(6, 3, H, W).permute(0, 2, 3, 1)
    return cube

def rays_to_cube(rays_o:torch.Tensor, rays_d:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    rays_o, rays_d: [..., 3], float32
    rays_idx: [..., 1], int64, cube index
    rays_uv: [..., 2], float32, uv in cube
    '''
    t_rtb = (1.0 - rays_o) / rays_d
    t_ldf = (-1.0 - rays_o) / rays_d
    rays_idx = ...  # TODO
    rays_uv = ...
    return rays_idx, rays_uv

def cubemap_to_flatten(cubemap:torch.Tensor):
    '''
    cubemap: right, left, top, down, back, front
    flatten:
        top
    left, front, right, back
        down
    '''
    N, H, W, C = cubemap.shape
    flatten = torch.zeros((3 * H, 4 * W, C), dtype=cubemap.dtype, device=cubemap.device)
    flatten[:H, W:2*W, :] = cubemap[2, :, :, :].flip(0)
    flatten[H:2*H, :W, :] = cubemap[1, :, :, :].flip(1)
    flatten[H:2*H, W:2*W, :] = cubemap[5, :, :, :].flip(1)
    flatten[H:2*H, 2*W:3*W, :] = cubemap[0, :, :, :].flip(1)
    flatten[H:2*H, 3*W:, :] = cubemap[4, :, :, :].flip(1)
    flatten[2*H:, W:2*W, :] = cubemap[3, :, :, :].flip(0)
    return flatten

def cubemap_to_flatten_kjl(cubemap:torch.Tensor):
    # NOTE: specific function for kujiale dataset
    N, H, W, C = cubemap.shape
    flatten = torch.zeros((3 * H, 4 * W, C), dtype=cubemap.dtype, device=cubemap.device)
    flatten[:H, W:2*W, :] = cubemap[0, :, :, :].transpose(0, 1).flip(1)
    flatten[H:2*H, :W, :] = cubemap[1, :, :, :]
    flatten[H:2*H, W:2*W, :] = cubemap[2, :, :, :]
    flatten[H:2*H, 2*W:3*W, :] = cubemap[3, :, :, :]
    flatten[H:2*H, 3*W:, :] = cubemap[4, :, :, :]
    flatten[2*H:, W:2*W, :] = cubemap[5, :, :, :].transpose(0, 1).flip(0)
    return flatten

def flatten_to_cubemap(flatten:torch.Tensor):
    '''
    cubemap: right, left, top, down, back, front
    flatten:
        top
    left, front, right, back
        down
    '''
    HF, WF, C = flatten.shape
    H, W = HF // 3, WF // 4
    cubemap = torch.zeros((6, H, W , C), dtype=flatten.dtype, device=flatten.device)
    cubemap[2, :, :, :] = flatten[:H, W:2*W, :].flip(0)
    cubemap[1, :, :, :] = flatten[H:2*H, :W, :].flip(1)
    cubemap[5, :, :, :] = flatten[H:2*H, W:2*W, :].flip(1)
    cubemap[0, :, :, :] = flatten[H:2*H, 2*W:3*W, :].flip(1)
    cubemap[4, :, :, :] = flatten[H:2*H, 3*W:, :].flip(1)
    cubemap[3, :, :, :] = flatten[2*H:, W:2*W, :].flip(0)
    return cubemap

def latlong_to_cubemap(latlong_map:torch.Tensor, render_size:Union[int, Tuple[int]]) -> torch.Tensor:
    '''
    latlong_map: [Hi, Wi, 3]
    render_size: Ho, Wo
    cubemap: [6, Ho, Wo, 3]

    references:
        * http://www.paulbourke.net/panorama/cubemaps/
        * https://github.com/NVlabs/nvdiffrec/blob/main/render/util.py
    '''
    dtype = latlong_map.dtype
    device = latlong_map.device
    height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=dtype, device=device),
        torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=dtype, device=device),
        indexing='ij',
    )
    cube = grid_to_cube(gx, gy)
    rd = torch.nn.functional.normalize(cube, dim=-1)
    phi = torch.atan2(rd[..., [0]], -rd[..., [2]])  # phi in range(-pi, pi)
    theta = torch.acos(torch.clamp(rd[..., [1]], min=-1, max=1))  # theta in range(0, pi)
    uv = torch.cat([phi / (2 * torch.pi) + 0.5, theta / torch.pi], dim=-1)  # scale to (0, 1)
    # NOTE: except for the range, dr.texture is as same as torch.nn.functional.grid_sample
    cubemap = dr.texture(latlong_map.unsqueeze(0), uv, filter_mode='linear', boundary_mode='wrap')
    return cubemap

def cubemap_to_latlong(cubemap:torch.Tensor, render_size:Union[int, Tuple[int]]) -> torch.Tensor:
    '''
    cubemap: [6, Hi, Wi, 3]
    render_size: Ho, Wo
    latlong_map: [Ho, Wo, 3]

    NOTE: cubemap_to_latlong and latlong_to_cubemap are not reciprocal
    '''
    dtype = cubemap.dtype
    device = cubemap.device
    height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=dtype, device=device),
        torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=dtype, device=device),
        indexing='ij',
    )
    psi = gx * torch.pi  # phi in range(-pi, pi)
    theta = gy * (torch.pi / 2) + torch.pi / 2  # theta in range(0, pi)
    rd = torch.stack([
        torch.sin(theta) * torch.sin(psi),
        torch.cos(theta),
        -torch.sin(theta) * torch.cos(psi)
    ], dim=-1)
    latlong_map = dr.texture(cubemap.unsqueeze(0), rd.unsqueeze(0), filter_mode='linear', boundary_mode='cube').squeeze(0)
    return latlong_map

def latlong_depth_to_pcd(depth:torch.Tensor):
    '''
    depth: [H, W, 1]
    xyz: [H, W, 3]
    '''
    H, W, _ = depth.shape
    dtype = depth.dtype
    device = depth.device
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=dtype, device=device),
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=dtype, device=device),
        indexing='ij',
    )
    psi = gx * torch.pi  # phi in range(-pi, pi)
    theta = gy * (torch.pi / 2) + torch.pi / 2  # theta in range(0, pi)
    rd = torch.stack([
        torch.sin(theta) * torch.sin(psi),
        torch.cos(theta),
        -torch.sin(theta) * torch.cos(psi)
    ], dim=-1)
    xyz = depth * rd
    return xyz


class NVDiffRendererScene:
    def __init__(self, device='cuda', latlong_map:Optional[torch.Tensor]=None, cubemap:Optional[torch.Tensor]=None):
        super().__init__()
        # NOTE: no check here
        # assert latlong_map is None or cubemap is not None
        self.device = torch.device(device)
        self.latlong_map = latlong_map.to(device=self.device, dtype=torch.float32, memory_format=torch.contiguous_format) if latlong_map is not None else None
        self.cubemap = cubemap.to(device=self.device, dtype=torch.float32, memory_format=torch.contiguous_format) if cubemap is not None else None
        self.enable_nvdiffrast_cuda_ctx()

    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def enable_nvdiffrast_opengl_ctx(self):
        self.ctx = dr.RasterizeGLContext(device=self.device)

    def clear(self):
        self.latlong_map = None
        self.cubemap = None

    @classmethod
    def from_files(cls, device='cuda', latlong_map_path:Optional[str]=None, cubemap_paths:Optional[List[str]]=None, cubemap_kjl=False):
        assert latlong_map_path is not None or cubemap_paths is not None
        if latlong_map_path is not None:
            latlong_map = Image.open(latlong_map_path)
            latlong_map = image_to_tensor(latlong_map, device=device)
        else:
            latlong_map = None
        if cubemap_paths is not None:
            assert len(cubemap_paths) == 6
            cubemap = [Image.open(p) for p in cubemap_paths]
            cubemap = image_to_tensor(cubemap, device=device)
            if cubemap_kjl:
                cubemap = flatten_to_cubemap(cubemap_to_flatten_kjl(cubemap))
        else:
            cubemap = None
        return cls(device=device, latlong_map=latlong_map, cubemap=cubemap)

    @classmethod
    def from_images(cls, device='cuda', latlong_map:Optional[Image.Image]=None, cubemaps:Optional[List[Image.Image]]=None, cubemap_kjl=False):
        assert latlong_map is not None or cubemaps is not None
        if latlong_map is not None:
            latlong_map = image_to_tensor(latlong_map, device=device)
        else:
            latlong_map = None
        if cubemaps is not None:
            assert len(cubemaps) == 6
            cubemap = image_to_tensor(cubemaps, device=device)
            if cubemap_kjl:
                cubemap = flatten_to_cubemap(cubemap_to_flatten_kjl(cubemap))
        else:
            cubemap = None
        return cls(device=device, latlong_map=latlong_map, cubemap=cubemap)

    def update_from_files(self, latlong_map_path:Optional[str]=None, cubemap_paths:Optional[List[str]]=None, cubemap_kjl=False):
        assert latlong_map_path is not None or cubemap_paths is not None
        if latlong_map_path is not None:
            latlong_map = Image.open(latlong_map_path)
            latlong_map = image_to_tensor(latlong_map, device=self.device)
        else:
            latlong_map = None
        if cubemap_paths is not None:
            assert len(cubemap_paths) == 6
            cubemap = [Image.open(p) for p in cubemap_paths]
            cubemap = image_to_tensor(cubemap, device=self.device)
            if cubemap_kjl:
                cubemap = flatten_to_cubemap(cubemap_to_flatten_kjl(cubemap))
        else:
            cubemap = None
        self.latlong_map = latlong_map
        self.cubemap = cubemap
        return self

    def update_from_images(self, latlong_map:Optional[Image.Image]=None, cubemaps:Optional[List[Image.Image]]=None, cubemap_kjl=False):
        assert latlong_map is not None or cubemaps is not None
        if latlong_map is not None:
            latlong_map = image_to_tensor(latlong_map, device=self.device)
        else:
            latlong_map = None
        if cubemaps is not None:
            assert len(cubemaps) == 6
            cubemap = image_to_tensor(cubemaps, device=self.device)
            if cubemap_kjl:
                cubemap = flatten_to_cubemap(cubemap_to_flatten_kjl(cubemap))
        else:
            cubemap = None
        self.latlong_map = latlong_map
        self.cubemap = cubemap
        return self

    def perspective_rendering(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor, render_size:Union[int, Tuple[int]],
        render_cubemap=False,
        render_uv=False,
        render_latlong_map=False,
    ):
        '''
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        '''
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        rays_o, rays_d = c2ws_to_ray_matrices(c2ws, intrinsics, height, width, perspective=True)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        out = {'rays_o': rays_o, 'rays_d': rays_d}

        if render_cubemap:
            cubemap_attr = dr.texture(self.cubemap.unsqueeze(0), rays_d, filter_mode='linear', boundary_mode='cube')
            out.update({'cubemap_attr': cubemap_attr})

        if render_uv:
            phi = torch.atan2(rays_d[..., [0]], -rays_d[..., [2]])
            theta = torch.acos(torch.clamp(rays_d[..., [1]], min=-1, max=1))
            uv = torch.cat([phi / (2 * torch.pi) + 0.5, theta / torch.pi], dim=-1)
            out.update({'uv': uv})

            if render_latlong_map:
                latlong_map_attr = dr.texture(self.latlong_map.unsqueeze(0), uv, filter_mode='linear')
                out.update({'latlong_map_attr': latlong_map_attr})
        return out

    def perspective_inverse_rendering_scipy(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor, images:torch.Tensor,
        render_size:Union[int, Tuple[int]],
        texture_size:Union[int, Tuple[int]],
        render_cubemap=False,
        render_uv=False,
        render_latlong_map=False,
    ):
        '''
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        images: [M, H, W, C]
        '''
        batch_size = c2ws.shape[0]
        dtype = c2ws.dtype
        device = c2ws.device
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        height_t, width_t = (texture_size, texture_size) if isinstance(texture_size, int) else texture_size
        rays_o, rays_d = c2ws_to_ray_matrices(c2ws, intrinsics, height, width, perspective=True)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        out = {'rays_o': rays_o, 'rays_d': rays_d}

        if render_cubemap:
            raise NotImplementedError
            rays_idx, rays_uv = rays_to_cube(rays_o, rays_d)
            cubemap_attr = ...
            out.update({'cubemap_attr': cubemap_attr})

        if render_uv:
            phi = torch.atan2(rays_d[..., [0]], -rays_d[..., [2]])
            theta = torch.acos(torch.clamp(rays_d[..., [1]], min=-1, max=1))
            uv = torch.cat([phi / (2 * torch.pi) + 0.5, theta / torch.pi], dim=-1)
            out.update({'uv': uv})

            if render_latlong_map:
                gy, gx = torch.meshgrid(
                    torch.linspace(-1.0 + 1.0 / height_t, 1.0 - 1.0 / height_t, height_t, dtype=dtype, device=device),
                    torch.linspace(-1.0 + 1.0 / width_t, 1.0 - 1.0 / width_t, width_t, dtype=dtype, device=device),
                    indexing='ij',
                )
                guv = torch.stack([gx, gy], dim=-1) * 0.5 + 0.5
                samples_u = np.asarray(uv.detach().cpu().numpy(), dtype=np.float32)
                samples_v = np.asarray(images.detach().cpu().numpy(), dtype=np.float32)
                predicts_u = np.asarray(guv.detach().cpu().numpy(), dtype=np.float32)
                print(f"Interpolating, M={math.prod(samples_u.shape[:-1])}, N={math.prod(predicts_u.shape[:-1])}, may take a while ...")
                t = perf_counter()
                interp = LinearNDInterpolator(samples_u.reshape(-1, 2), samples_v.reshape(-1, 3))
                predicts_v = interp(predicts_u)
                print(f"Interpolating wastes {perf_counter() - t} sec")
                latlong_map_attr = torch.as_tensor(predicts_v, dtype=dtype, device=device)
                latlong_map_attr = torch.nan_to_num(latlong_map_attr, nan=0.0, posinf=0.0, neginf=0.0)
                out.update({'latlong_map_attr': latlong_map_attr})
        return out

    def perspective_inverse_rendering_torch(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor, images:torch.Tensor,
        render_size:Union[int, Tuple[int]],
        texture_size:Union[int, Tuple[int]],
        render_cubemap=False,
        render_uv=False,
        render_latlong_map=False,
    ):
        '''
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        images: [M, H, W, C]
        '''
        batch_size = c2ws.shape[0]
        dtype = c2ws.dtype
        device = c2ws.device
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        height_t, width_t = (texture_size, texture_size) if isinstance(texture_size, int) else texture_size
        rays_o, rays_d = c2ws_to_ray_matrices(c2ws, intrinsics, height, width, perspective=True)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        out = {'rays_o': rays_o, 'rays_d': rays_d}

        if render_cubemap:
            raise NotImplementedError
            rays_idx, rays_uv = rays_to_cube(rays_o, rays_d)
            cubemap_attr = ...
            out.update({'cubemap_attr': cubemap_attr})

        if render_uv:
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / height_t, 1.0 - 1.0 / height_t, height_t, dtype=dtype, device=device),
                torch.linspace(-1.0 + 1.0 / width_t, 1.0 - 1.0 / width_t, width_t, dtype=dtype, device=device),
                indexing='ij',
            )
            psi = gx * torch.pi
            theta = gy * (torch.pi / 2) + torch.pi / 2
            rd = torch.stack([
                torch.sin(theta) * torch.sin(psi),
                torch.cos(theta),
                -torch.sin(theta) * torch.cos(psi)
            ], dim=-1)
            rd_ndc, rd_depth = project(inverse_transform(torch.cat([rd, torch.ones_like(rd[..., [0]])], dim=-1), [c2ws.unsqueeze(-3)]), intrinsics, perspective=True)
            rd_mask = torch.logical_and(torch.logical_and(rd_ndc >= -1.0, rd_ndc <= 1.0).all(dim=-1, keepdim=True), rd_depth > 0.0)
            uv = rd_ndc
            uv_alpha = rd_mask.to(dtype=dtype)
            out.update({'uv': uv, 'uv_alpha': uv_alpha})

            if render_latlong_map:
                latlong_map_attrs = torch.nn.functional.grid_sample(
                    images.permute(0, 3, 1, 2),
                    uv,
                    padding_mode='zeros',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                latlong_map_attr = (latlong_map_attrs * uv_alpha).sum(dim=0) / uv_alpha.sum(dim=0)
                latlong_map_attr = latlong_map_attr.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
                out.update({'latlong_map_attr': latlong_map_attr})
        return out



#### some test functions ####


def test_convert():
    pano_path = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/0.png'
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in range(6)]
    result_dir = 'test_result/test_panorama'

    pano = Image.open(pano_path)
    pano = image_to_tensor(pano)

    cubemap = [Image.open(p) for p in cubemap_path]
    cubemap = image_to_tensor(cubemap)

    cubemap_converted = latlong_to_cubemap(pano.contiguous(), (512, 512))
    flatten_converted = cubemap_to_flatten(cubemap_converted)
    pano_converted = cubemap_to_latlong(cubemap_converted.contiguous(), (1024, 2048))

    flatten_kjl_converted = cubemap_to_flatten_kjl(cubemap)
    cubemap_kjl_converted = flatten_to_cubemap(flatten_kjl_converted)
    pano_kjl_converted = cubemap_to_latlong(cubemap_kjl_converted.contiguous(), (1024, 2048))

    os.makedirs(result_dir, exist_ok=True)
    tensor_to_image(flatten_converted).save(os.path.join(result_dir, 'flatten.png'))
    tensor_to_image(pano_converted).save(os.path.join(result_dir, 'pano.png'))
    tensor_to_image(flatten_kjl_converted).save(os.path.join(result_dir, 'flatten_kjl.png'))
    tensor_to_image(pano_kjl_converted).save(os.path.join(result_dir, 'pano_kjl.png'))

def test_multi_converts():
    pano_path = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/0.png'
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in range(6)]
    result_dir = 'test_result/test_panorama'

    pano = Image.open(pano_path)
    pano = image_to_tensor(pano)

    cubemap = [Image.open(p) for p in cubemap_path]
    cubemap = image_to_tensor(cubemap)

    flatten_kjl_converted = cubemap_to_flatten_kjl(cubemap)
    cubemap_kjl_converted = flatten_to_cubemap(flatten_kjl_converted)
    pano_kjl_converted = cubemap_to_latlong(cubemap_kjl_converted.contiguous(), (1024, 2048))
    pano_kjl_converted_raw = pano_kjl_converted.clone()
    cubemap_kjl_converted_raw = cubemap_kjl_converted.clone()

    N = 20
    pano_psnr_list = []
    cubemap_psnr_list = []
    for iter in range(N):
        pano_kjl_converted = cubemap_to_latlong(cubemap_kjl_converted.contiguous(), (1024, 2048))
        cubemap_kjl_converted = latlong_to_cubemap(pano_kjl_converted.contiguous(), (512, 512))

        pano_psnr = 10 * torch.log10(1 / (pano_kjl_converted.cpu() - pano_kjl_converted_raw.cpu()).square().mean())
        cubemap_psnr = 10 * torch.log10(1 / (cubemap_kjl_converted.cpu() - cubemap_kjl_converted_raw.cpu()).square().mean())
        pano_psnr_list.append(pano_psnr.item())
        cubemap_psnr_list.append(cubemap_psnr.item())

    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(12, 10), dpi=200)
    plt.scatter(range(N), pano_psnr_list, label='panorama')
    plt.scatter(range(N), cubemap_psnr_list, label='cubemap')
    plt.legend()
    plt.xlabel('conversion steps')
    plt.ylabel('psnr')
    plt.xticks(range(N))
    plt.savefig(os.path.join(result_dir, 'multi_converts.png'))

def test_renderer():
    pano_path = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/0.png'
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in range(6)]
    result_dir = 'test_result/test_panorama'

    renderer = NVDiffRendererScene.from_files(latlong_map_path=pano_path, cubemap_paths=cubemap_path, cubemap_kjl=True)

    N, H, W = 250, 1024, 1024
    c2ws = generate_orbit_views_c2ws(N+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:N]
    c2ws[:, :3, 3] = 0.0
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    for _ in range(10):
        t = perf_counter()
        map_attr_cubemap = renderer.perspective_rendering(c2ws, intrinsics, (H, W), render_cubemap=True)['cubemap_attr']
        print('renderer_cubemap', perf_counter() - t)

    for _ in range(10):
        t = perf_counter()
        map_attr_pano = renderer.perspective_rendering(c2ws, intrinsics, (H, W), render_uv=True, render_latlong_map=True)['latlong_map_attr']
        print('renderer_pano', perf_counter() - t)

    psnr = 10 * torch.log10(1 / (map_attr_cubemap.cpu() - map_attr_pano.cpu()).square().mean())
    print('n_views', N, 'psnr', psnr.item())

    os.makedirs(result_dir, exist_ok=True)
    imageio.mimsave(
        os.path.join(result_dir, 'map_attr_cubemap.mp4'),
        tensor_to_video(map_attr_cubemap)[..., [0,1,2]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'map_attr_pano.mp4'),
        tensor_to_video(map_attr_pano)[..., [0,1,2]],
        fps=25,
    )

def test_inverse_renderer_single():
    pano_path = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/0.png'
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in range(6)]
    result_dir = 'test_result/test_panorama'

    renderer = NVDiffRendererScene.from_files(latlong_map_path=pano_path, cubemap_paths=cubemap_path, cubemap_kjl=True)

    N, H, W = 8, 1024, 1024
    c2ws = generate_orbit_views_c2ws(N+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:N]
    c2ws[:, :3, 3] = 0.0
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    render_out = renderer.perspective_rendering(c2ws, intrinsics, (H, W), render_cubemap=True, render_uv=True, render_latlong_map=True)
    map_uv = render_out['uv']
    map_uv_vis = torch.cat([map_uv, torch.ones_like(map_uv[..., [0]])], dim=-1)
    map_attr_cubemap = render_out['cubemap_attr']
    map_attr_pano = render_out['latlong_map_attr']

    for _ in range(1):  # NOTE: too slow
        t = perf_counter()
        map_attr_pano_rec_single_scipy = renderer.perspective_inverse_rendering_scipy(c2ws[[0]], intrinsics, map_attr_pano[[0]], (H, W), (1024, 2048), render_uv=True, render_latlong_map=True)['latlong_map_attr']
        print('scipy method', perf_counter() - t)

    for _ in range(10):
        t = perf_counter()
        map_attr_pano_rec_single_torch = renderer.perspective_inverse_rendering_torch(c2ws[[0]], intrinsics, map_attr_pano[[0]], (H, W), (1024, 2048), render_uv=True, render_latlong_map=True)['latlong_map_attr']
        print('pytorch method', perf_counter() - t)

    #### begin: consistency checking ####
    from ...utils.extra_scene_utils import Perspective2Panorama
    for _ in range(10):
        t = perf_counter()
        map_attr_pano_rec_single_fc, _ = Perspective2Panorama(tensor_to_video(map_attr_pano[[0]])[0], 49.1, 0.0, 0.0).GetEquirec(1024, 2048)
        map_attr_pano_rec_single_fc = map_attr_pano_rec_single_fc.astype(np.uint8)
        print('fangchuan method', perf_counter() - t)
    #### end: consistency checking ####

    os.makedirs(result_dir, exist_ok=True)
    tensor_to_image(map_attr_pano.permute(1, 0, 2, 3).reshape(H, N*W, -1)).save(os.path.join(result_dir, 'image_rec.png'))
    tensor_to_image(map_uv_vis.permute(1, 0, 2, 3).reshape(H, N*W, -1)).save(os.path.join(result_dir, 'image_uv.png'))
    tensor_to_image(map_attr_pano_rec_single_scipy).save(os.path.join(result_dir, 'pano_rec_single_scipy.png'))
    tensor_to_image(map_attr_pano_rec_single_torch).save(os.path.join(result_dir, 'pano_rec_single_torch.png'))
    Image.fromarray(map_attr_pano_rec_single_fc).save(os.path.join(result_dir, 'pano_rec_single_fc.png'))


def test_inverse_renderer_multi():
    pano_path = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/0.png'
    cubemap_path = [f'/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/cubemap_rgb/0_skybox{i}_sami.png' for i in range(6)]
    result_dir = 'test_result/test_panorama'

    renderer = NVDiffRendererScene.from_files(latlong_map_path=pano_path, cubemap_paths=cubemap_path, cubemap_kjl=True)

    N, H, W = 8, 1024, 1024
    c2ws = generate_orbit_views_c2ws(N+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:N]
    c2ws[:, :3, 3] = 0.0
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    render_out = renderer.perspective_rendering(c2ws, intrinsics, (H, W), render_cubemap=True, render_uv=True, render_latlong_map=True)
    map_uv = render_out['uv']
    map_uv_vis = torch.cat([map_uv, torch.ones_like(map_uv[..., [0]])], dim=-1)
    map_attr_cubemap = render_out['cubemap_attr']
    map_attr_pano = render_out['latlong_map_attr']

    # for _ in range(1):  # NOTE: too slow
    #     t = perf_counter()
    #     map_attr_pano_rec_multi_scipy = renderer.perspective_inverse_rendering_scipy(c2ws, intrinsics, map_attr_pano, (H, W), (1024, 2048), render_uv=True, render_latlong_map=True)['latlong_map_attr']
    #     print('scipy method', perf_counter() - t)

    #### begin: consistency checking ####
    from ...utils.extra_scene_utils import MultiPers2Panorama
    for _ in range(10):
        t = perf_counter()
        map_attr_pano_rec_multi_fc, _ = MultiPers2Panorama(tensor_to_video(map_attr_pano), [[49.1, 360.0 * i / N, 0.0] for i in range(N)]).GetEquirec(1024, 2048)
        map_attr_pano_rec_multi_fc = map_attr_pano_rec_multi_fc.astype(np.uint8)
        print('fangchuan method', perf_counter() - t)
    #### end: consistency checking ####

    for _ in range(10):
        t = perf_counter()
        map_attr_pano_rec_multi_torch = renderer.perspective_inverse_rendering_torch(c2ws, intrinsics, map_attr_pano, (H, W), (1024, 2048), render_uv=True, render_latlong_map=True)['latlong_map_attr']
        print('pytorch method', perf_counter() - t)

    os.makedirs(result_dir, exist_ok=True)
    tensor_to_image(map_attr_pano.permute(1, 0, 2, 3).reshape(H, N*W, -1)).save(os.path.join(result_dir, 'image_rec.png'))
    tensor_to_image(map_uv_vis.permute(1, 0, 2, 3).reshape(H, N*W, -1)).save(os.path.join(result_dir, 'image_uv.png'))
    # tensor_to_image(map_attr_pano_rec_multi_scipy).save(os.path.join(result_dir, 'pano_rec_multi_scipy.png'))
    Image.fromarray(map_attr_pano_rec_multi_fc).save(os.path.join(result_dir, 'pano_rec_multi_fc.png'))
    tensor_to_image(map_attr_pano_rec_multi_torch).save(os.path.join(result_dir, 'pano_rec_multi_torch.png'))


