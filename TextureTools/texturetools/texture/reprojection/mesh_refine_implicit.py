'''
Mesh refine inplicit in MVLRM/AIDoll/iris3d_cr/MVMeshRecon/MVDiffusion
* UV + texture
'''

from glob import glob
import os
from typing import List, Optional, Union
import cv2
import imageio
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from ...camera.conversion import undiscretize
from ...render.nvdiffrast.renderer_base import NVDiffRendererBase
from ...mesh.structure import Mesh, Texture

def overshoot_sigmoid(x, eps=0.1):
    return torch.sigmoid(x) / (1 - 2 * eps) + eps

class OvershootSigmoid(nn.Module):
    def __init__(self, eps=0.1) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return overshoot_sigmoid(x, eps=self.eps)
        
def trunc_rev_sigmoid(x, eps=1e-2):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

class TruncRevSigmoid(nn.Module):
    def __init__(self, eps=1e-2) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return trunc_rev_sigmoid(x, eps=self.eps)

def make_grid(H, W, dtype=torch.float32, device='cpu'):
    ys = torch.arange(H, dtype=dtype, device=device)
    xs = torch.arange(W, dtype=dtype, device=device)
    grid_i = torch.cartesian_prod(ys, xs).reshape(H, W, 2).flip(-1)
    grid_f = undiscretize(grid_i, H=H, W=W)
    return grid_f


class LipipsLoss(nn.Module):
    def __init__(self, input_H=512, input_W=512):
        super().__init__()
        import lpips
        self.input_H = input_H
        self.input_W = input_W
        self.model = lpips.LPIPS(net='vgg').requires_grad_(False)
    
    def forward(self, x, y):
        '''
        x, y: [N, C, H, W]
        '''
        if x.ndim > 4:
            x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        if y.ndim > 4:
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
        if x.shape[-2] != self.input_H or x.shape[-1] != self.input_W:
            x = torch.nn.functional.interpolate(x, (512, 512), mode='nearest')
        if y.shape[-2] != self.input_H or y.shape[-1] != self.input_W:
            y = torch.nn.functional.interpolate(y, (512, 512), mode='nearest')
        return self.model(x, y, normalize=True).mean()


class FourierTransform(nn.Module):
    def __init__(self, n:int):
        super().__init__()
        self.n = n
        scale = torch.arange(n, dtype=torch.float32)
        self.register_buffer('scale', scale)  # torch.pow(2.0, scale)

    def forward(self, x:torch.Tensor):
        x = x * torch.pi + torch.pi
        kx = self.scale * x.unsqueeze(-1)
        return torch.cat([torch.sin(kx), torch.cos(kx)], dim=-1).reshape(*x.shape[:-1], -1)


class TinyMLPV1(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        assert n_dim in [2, 3], f'support n_dim in [2, 3] but {n_dim}'
        self.n_dim = n_dim
        
        self.model = nn.Sequential(
            FourierTransform(32),
            nn.Linear(n_dim * 2 * 32, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 3),
        )
    
    def forward(self, x:torch.Tensor):
        '''
        x: [B, H, W, n_dim]
        y: [B, H, W, 3]
        '''
        return self.model(x)


class TinyMLPV2(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        assert n_dim in [2, 3], f'support n_dim in [2, 3] but {n_dim}'
        self.n_dim = n_dim
        
        if n_dim == 2:
            self.grid = nn.Parameter(torch.zeros((8, 16, 16)))
        elif n_dim == 3:
            self.grid = nn.Parameter(torch.zeros((8, 16, 16, 16)))
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )
    
    def forward(self, x:torch.Tensor):
        '''
        x: [B, H, W, n_dim]
        y: [B, H, W, 3]
        '''
        x = torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).repeat_interleave(x.shape[0], dim=0), 
            x if self.n_dim == 2 else x.unsqueeze(-4), 
            mode='bilinear',
            align_corners=False,
        )
        if self.n_dim == 3:
            x = x.squeeze(-3)
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        return x


class TinyMLPV3(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        assert n_dim in [2, 3], f'support n_dim in [2, 3] but {n_dim}'
        self.n_dim = n_dim

        import tinycudann as tcnn # type: ignore
        self.encoder = tcnn.Encoding(
            n_input_dims=n_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
            },
        )
        self.color_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 16,
                "n_hidden_layers": 3,
            },
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.encoder(x * 0.5 + 0.5)
        x = self.color_net(x)
        x = torch.sigmoid(x)
        x = x.reshape(*batch_shape, -1)
        return x


def refine_mesh_implicit_uv(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    neural_network: torch.nn.Module,
    map_Kd: Optional[torch.Tensor]=None, 
    weights: Optional[Union[List[float], np.ndarray]]=None,
    render_size=512, 
    texture_size=1024, 
    n_iters=100,
    learning_rate=1e-3,
    visualize=False,
    perspective=True,
):
    '''
    mesh: Mesh
    map_Kd: (H', W', 4), rgba of map_Kd
    cam2world_matrices: (M, 4, 4)
    intrinsics: (M ,3, 3)
    input_rgb: (M, 3, H, W) or (M, 4, H, W), source rgb images
    '''
    if visualize:
        os.makedirs('.cache', exist_ok=True)
    if map_Kd is None:
        # print('initial map_Kd is None, start from zeros and disable blending')
        map_Kd = torch.zeros((texture_size, texture_size, 3), dtype=c2ws.dtype, device=c2ws.device)
    if map_Kd.shape[-1] == 3:
        map_Kd = torch.cat([map_Kd, mesh.compute_uv_mask(texture_size=texture_size).to(dtype=map_Kd.dtype, device=map_Kd.device)], dim=-1)
    
    mesh_renderer = NVDiffRendererBase()
    if perspective:
        mesh_renderer.enable_perspective()
    else:
        mesh_renderer.enable_orthogonal()
    
    map_Kd_origin = map_Kd.clone()
    grid_2d = make_grid(texture_size, texture_size, device="cuda")
    if visualize:
        save_image(torch.cat([
            grid_2d[..., [0]], 
            torch.full_like(grid_2d[..., [0]], -1), 
            grid_2d[..., [1]]
        ], dim=-1).mul(0.5).add(0.5).permute(2, 0, 1), f'.cache/grid_2d.png')
    map_Kd_net = neural_network.to(device='cuda')
    opt = torch.optim.Adam([p for p in map_Kd_net.parameters()], lr=learning_rate)

    lipips_loss = LipipsLoss().to(device='cuda')
    for iter in range(n_iters):
        render_result = mesh_renderer.simple_rendering(
            mesh, None, map_Kd_net, None, 
            c2ws, intrinsics, render_size=render_size,
            render_uv=True, render_map_network=True,
        )
        pred = render_result['map_attr'].permute(0, 3, 1, 2)
        loss_1 = nn.MSELoss()(images[:, [3], :, :] * pred, images[:, [3], :, :] * images[:, :3, :, :])
        loss_2 = lipips_loss(images[:, [3], :, :] * pred, images[:, [3], :, :] * images[:, :3, :, :])
        loss = loss_1 + loss_2
        opt.zero_grad()
        loss.backward()
        opt.step()

        if visualize:
            print(iter, loss_1.item(), loss_2.item())
            save_image(torch.cat([pred, images[:, :3, :, :]], dim=-2), f'.cache/image_{iter:04d}.png')
            with torch.no_grad():
                map_Kd = map_Kd_net(grid_2d.unsqueeze(0)).squeeze(0)
                map_Kd = torch.cat([map_Kd, map_Kd_origin[..., [3]]], dim=-1)
            save_image(map_Kd.permute(2, 0, 1), f'.cache/map_Kd_{iter:04d}.png')

    with torch.no_grad():
        map_Kd = map_Kd_net(grid_2d.unsqueeze(0)).squeeze(0)
        map_Kd = torch.cat([map_Kd, map_Kd_origin[..., [3]]], dim=-1)
    if visualize:
        save_image(map_Kd.permute(2, 0, 1), '.cache/map_Kd.png')
        video_out_path = ".cache/map_Kd.mp4"
        image_paths = sorted(glob(f"./.cache/map_Kd_*.png"))
        H, W = 1024, 1024
        video = np.zeros((len(image_paths), H, W, 3), dtype=np.uint8)
        for i, image_path in enumerate(tqdm(image_paths, total=len(image_paths))):
            video[i] = cv2.putText(cv2.resize(np.flip(cv2.imread(image_path), -1), (W, H), interpolation=cv2.INTER_LINEAR_EXACT), f'step: {i:04d}', [0, 20], 0, 1, [255, 0, 255], 2)
        imageio.mimsave(video_out_path, video, fps=15)
    return map_Kd


def refine_mesh_implicit_ccm(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    neural_network: torch.nn.Module,
    map_Kd: Optional[torch.Tensor]=None,
    weights: Optional[Union[List[float], np.ndarray]]=None,
    render_size=512, 
    texture_size=1024, 
    n_iters=100,
    learning_rate=1e-3,
    visualize=False,
    perspective=True,
):
    '''
    mesh: Mesh
    map_Kd: (H', W', 4), rgba of map_Kd
    cam2world_matrices: (M, 4, 4)
    intrinsics: (M ,3, 3)
    input_rgb: (M, 3, H, W) or (M, 4, H, W), source rgb images
    '''
    if visualize:
        os.makedirs('.cache', exist_ok=True)
    if map_Kd is None:
        # print('initial map_Kd is None, start from zeros and disable blending')
        map_Kd = torch.zeros((texture_size, texture_size, 3), dtype=c2ws.dtype, device=c2ws.device)
    if map_Kd.shape[-1] == 3:
        map_Kd = torch.cat([map_Kd, mesh.compute_uv_mask(texture_size=texture_size).to(dtype=map_Kd.dtype, device=map_Kd.device)], dim=-1)

    mesh_renderer = NVDiffRendererBase()
    if perspective:
        mesh_renderer.enable_perspective()
    else:
        mesh_renderer.enable_orthogonal()
    
    map_Kd_origin = map_Kd.clone()
    grid_2d = mesh_renderer.simple_inverse_rendering(
        mesh, None, None, None,
        None, None, texture_size,
        render_world_position=True,
        enable_antialis=False,
    )['world_position'].squeeze(0)
    if visualize:
        save_image(grid_2d.mul(0.5).add(0.5).permute(2, 0, 1), f'.cache/grid_2d.png')
    voxel_net = neural_network.to(device='cuda')
    opt = torch.optim.Adam([p for p in voxel_net.parameters()], lr=learning_rate)

    lipips_loss = LipipsLoss().to(device='cuda')
    for iter in range(n_iters):
        render_result = mesh_renderer.simple_rendering(
            mesh, None, None, voxel_net, 
            c2ws, intrinsics, render_size=render_size,
            render_world_position=True, render_voxel_network=True,
        )
        pred = render_result['voxel_attr'].permute(0, 3, 1, 2)
        loss_1 = nn.MSELoss()(images[:, [3], :, :] * pred, images[:, [3], :, :] * images[:, :3, :, :])
        loss_2 = lipips_loss(images[:, [3], :, :] * pred, images[:, [3], :, :] * images[:, :3, :, :])
        loss = loss_1 + loss_2
        opt.zero_grad()
        loss.backward()
        opt.step()

        if visualize:
            print(iter, loss_1.item(), loss_2.item())
            save_image(torch.cat([pred, images[:, :3, :, :]], dim=-2), f'.cache/image_{iter:04d}.png')
            with torch.no_grad():
                map_Kd = voxel_net(grid_2d.unsqueeze(0)).squeeze(0)
                map_Kd = torch.cat([map_Kd, map_Kd_origin[..., [3]]], dim=-1)
            save_image(map_Kd.permute(2, 0, 1), f'.cache/map_Kd_{iter:04d}.png')

    with torch.no_grad():
        map_Kd = voxel_net(grid_2d.unsqueeze(0)).squeeze(0)
        map_Kd = torch.cat([map_Kd, map_Kd_origin[..., [3]]], dim=-1)
    if visualize:
        save_image(map_Kd.permute(2, 0, 1), '.cache/map_Kd.png')
        video_out_path = ".cache/map_Kd.mp4"
        image_paths = sorted(glob(f"./.cache/map_Kd_*.png"))
        H, W = 1024, 1024
        video = np.zeros((len(image_paths), H, W, 3), dtype=np.uint8)
        for i, image_path in enumerate(tqdm(image_paths, total=len(image_paths))):
            video[i] = cv2.putText(cv2.resize(np.flip(cv2.imread(image_path), -1), (W, H), interpolation=cv2.INTER_LINEAR_EXACT), f'step: {i:04d}', [0, 20], 0, 1, [255, 0, 255], 2)
        imageio.mimsave(video_out_path, video, fps=15)
    return map_Kd


