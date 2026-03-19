'''
Video Exporter for Demo
'''
from copy import deepcopy
from functools import partial
from glob import glob
import inspect
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torchvision.ops import box_iou
from torchvision.utils import save_image
import cv2
import imageio
from tqdm import tqdm
from PIL import Image
import trimesh
from ..mesh.structure import Texture
from ..camera.generator import generate_canonical_views_c2ws, generate_intrinsics, generate_orbit_views_c2ws, generate_semisphere_views_c2ws, generate_box_views_c2ws
from ..mesh.structure import Texture
from ..render.nvdiffrast.renderer_base import NVDiffRendererBase
from ..render.nvdiffrast.renderer_pbr import PBRModel
from ..io.mesh_loader import load_whole_mesh
from ..utils.parse_color import parse_color
from ..camera.rotation import matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from ..image.utils import masks_to_boxes
from ..camera.conversion import inverse_transform, project, discretize


class VideoExporter:
    def __init__(self) -> None:
        self.mesh_renderer = NVDiffRendererBase(device='cuda')

    def export_video(
        self, 
        texture:Texture, pbr_model:PBRModel,
        c2ws, intrinsics, render_size: Union[int, Tuple[int]],
        key='rgb', background:Optional[Union[str, float, List[float], Tuple[float]]]=None,
        with_alpha=True, normalize=False, ndc=False,
        chunk_size=1,
    ):
        '''
        intrinsics: [1, 3, 3] or [3, 3]
        c2ws: [num_views, 4, 4]
        video: [N, H, W, 4]
        cover: [H, W, 4]
        '''
        mesh = texture.mesh
        v_rgb = texture.v_rgb
        map_Kd = texture.map_Kd
        map_Ks = texture.map_Ks
        device = 'cuda'

        if map_Kd is not None:
            if map_Ks is not None and pbr_model is not None:
                # FIXME: unified pbr renderer
                renderer = partial(
                    self.mesh_renderer.uv_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(),
                )
            else:
                renderer = partial(
                    self.mesh_renderer.uv_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(),
                )
        elif map_Kd is not None:
            renderer = partial(
                self.mesh_renderer.vertex_rendering,
                mesh=mesh,
                v_rgb=v_rgb.contiguous(),
            )
        else:
            renderer = partial(
                self.mesh_renderer.geometry_rendering,
                mesh=mesh,
            )
        valid_kwargs = {k for k in inspect.signature(renderer.func).parameters.keys() if k.startswith('render_') and k != 'render_size'}
        if f'render_{key}' in valid_kwargs:
            extra_kwargs = {f'render_{key}': True}
        else:
            extra_kwargs = dict()
        background = parse_color(background)
        if background is not None:
            background = background.to(dtype=torch.float32, device=device)
        num_views, _, _ = c2ws.shape
        c2ws = c2ws.to(dtype=torch.float32, device=device)
        intrinsics = intrinsics.expand(num_views, intrinsics.shape[-2], intrinsics.shape[-1]).to(dtype=torch.float32, device=device)
        render_size = (render_size, render_size) if isinstance(render_size, int) else render_size
        if with_alpha:
            video = torch.empty((num_views, *render_size, 4), dtype=torch.float32)
        else:
            video = torch.empty((num_views, *render_size, 3), dtype=torch.float32)
        if normalize:
            normalize_scale = None  # NOTE: (min, max) of selected pixels
        with torch.no_grad():
            for idx in range(0, c2ws.shape[0], chunk_size):
                render_result = renderer(
                    c2ws=c2ws[idx:idx+chunk_size],
                    intrinsics=intrinsics[idx:idx+chunk_size],
                    render_size=render_size,
                    **extra_kwargs,
                )
                rgb = render_result[key]
                alpha = render_result['alpha']
                if rgb.shape[-1] == 1:
                    rgb = torch.repeat_interleave(rgb, 3, dim=-1)
                elif rgb.shape[-1] == 2:
                    rgb = torch.cat([
                        rgb[..., [0]],
                        rgb[..., [1]],
                        torch.full_like(rgb[..., [0]], -1.0 if ndc else 0.0),
                    ], dim=-1)
                elif rgb.shape[-1] == 3:
                    rgb = rgb
                elif rgb.shape[-1] > 3:
                    rgb = rgb[..., :3]
                else:
                    raise ValueError(f'rgb shape error: {rgb.shape}')
                if normalize:
                    mask = torch.repeat_interleave(alpha > 0, 3, dim=-1)
                    rgb_sel = rgb[mask]
                    if normalize_scale is None:
                        normalize_scale = (rgb_sel.min(), rgb_sel.max())
                    rgb[mask] = (rgb_sel - normalize_scale[0]) / (normalize_scale[1] - normalize_scale[0])
                if ndc:
                    rgb = rgb * 0.5 + 0.5
                # NOTE: change background after normalize and ndc
                if background is not None:
                    rgb = rgb * alpha + background * (1 - alpha)
                if with_alpha:
                    video[idx: idx+chunk_size] = torch.cat([rgb, alpha], dim=-1).cpu()
                else:
                    video[idx: idx+chunk_size] = rgb.cpu()
        return video

    def export_orbit_video(
        self,
        mesh_obj:Union[str, trimesh.Trimesh, Texture],
        video_path,
        n_frames=120,
        enhance_mode=None,
        perspective=True,
        video_type='rgb',
        save_frames=False,
        save_grid=False,
        save_cover=False,
        save_camera=False,
        rename_with_euler=False,
    ):
        ext = os.path.splitext(video_path)[1]
        assert ext in ['.mp4', '.gif']
        assert video_type in [
            'rgb',
            'albedo',
            'world_normal',
            'camera_normal',
            'world_position',
            'camera_position',
            'z_depth',
            'distance',
        ]
        if video_type in ['z_depth', 'distance']:
            normalize = True
        else:
            normalize = False
        if video_type in ['world_normal', 'camera_normal', 'world_position', 'camera_position']:
            ndc = True
        else:
            ndc = False

        # load mesh
        if isinstance(mesh_obj, str):
            mesh = load_whole_mesh(mesh_obj)
            texture = Texture.from_trimesh(mesh)
            texture.mesh.scale_to_bbox().apply_transform()
        elif isinstance(mesh_obj, trimesh.Trimesh):
            texture = Texture.from_trimesh(mesh_obj)
        elif isinstance(mesh_obj, Texture):
            texture = mesh_obj
        else:
            raise NotImplementedError(f'mesh_obj {type(mesh_obj)} is not supported.')
        assert texture.map_Kd is not None, 'missing map_Kd in texture'
        texture = texture.to(device='cuda')
        texture.reset_map_Kd_mask()

        # initialize renderer
        if enhance_mode is None:
            c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
        elif enhance_mode == 'pitch':
            c2ws = torch.cat([
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-2.425, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:n_frames],
            ])
        elif enhance_mode == 'box':
            c2ws = generate_box_views_c2ws(radius=2.8)
        elif enhance_mode == 'canonical':
            c2ws = generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8))
        else:
            raise NotImplementedError(f'enhance_mode {enhance_mode} is not supported')
        if perspective:
            intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(0.85, 0.85, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        camera_info = {
            'c2ws': c2ws,
            'intrinsics': intrinsics,
            'perspective': perspective,
        }

        # render video
        video = self.export_video(
            texture, None,
            c2ws, intrinsics, render_size=1024,
            key=video_type, background='white',
            with_alpha=True, normalize=normalize, ndc=ndc,
            chunk_size=1,
        )

        # export video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=15)
        if save_frames:
            frames_path = os.path.splitext(video_path)[0] + '_frames'
            os.makedirs(frames_path, exist_ok=True)
            if rename_with_euler:
                eulers = torch.rad2deg(matrix_to_euler_angles(c2ws[:, :3, :3], convention='XYZ'))
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}-{eulers[i, 0].item():.4f}-{eulers[i, 1].item():.4f}-{eulers[i, 2].item():.4f}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
            else:
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
        if save_grid:
            grid_path = os.path.splitext(video_path)[0] + '_grid.png'
            n_cols = int(math.floor(math.sqrt(n_frames)))
            n_rows = int(math.ceil(n_frames / n_cols))
            if n_cols * n_rows > n_frames:
                video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
            cv2.imwrite(grid_path, video[..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3]))
        if save_cover:
            cover_path = os.path.splitext(video_path)[0] + '_cover.png'
            cv2.imwrite(cover_path, video[0, ..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
        if save_camera:
            camera_path = os.path.splitext(video_path)[0] + '_camera.pth'
            torch.save(camera_info, camera_path)

    def export_scene_cad_video(
        self,
        mesh_obj:Union[str, trimesh.Trimesh, Texture],
        w2c:torch.Tensor,
        fov:float,
        video_path,
        n_frames=120,
        enhance_mode=None,
        perspective=True,
        video_type='rgb',
        save_frames=False,
        save_grid=False,
        save_cover=False,
        save_camera=False,
        rename_with_euler=False,
    ):
        ext = os.path.splitext(video_path)[1]
        assert ext in ['.mp4', '.gif']
        assert video_type in [
            'rgb',
            'albedo',
            'world_normal',
            'camera_normal',
            'world_position',
            'camera_position',
            'z_depth',
            'distance',
        ]
        if video_type in ['z_depth', 'distance']:
            normalize = True
        else:
            normalize = False
        if video_type in ['world_normal', 'camera_normal', 'world_position', 'camera_position']:
            ndc = True
        else:
            ndc = False

        # load mesh
        if isinstance(mesh_obj, str):
            mesh = load_whole_mesh(mesh_obj)
            texture = Texture.from_trimesh(mesh)
        elif isinstance(mesh_obj, trimesh.Trimesh):
            texture = Texture.from_trimesh(mesh_obj)
        elif isinstance(mesh_obj, Texture):
            texture = mesh_obj
        else:
            raise NotImplementedError(f'mesh_obj {type(mesh_obj)} is not supported.')
        assert texture.map_Kd is not None, 'missing map_Kd in texture'
        texture = texture.to(device='cuda')
        texture.reset_map_Kd_mask()

        # initialize renderer
        if enhance_mode is None:
            c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
        elif enhance_mode == 'pitch':
            c2ws = torch.cat([
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-2.425, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:n_frames],
            ])
        elif enhance_mode == 'box':
            c2ws = generate_box_views_c2ws(radius=2.8)
        elif enhance_mode == 'canonical':
            c2ws = generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8))
        else:
            raise NotImplementedError(f'enhance_mode {enhance_mode} is not supported')
        if perspective:
            intrinsics = generate_intrinsics(fov, fov, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(fov, fov, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        c2ws[:, :3, 3] = 0.0
        w2c = w2c.to(dtype=torch.float32, device='cuda')
        # c2ws = w2c.expand_as(c2ws).inverse()  # for debug
        trans = torch.eye(4, dtype=torch.float32, device='cuda')
        center = texture.mesh.center()
        trans[:3, 3] = -center
        c2ws_origin = c2ws.clone()
        c2ws = torch.linalg.inv(trans) @ c2ws @ trans @ w2c.inverse()
        intrinsics = intrinsics.to(dtype=torch.float32, device='cuda')
        camera_info = {
            'w2c': w2c,
            'c2ws': c2ws,
            'rots': c2ws_origin,
            'intrinsics': intrinsics,
            'perspective': perspective,
        }

        # render video
        video = self.export_video(
            texture, None,
            c2ws, intrinsics, render_size=1024,
            key=video_type, background='white',
            with_alpha=True, normalize=normalize, ndc=ndc,
        )

        # export video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=15)

        if save_frames:
            frames_path = os.path.splitext(video_path)[0] + '_frames'
            os.makedirs(frames_path, exist_ok=True)
            if rename_with_euler:
                eulers = torch.rad2deg(matrix_to_euler_angles(c2ws[:, :3, :3], convention='XYZ'))
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}-{eulers[i, 0].item():.4f}-{eulers[i, 1].item():.4f}-{eulers[i, 2].item():.4f}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
            else:
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
        if save_grid:
            grid_path = os.path.splitext(video_path)[0] + '_grid.png'
            n_cols = int(math.floor(math.sqrt(n_frames)))
            n_rows = int(math.ceil(n_frames / n_cols))
            if n_cols * n_rows > n_frames:
                video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
            cv2.imwrite(grid_path, video[..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3]))
        if save_cover:
            cover_path = os.path.splitext(video_path)[0] + '_cover.png'
            cv2.imwrite(cover_path, video[0, ..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
        if save_camera:
            camera_path = os.path.splitext(video_path)[0] + '_camera.pth'
            torch.save(camera_info, camera_path)

    def export_scene_cad_video_with_scale(
        self,
        mesh_obj:Union[str, trimesh.Trimesh, Texture],
        w2c:torch.Tensor,
        fov:float,
        bbox:List[float],
        video_path,
        n_frames=120,
        enhance_mode=None,
        perspective=True,
        video_type='rgb',
        save_frames=False,
        save_grid=False,
        save_cover=False,
        save_camera=False,
        rename_with_euler=False,
    ):
        ext = os.path.splitext(video_path)[1]
        assert ext in ['.mp4', '.gif']
        assert video_type in [
            'rgb',
            'albedo',
            'world_normal',
            'camera_normal',
            'world_position',
            'camera_position',
            'z_depth',
            'distance',
        ]
        if video_type in ['z_depth', 'distance']:
            normalize = True
        else:
            normalize = False
        if video_type in ['world_normal', 'camera_normal', 'world_position', 'camera_position']:
            ndc = True
        else:
            ndc = False

        # load mesh
        if isinstance(mesh_obj, str):
            mesh = load_whole_mesh(mesh_obj)
            texture = Texture.from_trimesh(mesh)
        elif isinstance(mesh_obj, trimesh.Trimesh):
            texture = Texture.from_trimesh(mesh_obj)
        elif isinstance(mesh_obj, Texture):
            texture = mesh_obj
        else:
            raise NotImplementedError(f'mesh_obj {type(mesh_obj)} is not supported.')
        assert texture.map_Kd is not None, 'missing map_Kd in texture'
        texture = texture.to(device='cuda')
        texture.reset_map_Kd_mask()

        # initialize renderer
        if enhance_mode is None:
            c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
        elif enhance_mode == 'pitch':
            c2ws = torch.cat([
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-2.425, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:n_frames],
            ])
        elif enhance_mode == 'box':
            c2ws = generate_box_views_c2ws(radius=2.8)
        elif enhance_mode == 'canonical':
            c2ws = generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8))
        else:
            raise NotImplementedError(f'enhance_mode {enhance_mode} is not supported')
        if perspective:
            intrinsics = generate_intrinsics(fov, fov, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(fov, fov, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        c2ws[:, :3, 3] = 0.0
        w2c = w2c.to(dtype=torch.float32, device='cuda')
        # c2ws = w2c.expand_as(c2ws).inverse()  # NOTE: for debug
        trans = torch.eye(4, dtype=torch.float32, device='cuda')
        center = texture.mesh.center()
        trans[:3, 3] = -center
        c2ws_origin = c2ws.clone()
        # c2ws = torch.linalg.inv(trans) @ c2ws @ trans @ w2c.inverse()  # NOTE: compute later
        intrinsics = intrinsics.to(dtype=torch.float32, device='cuda')

        scale = torch.nn.Parameter(torch.ones((c2ws.shape[0],), dtype=torch.float32, device='cuda') * 10.0)
        optim = torch.optim.Adam([scale], lr=0.5)
        n_iters = 1000
        mask_orgin = torch.zeros((1024, 1024, 1), dtype=torch.float32, device='cuda')
        mask_orgin[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), :] = 1.0
        for n_iter in range(n_iters):        
            scale_matrix = torch.eye(4, dtype=torch.float32, device='cuda').repeat(c2ws.shape[0], 1, 1)
            scale_matrix[:, 0, 0] = scale
            scale_matrix[:, 1, 1] = scale
            scale_matrix[:, 2, 2] = scale

            c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scale_matrix @ trans @ w2c.inverse()
            render_result = self.mesh_renderer.simple_rendering(
                texture.mesh, None, None, None,
                c2ws=c2ws,
                intrinsics=intrinsics,
                render_size=(1024, 1024),
                enable_antialis=True,
            )
            loss = torch.nn.functional.l1_loss(render_result['alpha'], mask_orgin)
            # print(n_iter, loss.item())

            loss.backward()
            optim.step()
            optim.zero_grad()

        scales_max = scale_matrix.detach()
        c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scales_max @ trans @ w2c.inverse()
        camera_info = {
            'w2c': w2c,
            'c2ws': c2ws,
            'rots': c2ws_origin,
            'scales': scales_max,
            'intrinsics': intrinsics,
            'perspective': perspective,
        }

        # render video
        video = self.export_video(
            texture, None,
            c2ws, intrinsics, render_size=1024,
            key=video_type, background='white',
            with_alpha=True, normalize=normalize, ndc=ndc,
        )

        # export video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=15)

        if save_frames:
            frames_path = os.path.splitext(video_path)[0] + '_frames'
            os.makedirs(frames_path, exist_ok=True)
            if rename_with_euler:
                eulers = torch.rad2deg(matrix_to_euler_angles(c2ws[:, :3, :3], convention='XYZ'))
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}-{eulers[i, 0].item():.4f}-{eulers[i, 1].item():.4f}-{eulers[i, 2].item():.4f}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
            else:
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
        if save_grid:
            grid_path = os.path.splitext(video_path)[0] + '_grid.png'
            n_cols = int(math.floor(math.sqrt(n_frames)))
            n_rows = int(math.ceil(n_frames / n_cols))
            if n_cols * n_rows > n_frames:
                video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
            cv2.imwrite(grid_path, video[..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3]))
        if save_cover:
            cover_path = os.path.splitext(video_path)[0] + '_cover.png'
            cv2.imwrite(cover_path, video[0, ..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
        if save_camera:
            camera_path = os.path.splitext(video_path)[0] + '_camera.pth'
            torch.save(camera_info, camera_path)

    def export_scene_cad_video_with_scale_search(
        self,
        mesh_obj:Union[str, trimesh.Trimesh, Texture],
        w2c:torch.Tensor,
        fov:float,
        bbox:List[float],
        video_path,
        n_frames=120,
        enhance_mode=None,
        perspective=True,
        video_type='rgb',
        save_frames=False,
        save_grid=False,
        save_cover=False,
        save_camera=False,
        rename_with_euler=False,
        debug=False,
    ):
        ext = os.path.splitext(video_path)[1]
        assert ext in ['.mp4', '.gif']
        assert video_type in [
            'rgb',
            'albedo',
            'world_normal',
            'camera_normal',
            'world_position',
            'camera_position',
            'z_depth',
            'distance',
        ]
        if video_type in ['z_depth', 'distance']:
            normalize = True
        else:
            normalize = False
        if video_type in ['world_normal', 'camera_normal', 'world_position', 'camera_position']:
            ndc = True
        else:
            ndc = False

        # load mesh
        if isinstance(mesh_obj, str):
            mesh = load_whole_mesh(mesh_obj)
            texture = Texture.from_trimesh(mesh)
        elif isinstance(mesh_obj, trimesh.Trimesh):
            texture = Texture.from_trimesh(mesh_obj)
        elif isinstance(mesh_obj, Texture):
            texture = mesh_obj
        else:
            raise NotImplementedError(f'mesh_obj {type(mesh_obj)} is not supported.')
        assert texture.map_Kd is not None, 'missing map_Kd in texture'
        texture = texture.to(device='cuda')
        texture.reset_map_Kd_mask()

        # initialize renderer
        if enhance_mode is None:
            c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
        elif enhance_mode == 'pitch':
            c2ws = torch.cat([
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-2.425, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:n_frames],
            ])
        elif enhance_mode == 'box':
            c2ws = generate_box_views_c2ws(radius=2.8)
        elif enhance_mode == 'canonical':
            c2ws = generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8))
        else:
            raise NotImplementedError(f'enhance_mode {enhance_mode} is not supported')
        if perspective:
            intrinsics = generate_intrinsics(fov, fov, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(fov, fov, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        bbox = torch.as_tensor(deepcopy(bbox), dtype=torch.int64, device='cuda')
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        c2ws[:, :3, 3] = 0.0
        w2c = w2c.to(dtype=torch.float32, device='cuda')
        # c2ws = w2c.expand_as(c2ws).inverse()  # NOTE: for debug
        trans = torch.eye(4, dtype=torch.float32, device='cuda')
        center = texture.mesh.center()
        trans[:3, 3] = -center
        c2ws_origin = c2ws.clone()
        # c2ws = torch.linalg.inv(trans) @ c2ws @ trans @ w2c.inverse()  # NOTE: compute later
        intrinsics = intrinsics.to(dtype=torch.float32, device='cuda')

        batch_size = c2ws.shape[0]
        n_steps = 100
        chunk_size = 64

        ts = torch.linspace(10.0, 0.1, 100)
        scales = torch.eye(4, dtype=torch.float32, device='cuda').repeat(n_steps, batch_size, 1, 1)
        scales[:, :, 0, 0] = ts.unsqueeze(-1)
        scales[:, :, 1, 1] = ts.unsqueeze(-1)
        scales[:, :, 2, 2] = ts.unsqueeze(-1)
        c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scales @ trans @ w2c.inverse()

        c2ws = c2ws.reshape(-1, 4, 4)
        bboxes = torch.zeros((n_steps * batch_size, 4), dtype=torch.int64, device='cuda')
        if debug:
            masks = torch.zeros((n_steps * batch_size, 1024, 1024, 1), dtype=torch.float32)
        for idx in tqdm(range(0, n_steps * batch_size, chunk_size)):
            render_result = self.mesh_renderer.simple_rendering(
                texture.mesh, None, None, None,
                c2ws=c2ws[idx:idx+chunk_size],
                intrinsics=intrinsics,
                render_size=(1024, 1024),
                enable_antialis=True,
            )
            bboxes[idx:idx+chunk_size] = masks_to_boxes(render_result['alpha'].permute(0, 3, 1, 2))
            if debug:
                masks[idx:idx+chunk_size] = render_result['alpha'].cpu()
        if debug:
            # NOTE: Maximum supported image dimension is 65500 pixels
            masks = masks.reshape(n_steps, batch_size, 1024, 1024, 1)
        c2ws = c2ws.reshape(n_steps, batch_size, 4, 4)
        bboxes = bboxes.reshape(n_steps, batch_size, 4)

        bbox_center = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        bboxes_center = (bboxes[0, :, 0] + bboxes[0, :, 2]) / 2.0, (bboxes[0, :, 1] + bboxes[0, :, 3]) / 2.0
        bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes_center[0] + bbox_center[0]
        bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes_center[1] + bbox_center[1]
        bboxes[:, :, 2] = bboxes[:, :, 2] - bboxes_center[0] + bbox_center[0]
        bboxes[:, :, 3] = bboxes[:, :, 3] - bboxes_center[1] + bbox_center[1]
        ious = box_iou(bbox.unsqueeze(0), bboxes.reshape(-1, 4)).squeeze(0).reshape(n_steps, batch_size)
        ious_max = torch.max(ious, dim=0, keepdim=False).indices
        scales_max = scales.gather(0, ious_max.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 4)).reshape(batch_size, 4, 4)

        c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scales_max @ trans @ w2c.inverse()
        camera_info = {
            'w2c': w2c,
            'c2ws': c2ws,
            'rots': c2ws_origin,
            'scales': scales_max,
            'intrinsics': intrinsics,
            'perspective': perspective,
        }

        # render video
        video = self.export_video(
            texture, None,
            c2ws, intrinsics, render_size=1024,
            key=video_type, background='white',
            with_alpha=True, normalize=normalize, ndc=ndc,
        )

        # export video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=15)

        if save_frames:
            frames_path = os.path.splitext(video_path)[0] + '_frames'
            os.makedirs(frames_path, exist_ok=True)
            if rename_with_euler:
                eulers = torch.rad2deg(matrix_to_euler_angles(c2ws[:, :3, :3], convention='XYZ'))
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}-{eulers[i, 0].item():.4f}-{eulers[i, 1].item():.4f}-{eulers[i, 2].item():.4f}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
            else:
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
        if save_grid:
            grid_path = os.path.splitext(video_path)[0] + '_grid.png'
            n_cols = int(math.floor(math.sqrt(n_frames)))
            n_rows = int(math.ceil(n_frames / n_cols))
            if n_cols * n_rows > n_frames:
                video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
            cv2.imwrite(grid_path, video[..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3]))
        if save_cover:
            cover_path = os.path.splitext(video_path)[0] + '_cover.png'
            cv2.imwrite(cover_path, video[0, ..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
        if save_camera:
            camera_path = os.path.splitext(video_path)[0] + '_camera.pth'
            torch.save(camera_info, camera_path)
        if debug:
            for i in range(batch_size):
                debug_path = os.path.splitext(video_path)[0] + f'_debug_{i:04d}.jpg'
                save_image(masks[:, i, :, :, :].permute(0, 3, 1, 2), debug_path)
            np.savetxt(os.path.splitext(video_path)[0] + f'_debug.txt', ious.detach().cpu().numpy())

    def export_scene_cad_video_with_scale_search_v2(
        self,
        mesh_obj:Union[str, trimesh.Trimesh, Texture],
        w2c:torch.Tensor,
        fov:float,
        bbox:List[float],
        video_path,
        n_frames=120,
        enhance_mode=None,
        perspective=True,
        video_type='rgb',
        save_frames=False,
        save_grid=False,
        save_cover=False,
        save_camera=False,
        rename_with_euler=False,
    ):
        ext = os.path.splitext(video_path)[1]
        assert ext in ['.mp4', '.gif']
        assert video_type in [
            'rgb',
            'albedo',
            'world_normal',
            'camera_normal',
            'world_position',
            'camera_position',
            'z_depth',
            'distance',
        ]
        if video_type in ['z_depth', 'distance']:
            normalize = True
        else:
            normalize = False
        if video_type in ['world_normal', 'camera_normal', 'world_position', 'camera_position']:
            ndc = True
        else:
            ndc = False

        # load mesh
        if isinstance(mesh_obj, str):
            mesh = load_whole_mesh(mesh_obj)
            texture = Texture.from_trimesh(mesh)
        elif isinstance(mesh_obj, trimesh.Trimesh):
            texture = Texture.from_trimesh(mesh_obj)
        elif isinstance(mesh_obj, Texture):
            texture = mesh_obj
        else:
            raise NotImplementedError(f'mesh_obj {type(mesh_obj)} is not supported.')
        assert texture.map_Kd is not None, 'missing map_Kd in texture'
        texture = texture.to(device='cuda')
        texture.reset_map_Kd_mask()

        # initialize renderer
        if enhance_mode is None:
            c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
        elif enhance_mode == 'pitch':
            c2ws = torch.cat([
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-2.425, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=-1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:n_frames],
                generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:n_frames],
            ])
        elif enhance_mode == 'box':
            c2ws = generate_box_views_c2ws(radius=2.8)
        elif enhance_mode == 'canonical':
            c2ws = generate_canonical_views_c2ws(radius=2.8, steps=(8, 8, 8))
        else:
            raise NotImplementedError(f'enhance_mode {enhance_mode} is not supported')
        if perspective:
            intrinsics = generate_intrinsics(fov, fov, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(fov, fov, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        bbox = torch.as_tensor(deepcopy(bbox), dtype=torch.int64, device='cuda')
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        c2ws[:, :3, 3] = 0.0
        w2c = w2c.to(dtype=torch.float32, device='cuda')
        # c2ws = w2c.expand_as(c2ws).inverse()  # NOTE: for debug
        trans = torch.eye(4, dtype=torch.float32, device='cuda')
        center = texture.mesh.center()
        trans[:3, 3] = -center
        c2ws_origin = c2ws.clone()
        # c2ws = torch.linalg.inv(trans) @ c2ws @ trans @ w2c.inverse()  # NOTE: compute later
        intrinsics = intrinsics.to(dtype=torch.float32, device='cuda')

        batch_size = c2ws.shape[0]
        n_steps = 100
        chunk_size = 64

        ts = torch.linspace(10.0, 0.1, 100)
        scales = torch.eye(4, dtype=torch.float32, device='cuda').repeat(n_steps, batch_size, 1, 1)
        scales[:, :, 0, 0] = ts.unsqueeze(-1)
        scales[:, :, 1, 1] = ts.unsqueeze(-1)
        scales[:, :, 2, 2] = ts.unsqueeze(-1)
        c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scales @ trans @ w2c.inverse()

        c2ws = c2ws.reshape(-1, 4, 4)
        bboxes = torch.zeros((n_steps * batch_size, 4), dtype=torch.int64, device='cuda')
        v_pos_homo = torch.cat([texture.mesh.v_pos, torch.ones_like(texture.mesh.v_pos[..., [0]])], dim=-1)
        for idx in tqdm(range(0, n_steps * batch_size, chunk_size)):
            # NOTE: there is no need to render alpha here
            v_pos_pix = discretize(
                v_pos_ndc=project(
                    v_pos_homo=inverse_transform(
                        v_pos_homo=v_pos_homo,
                        transforms=[c2ws[idx:idx+chunk_size]],
                    ),
                    intrinsics=intrinsics,
                )[0], 
                H=1024, 
                W=1024,
            )
            bboxes[idx:idx+chunk_size] = torch.cat([v_pos_pix.min(dim=1).values, v_pos_pix.max(dim=1).values], dim=-1)
        c2ws = c2ws.reshape(n_steps, batch_size, 4, 4)
        bboxes = bboxes.reshape(n_steps, batch_size, 4)

        bbox_center = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        bboxes_center = (bboxes[0, :, 0] + bboxes[0, :, 2]) / 2.0, (bboxes[0, :, 1] + bboxes[0, :, 3]) / 2.0
        bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes_center[0] + bbox_center[0]
        bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes_center[1] + bbox_center[1]
        bboxes[:, :, 2] = bboxes[:, :, 2] - bboxes_center[0] + bbox_center[0]
        bboxes[:, :, 3] = bboxes[:, :, 3] - bboxes_center[1] + bbox_center[1]
        ious = box_iou(bbox.unsqueeze(0), bboxes.reshape(-1, 4)).squeeze(0).reshape(n_steps, batch_size)
        ious_max = torch.max(ious, dim=0, keepdim=False).indices
        scales_max = scales.gather(0, ious_max.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 4)).reshape(batch_size, 4, 4)

        c2ws = torch.linalg.inv(trans) @ c2ws_origin @ scales_max @ trans @ w2c.inverse()
        camera_info = {
            'w2c': w2c,
            'c2ws': c2ws,
            'rots': c2ws_origin,
            'scales': scales_max,
            'intrinsics': intrinsics,
            'perspective': perspective,
        }

        # render video
        video = self.export_video(
            texture, None,
            c2ws, intrinsics, render_size=1024,
            key=video_type, background='white',
            with_alpha=True, normalize=normalize, ndc=ndc,
        )

        # export video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=15)

        if save_frames:
            frames_path = os.path.splitext(video_path)[0] + '_frames'
            os.makedirs(frames_path, exist_ok=True)
            if rename_with_euler:
                eulers = torch.rad2deg(matrix_to_euler_angles(c2ws[:, :3, :3], convention='XYZ'))
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}-{eulers[i, 0].item():.4f}-{eulers[i, 1].item():.4f}-{eulers[i, 2].item():.4f}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
            else:
                for i, frame in enumerate(video):
                    cv2.imwrite(os.path.join(frames_path, f'{i:04d}.png'), frame.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))
        if save_grid:
            grid_path = os.path.splitext(video_path)[0] + '_grid.png'
            n_cols = int(math.floor(math.sqrt(n_frames)))
            n_rows = int(math.ceil(n_frames / n_cols))
            if n_cols * n_rows > n_frames:
                video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
            cv2.imwrite(grid_path, video[..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3]))
        if save_cover:
            cover_path = os.path.splitext(video_path)[0] + '_cover.png'
            cv2.imwrite(cover_path, video[0, ..., [2,1,0,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
        if save_camera:
            camera_path = os.path.splitext(video_path)[0] + '_camera.pth'
            torch.save(camera_info, camera_path)

    def export_condition(
        self,
        mesh_path:str,
        geometry_scale=1.0,
        n_views=4, n_rows=2, n_cols=2, H=512, W=512,
        scale=0.85, fov_deg=49.1, perspective=False, orbit=True,
        background:Optional[Union[str, float, List[float], Tuple[float]]]=None,
        return_info=False,
        return_image=True,
        return_mesh=False,
        return_camera=False,
    ) -> Dict[str, Image.Image]:
        assert n_views == (n_rows * n_cols), f'Value Error: (n_views, n_rows, n_cols)={(n_views, n_rows, n_cols)}'

        # load mesh
        mesh = load_whole_mesh(mesh_path)
        mesh = Texture.from_trimesh(mesh).mesh
        mesh = mesh.scale_to_bbox(scale=geometry_scale).apply_transform()
        mesh = mesh.to(device='cuda')

        # initialize renderer
        if orbit:
            c2ws = generate_orbit_views_c2ws(n_views+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_views]
        else:
            assert n_views in [1, 2, 4, 6], f'Value Error: n_views={n_views}'
            # NOTE: frbltd
            c2ws = generate_box_views_c2ws(radius=2.8)
            if n_views == 1:
                c2ws = c2ws[[0], :, :]
            elif n_views == 2:
                c2ws = c2ws[[0, 2], :, :]
            elif n_views == 4:
                c2ws = c2ws[[0, 1, 2, 3], :, :]
            elif n_views == 6:
                if n_rows == 2 and n_cols == 3:
                    # NOTE: frbltd -> frtbld
                    c2ws = c2ws[[0, 1, 4, 2, 3, 5], :, :]
        if perspective:
            intrinsics = generate_intrinsics(fov_deg, fov_deg, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(scale, scale, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        background = parse_color(background)
        if background is not None:
            background = background.to(dtype=torch.float32, device='cuda')
        if return_info:
            results = {
                'mesh': mesh,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'background': background,
            }
            return results

        # render image
        out = self.mesh_renderer.simple_rendering(
            mesh, None, None, None,
            c2ws, intrinsics, (H, W),
            render_world_normal=True,
            render_world_position=True,
            enable_antialis=False,
        )
        # change background to white and make grid
        alpha = out['alpha']
        ccm = out['world_position'].mul(0.5).add(0.5)
        if background is not None:
            ccm = ccm * alpha + background * (1.0 - alpha)
        normal = out['world_normal'].mul(0.5).add(0.5)
        if background is not None:
            normal = normal * alpha + background * (1.0 - alpha)
        if not return_image:
            results = {
                'alpha': alpha.detach().cpu().numpy(),
                'ccm': ccm.detach().cpu().numpy(),
                'normal': normal.detach().cpu().numpy(),
            }
            if return_mesh:
                results.update({'mesh': mesh})
            if return_camera:
                results.update({'c2ws': c2ws, 'intrinsics': intrinsics, 'perspective': perspective})
            return results
        alpha_im = alpha.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        alpha_im = Image.fromarray(alpha_im.reshape(n_rows, n_cols, H, W).transpose(0, 2, 1, 3).reshape(n_rows * H, n_cols * W), mode='L')
        ccm_im = ccm.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        ccm_im = Image.fromarray(ccm_im.reshape(n_rows, n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, 3), mode='RGB')
        normal_im = normal.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        normal_im = Image.fromarray(normal_im.reshape(n_rows, n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, 3), mode='RGB')
        results = {
            'alpha': alpha_im,
            'ccm': ccm_im,
            'normal': normal_im,
        }
        if return_mesh:
            results.update({'mesh': mesh})
        if return_camera:
            results.update({'c2ws': c2ws, 'intrinsics': intrinsics, 'perspective': perspective})
        return results

    def export_info(
        self,
        mesh_path:str,
        n_views=4, scale=0.85, fov_deg=49.1, perspective=False, orbit=True,
        background:Optional[Union[str, float, List[float], Tuple[float]]]=None,
    ) -> Dict[str, Image.Image]:
        # load mesh
        mesh = load_whole_mesh(mesh_path)
        texture = Texture.from_trimesh(mesh)
        texture.mesh.scale_to_bbox().apply_transform()
        texture = texture.to(device='cuda')

        # initialize renderer
        if orbit:
            c2ws = generate_orbit_views_c2ws(n_views+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_views]
        else:
            c2ws = generate_semisphere_views_c2ws(n_views, radius=2.8, seed=666, semi=False)
        if perspective:
            intrinsics = generate_intrinsics(fov_deg, fov_deg, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(scale, scale, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        background = parse_color(background)
        if background is not None:
            background = background.to(dtype=torch.float32, device='cuda')
        results = {
            'texture': texture,
            'c2ws': c2ws,
            'intrinsics': intrinsics,
            'background': background,
        }
        return results



################ test functions ################


def test():
    # koolai_room_893_26_v2, koolai_room_564_20_v2
    src = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_564_20_v3/craftsman_outputs/*.glb"
    root_dst = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_564_20_v3/craftsman_outputs_nvdiffrast"
    enhance_mode = 'box'
    video_exporter = VideoExporter()

    for p in tqdm(glob(src)):
        uid = os.path.splitext(os.path.basename(p))[0]

        p_dst = os.path.join(root_dst, uid, 'video_rgb.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        video_exporter.export_orbit_video(
            p, p_dst, video_type='rgb', n_frames=60, 
            save_frames=True, save_camera=True, enhance_mode=enhance_mode, rename_with_euler=False,
        )


def test_scene_cad():
    # koolai_room_893_26_v2, koolai_room_564_20_v2, koolai_room_560_52_v2, koolai_room_1574_64_v2
    src = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_560_52_v2/bbox_aligned_outputs/*.glb"
    root_dst = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_560_52_v2/bbox_aligned_outputs_nvdiffrast"
    enhance_mode = 'box'   # None or 'box'
    video_exporter = VideoExporter()

    for p in tqdm(glob(src)):
        uid = os.path.splitext(os.path.basename(p))[0]

        ## load camera for cad model
        p_c = os.path.join(os.path.dirname(os.path.dirname(p)), 'persp_instances', uid, 'params_raw.json')
        with open(p_c, 'r') as f:
            c = json.load(f)
        pitch, roll, yaw, fov = c['pitch'], c['roll'], c['yaw'], c['fov']
        # NOTE: euler in blender: pitch(up/down), roll(ccw/cw), yaw(right/left)
        euler_blender = torch.as_tensor([pitch, roll, yaw], dtype=torch.float32)
        # NOTE: euler in nvdiffrast: pitch(up/down), yaw(left/right), roll(cw/ccw)
        euler_nvdiffrast = torch.as_tensor([pitch, -yaw, -roll], dtype=torch.float32)
        # NOTE: method 1
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = euler_angles_to_matrix(torch.deg2rad(euler_nvdiffrast), convention='XYZ')
        # NOTE: method 2
        # w2c = trimesh.transformations.euler_matrix(math.radians(pitch), math.radians(-yaw), math.radians(-roll), axes='rxyz')
        # w2c = torch.as_tensor(w2c, dtype=torch.float32)

        ## render video
        p_dst = os.path.join(root_dst, uid, f'video_rgb.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        video_exporter.export_scene_cad_video(
            p, w2c, fov, p_dst, video_type='rgb', n_frames=60, 
            save_frames=True, save_camera=True, enhance_mode=enhance_mode, rename_with_euler=False,
        )

        p_dst = os.path.join(root_dst, uid, f'video_normal.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        video_exporter.export_scene_cad_video(
            p, w2c, fov, p_dst, video_type='camera_normal', n_frames=60, 
            save_frames=True, save_camera=True, enhance_mode=enhance_mode, rename_with_euler=False,
        )


def test_scene_cad_with_scale():
    # koolai_room_893_26_v2, koolai_room_564_20_v2, koolai_room_560_52_v2, koolai_room_1574_64_v2
    # src = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/bbox_aligned_outputs/*.glb"
    src = "/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/batch_testing/final_select_examples/living_rooms/easy/0005_4/bbox_aligned_outputs//*.glb"
    root_dst = os.path.join(os.path.dirname(os.path.dirname(src)), 'bbox_aligned_outputs_nvdiffrast')
    os.makedirs(root_dst, exist_ok=True)
    
    enhance_mode = 'box'   # None or 'box'
    video_exporter = VideoExporter()

    for p in tqdm(glob(src)):
        uid = os.path.splitext(os.path.basename(p))[0]

        ## load camera for cad model
        p_c = os.path.join(os.path.dirname(os.path.dirname(p)), 'persp_instances', uid, 'params_raw.json')
        with open(p_c, 'r') as f:
            c = json.load(f)
        pitch, roll, yaw, fov = c['pitch'], c['roll'], c['yaw'], c['fov']
        # NOTE: euler in blender: pitch(up/down), roll(ccw/cw), yaw(right/left)
        euler_blender = torch.as_tensor([pitch, roll, yaw], dtype=torch.float32)
        # NOTE: euler in nvdiffrast: pitch(up/down), yaw(left/right), roll(cw/ccw)
        euler_nvdiffrast = torch.as_tensor([pitch, -yaw, -roll], dtype=torch.float32)
        # NOTE: method 1
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = euler_angles_to_matrix(torch.deg2rad(euler_nvdiffrast), convention='XYZ')
        # NOTE: method 2
        # w2c = trimesh.transformations.euler_matrix(math.radians(pitch), math.radians(-yaw), math.radians(-roll), axes='rxyz')
        # w2c = torch.as_tensor(w2c, dtype=torch.float32)

        ## load bbox
        bbox_path = os.path.join(os.path.dirname(os.path.dirname(p)), 'persp_instances', uid, 'bbox_final.json')
        with open(bbox_path, 'r') as f:
            bbox = json.load(f)['boxes']

        ## render video
        p_dst = os.path.join(root_dst, uid, f'video_rgb.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        video_exporter.export_scene_cad_video_with_scale_search_v2(
            p, w2c, fov, bbox, p_dst, video_type='rgb', n_frames=60, 
            save_frames=True, save_camera=True, enhance_mode=enhance_mode, rename_with_euler=False,
        )

        p_dst = os.path.join(root_dst, uid, f'video_normal.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        video_exporter.export_scene_cad_video_with_scale_search_v2(
            p, w2c, fov, bbox, p_dst, video_type='camera_normal', n_frames=60, 
            save_frames=True, save_camera=True, enhance_mode=enhance_mode, rename_with_euler=False,
        )

