'''
texture api in MVLRM/AIDoll/iris3d_cr/MVMeshRecon/MVDiffusion
'''
import os
from typing import Iterable, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
import open3d as o3d
import rembg

from ...camera.conversion import c2w_to_w2c, proj_to_intr
from ...mesh.structure import Mesh, Texture
from .mesh_remapping import remapping_vertex_color, remapping_uv_texture, remapping_uv_texture_v2, initial_map_Kd_with_v_rgb
from ...video.export_nvdiffrast_video import VideoExporter
from ...utils.timer import CPUTimer


def core(texture:Texture, c2ws:torch.Tensor, intrinsics:torch.Tensor, video:torch.Tensor, weights=None, blending=False, visualize=False, perspective=True):
    if blending:
        with CPUTimer('texture: core: initialize'):
            if texture.map_Kd is None:
                v_rgb = remapping_vertex_color(
                    texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), v_rgb=None, 
                    render_size=video.shape[1], use_alpha=False,
                    weights=weights,
                    visualize=visualize,
                    perspective=perspective,
                )
                map_Kd_init = initial_map_Kd_with_v_rgb(
                    texture.mesh, v_rgb, 
                    texture_size=2048,
                    visualize=visualize,
                    perspective=perspective,
                )
            else:
                texture.reset_map_Kd_mask()
                map_Kd_init = texture.map_Kd
        with CPUTimer('texture: core: remapping'):
            map_Kd_re = remapping_uv_texture(
                texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), map_Kd=map_Kd_init,
                render_size=video.shape[1], texture_size=map_Kd_init.shape[0],
                weights=weights,
                use_inpainting=False,
                visualize=visualize,
                perspective=perspective,
            )
    else:
        map_Kd_re = remapping_uv_texture_v2(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), map_Kd=None,
            render_size=video.shape[1], texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
    texture.map_Kd = map_Kd_re
    return texture


def build_dataset(
    images: torch.Tensor,
    w2cs: torch.Tensor,
    projections: torch.Tensor,
    perspective=True,
):
    if images.shape[-1] != 4:
        rembg_session = rembg.new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        images_np = images.clamp(0, 1).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        _images = torch.empty((*images.shape[:-1], 4), dtype=images.dtype)
        for i, im in enumerate(images_np):
            im = rembg.remove(im, alpha_matting=True, session=rembg_session)
            _im = cv2.inpaint(im[:, :, :3], ~im[:, :, [3]], 3.0, cv2.INPAINT_TELEA)
            im = np.concatenate([_im, im[:, :, [3]]], axis=-1)
            _images[i] = torch.as_tensor(im, dtype=images.dtype).div(255.0)
        images = _images.to(device=images.device)
    c2ws = c2w_to_w2c(w2cs)
    intrinsics = proj_to_intr(projections.expand(c2ws.shape[0], 4, 4), perspective=perspective)
    
    dataset_dict = dict()
    dataset_dict['c2ws'] = c2ws
    dataset_dict['intrinsics'] = intrinsics
    dataset_dict['images'] = images
    dataset_dict = {k: v.to(device='cuda') if isinstance(v, torch.Tensor) else v for k, v in dataset_dict.items()}
    return dataset_dict


def build_dataset_v2(
    images: torch.Tensor,
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
):
    if images.shape[-1] != 4:
        rembg_session = rembg.new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        images_np = images.clamp(0, 1).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        _images = torch.empty((*images.shape[:-1], 4), dtype=images.dtype)
        for i, im in enumerate(images_np):
            im = rembg.remove(im, alpha_matting=True, session=rembg_session)
            _im = cv2.inpaint(im[:, :, :3], ~im[:, :, [3]], 3.0, cv2.INPAINT_TELEA)
            im = np.concatenate([_im, im[:, :, [3]]], axis=-1)
            _images[i] = torch.as_tensor(im, dtype=images.dtype).div(255.0)
        images = _images.to(device=images.device)
    intrinsics = intrinsics.expand(c2ws.shape[0], 3, 3)
    
    dataset_dict = dict()
    dataset_dict['c2ws'] = c2ws
    dataset_dict['intrinsics'] = intrinsics
    dataset_dict['images'] = images
    dataset_dict = {k: v.to(device='cuda') if isinstance(v, torch.Tensor) else v for k, v in dataset_dict.items()}
    return dataset_dict


def process_mesh(
    mesh: Mesh,
) -> Mesh:
    mesh = mesh.to_open3d()
    with CPUTimer('texture: process_mesh: remove_non_manifold_edges'):
        mesh = mesh.remove_non_manifold_edges()
    with CPUTimer('texture: process_mesh: remove_degenerate_triangles'):
        mesh = mesh.remove_degenerate_triangles()
    with CPUTimer('texture: process_mesh: remove_unreferenced_vertices'):
        mesh = mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 200_000:
        with CPUTimer('texture: process_mesh: simplify_quadric_decimation'):
            # mesh = mesh.simplify_quadric_decimation(200_000)
            device_o3d = o3d.core.Device('CPU:0')
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=device_o3d)
            target_reduction = 1 - 200_000 / len(mesh.triangle.indices)
            mesh = mesh.simplify_quadric_decimation(target_reduction)
            mesh = mesh.to_legacy()
            mesh = mesh.remove_non_manifold_edges()
            mesh = mesh.remove_degenerate_triangles()
            mesh = mesh.remove_unreferenced_vertices()
    mesh = Mesh.from_open3d(mesh)
    with CPUTimer('texture: process_mesh: unwrap_uv'):
        mesh.unwrap_uv()  # TODO: unwarp uv is necessary?
    return mesh


def opt_warpper(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    images: torch.Tensor,
    w2cs: torch.Tensor,
    projections: torch.Tensor,
    weights=None,
    mesh_path:Optional[str]=None,
    video_path:Optional[str]=None,
    visualize=False,
    perspective=True,
    blending=False,
):
    '''
    vertices: [V, 3]
    faces: [F, 3]
    images: [N, H, W, C]
    c2ws: [N, 4, 4]
    projections: [N, 4, 4] or [4, 4]
    intrinsics: [N, 3, 3] or [3, 3], normalized
    '''
    with CPUTimer('texture: build_dataset'):
        dataset_dict = build_dataset(
            images=images,
            w2cs=w2cs,
            projections=projections,
            perspective=perspective,
        )

    with CPUTimer('texture: process_mesh'):
        mesh = Mesh(vertices, faces)
        mesh = process_mesh(mesh)
        texture = Texture(mesh)
        texture = texture.to(device='cuda')
    
    intrinsics = dataset_dict['intrinsics']
    c2ws = dataset_dict['c2ws']
    images = dataset_dict['images']
    with CPUTimer('texture: core'):
        texture = core(
            texture, c2ws, intrinsics, images,
            weights=weights,
            visualize=visualize,
            perspective=perspective,
            blending=blending,
        )
    
    with CPUTimer('texture: export_mesh_and_render_video'):
        if mesh_path is not None:
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            texture.export(mesh_path)
        if video_path is not None:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_exportter = VideoExporter()
            video_exportter.export_orbit_video(texture, video_path)
    return texture


def opt_warpper_v2(
    texture,
    images: torch.Tensor,
    w2cs: torch.Tensor,
    projections: torch.Tensor,
    weights=None,
    mesh_path:Optional[str]=None,
    video_path:Optional[str]=None,
    visualize=False,
    perspective=True,
    blending=False,
):
    '''
    vertices: [V, 3]
    faces: [F, 3]
    images: [N, H, W, C]
    c2ws: [N, 4, 4]
    projections: [N, 4, 4] or [4, 4]
    intrinsics: [N, 3, 3] or [3, 3], normalized
    '''
    with CPUTimer('texture: build_dataset'):
        dataset_dict = build_dataset(
            images=images,
            w2cs=w2cs,
            projections=projections,
            perspective=perspective,
        )
    
    with CPUTimer('texture: process_mesh'):
        texture = texture.to(device='cuda')
    
    intrinsics = dataset_dict['intrinsics']
    c2ws = dataset_dict['c2ws']
    images = dataset_dict['images']
    with CPUTimer('texture: core'):
        texture = core(
            texture, c2ws, intrinsics, images,
            weights=weights,
            visualize=visualize,
            perspective=perspective,
            blending=blending,
        )
    
    with CPUTimer('texture: export_mesh_and_render_video'):
        if mesh_path is not None:
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            texture.export(mesh_path)
        if video_path is not None:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_exportter = VideoExporter()
            video_exportter.export_orbit_video(texture, video_path)
    return texture


def opt_warpper_v3(
    mesh,
    images: torch.Tensor,
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    weights=None,
    mesh_path:Optional[str]=None,
    video_path:Optional[str]=None,
    visualize=False,
    perspective=True,
    blending=False,
):
    '''
    vertices: [V, 3]
    faces: [F, 3]
    images: [N, H, W, C]
    c2ws: [N, 4, 4]
    projections: [N, 4, 4] or [4, 4]
    intrinsics: [N, 3, 3] or [3, 3], normalized
    '''
    with CPUTimer('texture: build_dataset_v2'):
        dataset_dict = build_dataset_v2(
            images=images,
            c2ws=c2ws,
            intrinsics=intrinsics,
        )

    with CPUTimer('texture: process_mesh'):
        mesh = process_mesh(mesh)
        texture = Texture(mesh)
        texture = texture.to(device='cuda')
    
    intrinsics = dataset_dict['intrinsics']
    c2ws = dataset_dict['c2ws']
    images = dataset_dict['images']
    with CPUTimer('texture: core'):
        texture = core(
            texture, c2ws, intrinsics, images,
            weights=weights,
            visualize=visualize,
            perspective=perspective,
            blending=blending,
        )
    
    with CPUTimer('texture: export_mesh_and_render_video'):
        if mesh_path is not None:
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            texture.export(mesh_path)
        if video_path is not None:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_exportter = VideoExporter()
            video_exportter.export_orbit_video(texture, video_path)
    return texture

