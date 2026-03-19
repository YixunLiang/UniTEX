from glob import glob
import math
import os
from typing import Union
from PIL import Image
import imageio
import numpy as np
import torch
from tqdm import tqdm
import trimesh

from ..camera.generator import generate_intrinsics, generate_orbit_views_c2ws
from ..mesh.structure import Texture
from ..renderers.nvdiffrast.renderer import NVDiffRendererBase
from ..io.mesh_loader import load_whole_mesh


def collate_tensor(list_dict_tensor, keys=None, device=None):
    '''
    convert list[dict[str, tensor]] to dict[str, list[tensor]], 
    and apply concat to values
    '''
    if len(list_dict_tensor) > 0:
        if keys is None:
            keys = list_dict_tensor[0].keys()
        return {k: torch.cat([v[k] if device is None else v[k].to(device=device) for v in list_dict_tensor], dim=0) for k in keys}
    return dict()


def export_video(
    mesh_obj:Union[str, trimesh.Trimesh, Texture], 
    save_path:str, 
    n_frames=180, 
    batch_size=1, 
    orbit=False, 
    perspective=False,
):
    '''
    * mesh_obj: path, or Trimesh, or build-in Texture
    * save_path: mp4/gif means video, png/jpg/webp means grid and cover
    * orbit: export orbit video, or export orbit video and top-view video
    * perspective: popular 49.1 deg perspective camera or 0.85 orthogonal camera
    NOTE: This API is used for CHECK texture map (map_Kd) only, 
        do not use it to export ccm/normal, or ensemble it in core code.
    '''
    ext = os.path.splitext(save_path)[1]
    assert ext in ['.png', '.jpg', '.webp', '.mp4', '.gif']

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
    mesh_renderer = NVDiffRendererBase()
    if orbit:
        # NOTE: cos psi = height / radius
        c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
    else:
        c2ws = torch.cat([
            generate_orbit_views_c2ws(n_frames//3+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames//3],
            generate_orbit_views_c2ws(n_frames//3+1, radius=2.8, height=2.8 * math.cos(math.radians(60)), theta_0=0.0, degree=True)[:n_frames//3],
            generate_orbit_views_c2ws(n_frames//3+1, radius=2.8, height=2.8 * math.cos(math.radians(30)), theta_0=0.0, degree=True)[:n_frames//3],
        ], dim=0)
    if perspective:
        intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
        mesh_renderer.enable_perspective()
    else:
        intrinsics = generate_intrinsics(0.85, 0.85, fov=False, degree=False)
        mesh_renderer.enable_orthogonal()
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    # render video
    render_result = []
    with torch.no_grad():
        for idx in range(0, len(c2ws), batch_size):
            render_result.append(
                mesh_renderer.simple_rendering(
                    texture.mesh, None, texture.map_Kd, None, 
                    c2ws[idx:idx+batch_size], intrinsics, render_size=1024,
                    render_uv=True, render_map_attr=True, background=1.0,
                )
            )
    render_result = collate_tensor(render_result, device='cpu')
    video = torch.cat([render_result['map_attr'][..., :3], render_result['alpha']], dim=-1)

    # export video or image grid
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if ext in ['.mp4', '.gif']:
        imageio.mimsave(save_path, video[..., [0,1,2]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=30)
    elif ext in ['.png', '.jpg', '.webp']:
        n_cols = int(math.floor(math.sqrt(n_frames)))
        n_rows = int(math.ceil(n_frames / n_cols))
        if n_cols * n_rows > n_frames:
            video = torch.cat([video, torch.zeros((n_cols * n_rows - n_frames, *video.shape[1:]), dtype=video.dtype, device=video.device)], dim=0)
        Image.fromarray(video[..., [0,1,2,3]].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8).reshape(n_rows, n_cols, video.shape[1], video.shape[2], video.shape[3]).transpose(0, 2, 1, 3, 4).reshape(n_rows * video.shape[1], n_cols * video.shape[2], video.shape[3])).save(save_path)
    else:
        raise NotImplementedError(f'ext {ext} is not supported.')


def test():
    src = "/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/mv_4_views_xl_v2/*/textured_mesh.glb"
    root_dst = "/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/mv_4_views_xl_v2_video"

    for p in tqdm(glob(src)):
        uid = os.path.basename(os.path.dirname(p))
        p_dst = os.path.join(root_dst, uid+'.mp4')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        export_video(p, p_dst)

