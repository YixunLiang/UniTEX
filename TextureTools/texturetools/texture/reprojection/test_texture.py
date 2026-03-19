
from glob import glob
import os
import shutil
import imageio
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from ...camera.generator import generate_intrinsics, generate_orbit_views_c2ws
from ...mesh.structure import Texture
from ...render.nvdiffrast.renderer_base import NVDiffRendererBase
from .mesh_refine import refine_mesh_uv
from .mesh_refine_implicit import TinyMLPV1, refine_mesh_implicit_uv, refine_mesh_implicit_ccm
from .mesh_remapping import remapping_uv_texture_v2, remapping_uv_texture, remapping_vertex_color, initial_map_Kd_with_v_rgb
from ...io.mesh_loader import load_whole_mesh
from ...video.export_nvdiffrast_uv_video import collate_tensor
from ...utils.empty_cache import empty_cache


def test_texture(mesh_path:str, save_dir:str, method=6, visualize=False, perspective=False):
    mesh = load_whole_mesh(mesh_path)
    texture = Texture.from_trimesh(mesh)
    if texture.map_Kd is None or texture.map_Kd.shape[0] != texture.map_Kd.shape[1] or \
        max(texture.map_Kd.shape[0], texture.map_Kd.shape[1]) > 4096 or \
        min(texture.map_Kd.shape[0], texture.map_Kd.shape[1]) < 8:
        raise ValueError('error: no texture, or texture is not square, or texture is larger than 4096*4096 or smaller than 8*8.')
    texture.mesh.scale_to_bbox().apply_transform()
    mesh_renderer = NVDiffRendererBase()

    c2ws = generate_orbit_views_c2ws(5, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:4]
    if perspective:
        intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
        mesh_renderer.enable_perspective()
    else:
        intrinsics = generate_intrinsics(0.85, 0.85, fov=False, degree=False)
        mesh_renderer.enable_orthogonal()

    texture = texture.to(device='cuda')
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')
    texture.reset_map_Kd_mask()
    render_result = mesh_renderer.simple_rendering(
        texture.mesh, None, texture.map_Kd, None, 
        c2ws, intrinsics, render_size=512,
        render_uv=True, render_map_attr=True, background=1.0,
    )
    video = torch.cat([render_result['map_attr'][..., :3], render_result['alpha']], dim=-1)
    
    if method == 0:
        map_Kd_re = remapping_uv_texture(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), map_Kd=None,
            render_size=video.shape[1], texture_size=2048,
            use_inpainting=False,
            visualize=visualize, 
            perspective=perspective,
        )
    elif method == 1:
        v_rgb = remapping_vertex_color(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), v_rgb=None, 
            render_size=video.shape[1], use_alpha=False,
            visualize=visualize,
            perspective=perspective,
        )
        map_Kd_re = initial_map_Kd_with_v_rgb(
            texture.mesh, v_rgb, 
            texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
    elif method == 2:
        v_rgb = remapping_vertex_color(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), v_rgb=None, 
            render_size=video.shape[1], use_alpha=False,
            visualize=visualize,
            perspective=perspective,
        )
        map_Kd_init = initial_map_Kd_with_v_rgb(
            texture.mesh, v_rgb, 
            texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
        map_Kd_re = remapping_uv_texture(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), map_Kd=map_Kd_init,
            render_size=video.shape[1], texture_size=2048,
            use_inpainting=False,
            visualize=visualize, 
            perspective=perspective,
        )
    elif method == 3:
        map_Kd_re = refine_mesh_uv(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), map_Kd=None,
            texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
    elif method == 4:
        map_Kd_re = refine_mesh_implicit_uv(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), TinyMLPV1(2),
            map_Kd=None,
            render_size=video.shape[1], texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
    elif method == 5:
        map_Kd_re = refine_mesh_implicit_ccm(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2), TinyMLPV1(3),
            map_Kd=None,
            render_size=video.shape[1], texture_size=2048,
            visualize=visualize,
            perspective=perspective,
        )
    elif method == 6:
        map_Kd_re = remapping_uv_texture_v2(
            texture.mesh, c2ws, intrinsics, video.permute(0, 3, 1, 2),
            map_Kd=None,
            render_size=video.shape[1], texture_size=2048,
            angle_degree_threshold=None,
            visualize=visualize,
            perspective=perspective,
        )
    else:
        raise NotImplementedError(f'method {method} is not supported.')

    c2ws = torch.cat([
        generate_orbit_views_c2ws(61, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:60],
        generate_orbit_views_c2ws(61, radius=2.8, height=1.4, theta_0=0.0, degree=True)[:60],
        generate_orbit_views_c2ws(61, radius=2.8, height=2.425, theta_0=0.0, degree=True)[:60],
    ], dim=0)
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    mesh_renderer.enable_perspective()
    texture = texture.to(device='cuda')
    map_Kd_re = map_Kd_re.to(device='cuda')
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')
    batch_size = 1
    
    render_result_raw = []
    render_result_re = []
    for idx in range(0, len(c2ws), batch_size):
        render_result_raw.append(mesh_renderer.simple_rendering(
            texture.mesh, None, texture.map_Kd, None, 
            c2ws[idx:idx+batch_size], intrinsics, render_size=1024,
            render_uv=True, render_map_attr=True, background=1.0,
        ))
        render_result_re.append(mesh_renderer.simple_rendering(
            texture.mesh, None, map_Kd_re, None, 
            c2ws[idx:idx+batch_size], intrinsics, render_size=1024,
            render_uv=True, render_map_attr=True, background=1.0,
        ))
    render_result_raw = collate_tensor(render_result_raw, device='cpu')
    render_result_re = collate_tensor(render_result_re, device='cpu')
    video_raw = render_result_raw['map_attr']
    video_re = render_result_re['map_attr']

    os.makedirs(save_dir, exist_ok=True)
    imageio.mimsave(os.path.join(save_dir, 'video_raw_re.mp4'), torch.cat([video_raw, video_re], dim=2).clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8), fps=30)
    save_image(texture.map_Kd.permute(2, 0, 1), os.path.join(save_dir, 'map_Kd_raw.png'))
    save_image(map_Kd_re.permute(2, 0, 1), os.path.join(save_dir, 'map_Kd_re.png'))



def test_texture_batch():
    mesh_path_list = glob("/home/chenxiao/下载/1114/DTC_objects_all_download_urls/*/3d-asset.glb")
    result_dir = "/home/chenxiao/下载/shared_results/outputs/DTC_objects/mv-texture-test-2"
    error_log = "/home/chenxiao/下载/shared_results/outputs/DTC_objects/mv-texture-test-2-error-log.txt"

    for mesh_path in tqdm(mesh_path_list):
        uid = os.path.basename(os.path.dirname(mesh_path))
        save_dir = os.path.join(result_dir, uid)
        os.makedirs(save_dir, exist_ok=True)
        try:
            test_texture(mesh_path, save_dir, visualize=False, perspective=False)
        except Exception as e:
            # raise e
            shutil.rmtree(save_dir)
            with open(error_log, 'a') as f:
                f.write(save_dir + ',' + e.__repr__() + '\n')
        finally:
            empty_cache()


