
import os
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from texturetools.mesh.structure import Texture
from texturetools.camera.generator import generate_intrinsics, generate_orbit_views_c2ws
from texturetools.io.mesh_loader import load_whole_mesh
from texturetools.renderers.nvdiffrast.renderer import NVDiffRendererBase
from texturetools.renderers.nvdiffrast.pbr import PBRModel
from texturetools.renderers.blender.blender_scripts import blender_rendering


def generate_cameras_v1(n_frames=8, perspective=True):
    c2ws = generate_orbit_views_c2ws(n_frames+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_frames]
    if perspective:
        intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    else:
        intrinsics = generate_intrinsics(0.85, 0.85, fov=False, degree=False)
    return c2ws, intrinsics


def generate_cameras_v2(n_frames=8, perspective=True):
    c2ws = generate_orbit_views_c2ws(1, radius=2.8, height=0.0, theta_0=0.0, degree=True)
    c2ws = c2ws.expand(n_frames, 4, 4)
    if perspective:
        intrinsics = []
        for fov_deg in np.linspace(0.0, 90.0, n_frames+1, endpoint=True)[1:]:
            intrinsics.append(generate_intrinsics(fov_deg, fov_deg, fov=True, degree=True))
        intrinsics = torch.stack(intrinsics, dim=0)
    else:
        intrinsics = []
        for scale in np.linspace(0.0, 2.0, n_frames+1, endpoint=True)[1:]:
            intrinsics.append(generate_intrinsics(scale, scale, fov=False, degree=False))
        intrinsics = torch.stack(intrinsics, dim=0)
    return c2ws, intrinsics


def unified_pbr_render(
    input_mesh_path:str,
    output_dir:str,
    c2ws:torch.Tensor,
    intrinsics:torch.Tensor,
    render_size:Union[int, Tuple[int]],
    perspective:bool=True,
    backend='nvdiffrast',
    env_hdr_path:Optional[str]=None,
):
    if backend == 'nvdiffrast':
        mesh = load_whole_mesh(input_mesh_path)
        texture = Texture.from_trimesh(mesh)
        texture.mesh.scale_to_bbox().apply_transform()
        texture.to(device='cuda')
        renderer = NVDiffRendererBase(device='cuda')
        if perspective:
            renderer.enable_perspective()
        else:
            renderer.enable_orthogonal()
        pbr_model = PBRModel(env_hdr_path=env_hdr_path, device='cuda')
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        map_Kd = texture.map_Kd
        map_Ks = texture.map_Ks
        map_normal = texture.map_normal
        if map_Kd is None:
            map_Kd = torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
        if map_Ks is None:
            # NOTE: [ones, roughness, metallic]
            map_Ks = torch.as_tensor([1.0, 1.0, 0.0], dtype=torch.float32, device='cuda')
        if map_normal is None:
            map_normal = torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
        render_result = renderer.simple_rendering(
            texture.mesh, None, (map_Kd, map_Ks, map_normal), None,
            c2ws, intrinsics, render_size,
            render_world_normal=True,
            render_world_position=True,
            render_v_attr=False,
            render_uv=True,
            render_map_attr=True,
        )
        alpha = render_result['alpha']
        world_position = render_result['world_position']
        world_normal = render_result['world_normal']
        image_Kd, image_Ks, image_normal = render_result['map_attr'].split([map_Kd.shape[-1], map_Ks.shape[-1], map_normal.shape[-1]], dim=-1)
        view_position = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
        diffuse, specular = pbr_model(
            view_position=view_position,
            world_position=world_position,
            world_normal=world_normal,
            map_Kd=image_Kd,
            map_Ks=image_Ks,
        )
        rgb = diffuse + specular
        rgba = torch.cat([rgb, alpha], dim=-1)
        images = rgba.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        output_dir = os.path.abspath(os.path.normpath(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(images):
            image = Image.fromarray(image)
            image.save(os.path.join(output_dir, f"{idx:04d}_rgb.png"))
    elif backend == 'blender':
        blender_rendering(
            input_mesh_path=input_mesh_path,
            output_dir=output_dir,
            c2ws=c2ws,
            intrinsics=intrinsics,
            render_size=render_size,
            perspective=perspective,
            env_hdr_path=env_hdr_path,
        )
    else:
        raise NotImplementedError(f'backend {backend} is not supported')


def batch_test(input_path, output_dir, env_hdr_path=None):
    for backend in ['blender', 'nvdiffrast']:
        postfix_1 = backend
        for perspective in [True, False]:
            postfix_2 = 'perspective' if perspective else 'orthogonal'
            for generate_cameras in [generate_cameras_v1, generate_cameras_v2]:
                postfix_3 = generate_cameras.__name__

                _output_dir = os.path.join(output_dir, '_'.join([postfix_1, postfix_2, postfix_3]))
                c2ws, intrinsics = generate_cameras(n_frames=8, perspective=perspective)
                unified_pbr_render(
                    input_mesh_path=input_path,
                    output_dir=_output_dir,
                    c2ws=c2ws,
                    intrinsics=intrinsics,
                    render_size=1024,
                    perspective=perspective,
                    backend=backend,
                    env_hdr_path=env_hdr_path,
                )


if __name__ == '__main__':
    input_path = 'gradio_examples_mesh/0941f2938b3147c7bfe16ec49d3ac500/raw_mesh.glb'
    # input_path = '/home/chenxiao/code/MVDiffusion/gradio_examples_mesh/cute_wolf/textured_mesh.glb'
    # input_path = '/home/chenxiao/下载/0211/f3f8505badb54dd492e4aa4daf750a21.glb'
    env_hdr_path = 'texturetools/renderers/envmaps/lilienstein_1k.hdr'
    output_dir = 'test_result/test_pbr_renderer/'
    batch_test(input_path, output_dir, env_hdr_path=env_hdr_path)

