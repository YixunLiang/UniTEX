'''
Mesh remapping in MVLRM/AIDoll/iris3d_cr/MVMeshRecon/MVDiffusion
* vertex color
* UV + texture
'''
from itertools import combinations
import math
import os
from typing import List, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import pymeshlab as ml
import trimesh

from ...mesh.structure import Mesh, Texture
from ...camera.conversion import project, apply_transform
from ...image.image_fusion import image_fusion
from ...render.nvdiffrast.renderer_base import NVDiffRendererBase
from .uv_dilation import uv_dilation, uv_dilation_cv
from ..stitching.mip import pull_push


def remapping_uv_texture_v2(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    map_Kd: Optional[torch.Tensor]=None,  # NOTE: ignore
    use_outpainting=True, use_cv_outpainting=False,
    render_size=512, 
    texture_size=1024, 
    angle_degree_threshold: Optional[float]=45.0,  # NOTE: 45.0 for generated result, None for simulated result
    visualize=False,
    visualize_detail=False,
    perspective=True,
):
    batch_size = c2ws.shape[0]
    map_Kd_alpha = mesh.compute_uv_mask(texture_size=texture_size).to(dtype=images.dtype, device=images.device)
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    # TODO: before using a faster interpolation algorithm, downsample images to 768*768
    if render_size > 768:
        images = torch.nn.functional.interpolate(images, (768, 768))
        render_size = 768

    # accumulate
    if visualize:
        for i in range(batch_size):
            cv2.imwrite(f'./.cache/images_{i:04d}.png', images.permute(0, 2, 3, 1)[i, :, :, [2,1,0,3]].mul(255.0).detach().cpu().numpy())
    renderer = NVDiffRendererBase()
    if perspective:
        renderer.enable_perspective()
        ray_angle_degree_threshold = angle_degree_threshold
        normal_angle_degree_threshold = None
    else:
        renderer.enable_orthogonal()
        ray_angle_degree_threshold = None
        normal_angle_degree_threshold = angle_degree_threshold
    ret = renderer.global_inverse_rendering(
        mesh, images.permute(0, 2, 3, 1),
        c2ws, intrinsics, render_size, texture_size,
        ray_angle_degree_threshold=ray_angle_degree_threshold,
        normal_angle_degree_threshold=normal_angle_degree_threshold,
    )
    map_Kd_rgb_acc = ret['map_attr']
    map_Kd_weight_acc = ret['map_alpha']
    if visualize:
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc.png', torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_weight_acc.png', torch.cat([map_Kd_weight_acc[..., [0,0,0]], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        trimesh.Trimesh(
            vertices=torch.cat([ret['samples_u'], torch.zeros_like(ret['samples_u'][:, [0]])], dim=-1).detach().cpu().numpy(),
            vertex_colors=ret['samples_v'].detach().cpu().numpy(),
        ).export(f'./.cache/samples_u_v.ply')
        trimesh.Trimesh(
            vertices=torch.cat([ret['predicts_u'], torch.zeros_like(ret['predicts_u'][:, [0]])], dim=-1).detach().cpu().numpy(),
            vertex_colors=ret['predicts_v'].detach().cpu().numpy(),
        ).export(f'./.cache/predicts_u_v.ply')
        if ret['samples_x'] is not None and ret['samples_y'] is not None:
            trimesh.Trimesh(
                vertices=ret['samples_x'][:, :3].detach().cpu().numpy(),
                vertex_colors=ret['samples_y'].detach().cpu().numpy(),
            ).export(f'./.cache/samples_x_y.ply')
        tex_to_pos = ret['tex_to_pos']
        tex_to_pos = torch.cat([tex_to_pos, torch.zeros_like(tex_to_pos[..., [0]])], dim=-1)
        tex_to_dir = ret['tex_to_pos_cam'][..., :3]
        tex_to_nrm = ret['tex_to_pos_cam'][..., 3:6]
        for i in range(batch_size):
            cv2.imwrite(f'./.cache/tex_to_pos_{i:04d}.png', tex_to_pos[i, :, :, [2,1,0]].mul(0.5).add(0.5).mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/tex_to_dir_{i:04d}.png', tex_to_dir[i, :, :, [2,1,0]].mul(0.5).add(0.5).mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/tex_to_nrm_{i:04d}.png', tex_to_nrm[i, :, :, [2,1,0]].mul(0.5).add(0.5).mul(255.0).detach().cpu().numpy())
        cv2.imwrite(f'./.cache/world_to_tex_ccm.png', ret['world_to_tex'][:, :, [2,1,0]].mul(0.5).add(0.5).mul(255.0).detach().cpu().numpy())
        cv2.imwrite(f'./.cache/world_to_tex_nrm.png', ret['world_to_tex'][:, :, [5,4,3]].mul(0.5).add(0.5).mul(255.0).detach().cpu().numpy())
        if visualize_detail:
            _intrinsics = intrinsics.expand(c2ws.shape[0], -1, -1)
            for idx in list(combinations(range(batch_size), 1)) + list(combinations(range(batch_size), 2)):
                _ret = renderer.global_inverse_rendering(
                    mesh, images[idx, ...].permute(0, 2, 3, 1),
                    c2ws[idx, ...], _intrinsics[idx, ...], render_size, texture_size,
                    ray_angle_degree_threshold=ray_angle_degree_threshold,
                    normal_angle_degree_threshold=normal_angle_degree_threshold,
                    inpainting_occlusion=False,
                )
                _map_Kd_rgb_acc = _ret['map_attr'][..., :-1]
                cv2.imwrite(
                    f'./.cache/map_Kd_rgb_acc_' + '_'.join(map(lambda i: f'{i:04d}', idx)) + '.png', 
                    torch.cat([_map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy(),
                )

    # blending
    # NOTE: DO NOT BLEND TEXTURE HERE!

    # inpainting
    # NOTE: DO NOT INPAINT TEXTURE HERE!

    # outpainting
    if use_outpainting:
        if visualize:
            map_Kd_rgb_acc_np = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
        map_Kd_mask_inpainting =(map_Kd_alpha <= 0.6)
        if use_cv_outpainting:
            map_Kd_rgb_acc[..., :3] = uv_dilation_cv(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        else:
            with torch.no_grad():
                map_Kd_rgb_outpainting, _ = pull_push(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), ~map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0))
                map_Kd_rgb_acc[..., :3] = map_Kd_rgb_outpainting.squeeze(0).permute(1, 2, 0)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_rgb_acc_outpainting.png', 
                np.concatenate([
                    map_Kd_rgb_acc_np[..., :3],
                    torch.cat([
                        map_Kd_rgb_acc[..., [2,1,0]], 
                        map_Kd_mask_inpainting.repeat_interleave(3, dim=-1),
                    ], dim=-2).clamp(0, 1).mul(255).detach().cpu().numpy(),
                ], axis=1),
            )
    
    map_Kd_rgb_acc = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1)
    return map_Kd_rgb_acc
    

def remapping_uv_texture(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    map_Kd: Optional[torch.Tensor]=None,
    weights: Optional[Union[List[float], np.ndarray]]=None,
    use_alpha=True, 
    overlay_confidence=0.2,
    use_blending=True, blending_type="soft", blending_confidence=0.2,
    use_inpainting=True, use_cv_inpainting=True, inpainting_confidence=0.2,
    use_outpainting=True, use_cv_outpainting=False,
    render_size=512, 
    texture_size=1024, 
    visualize=False,
    perspective=True,
):
    batch_size = c2ws.shape[0]
    assert weights is None or len(weights) == batch_size, \
        f'size mismatch: length of weights should be {batch_size} but {len(weights)}'
    assert blending_type in ['poisson', 'soft', 'hard'], \
        f'value error: {blending_type} is not supported'
    if map_Kd is None:
        # print('initial map_Kd is None, start from zeros and disable blending')
        map_Kd = torch.zeros((texture_size, texture_size, 3), dtype=c2ws.dtype, device=c2ws.device)
        use_blending = False
    map_Kd_alpha = mesh.compute_uv_mask(texture_size=texture_size).to(dtype=map_Kd.dtype, device=map_Kd.device)
    if weights is None:
        if batch_size == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif batch_size == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
        elif batch_size == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif batch_size == 2:
            weights = [1.0, 1.0]
        else:
            weights = [1.0] * batch_size
    weights = torch.as_tensor(weights, dtype=torch.float32, device=map_Kd.device)
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    # accumulate
    map_Kd_rgb_acc, map_Kd_weight_acc = project_uv_texture(
        mesh, c2ws, intrinsics, images, 
        map_Kd=map_Kd,
        weights=weights,
        use_alpha=use_alpha, 
        render_size=render_size, 
        texture_size=texture_size,
        visualize=visualize,
        perspective=perspective,
    )
    if visualize:
        cv2.imwrite(f'./.cache/map_Kd_init.png', torch.cat([map_Kd[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc.png', torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_weight_acc.png', torch.cat([(map_Kd_weight_acc / batch_size)[..., [0,0,0]], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc_div_weight_acc.png', torch.cat([torch.div(map_Kd_rgb_acc, map_Kd_weight_acc)[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())

    # overlay
    map_Kd_rgb_acc = torch.where(map_Kd_weight_acc > overlay_confidence, torch.div(map_Kd_rgb_acc, map_Kd_weight_acc), map_Kd_rgb_acc)
    if visualize:
        map_Kd_rgb_acc_np = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc_overlay.png', map_Kd_rgb_acc_np)
    
    # blending
    if use_blending:
        if blending_type == "poisson":
            map_Kd_rgb_acc[..., :3] = image_fusion(map_Kd_rgb_acc[..., :3].unsqueeze(0), map_Kd[..., :3].unsqueeze(0), (map_Kd_weight_acc > blending_confidence).unsqueeze(0)).squeeze(0)
        elif blending_type == "soft":
            map_Kd_rgb_acc = torch.where(map_Kd_weight_acc <= blending_confidence, (map_Kd * (blending_confidence - map_Kd_weight_acc) + map_Kd_rgb_acc) / blending_confidence, map_Kd_rgb_acc)
        elif blending_type == "hard":
            map_Kd_rgb_acc = torch.where(map_Kd_weight_acc <= blending_confidence, map_Kd, map_Kd_rgb_acc)
        else:
            raise NotImplementedError(blending_type)
        if visualize:
            map_Kd_rgb_acc_np = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
            cv2.imwrite(f'./.cache/map_Kd_rgb_acc_blending.png', map_Kd_rgb_acc_np)

    # inpainting
    if use_inpainting:
        if visualize:
            map_Kd_rgb_acc_np = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
        map_Kd_mask_inpainting = torch.logical_and(map_Kd_weight_acc <= inpainting_confidence, map_Kd_alpha > 0.6)
        if use_cv_inpainting:
            map_Kd_rgb_acc[..., :3] = uv_dilation_cv(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        else:
            with torch.no_grad():
                map_Kd_rgb_acc[..., :3] = uv_dilation(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_rgb_acc_inpainting.png', 
                np.concatenate([
                    map_Kd_rgb_acc_np[..., :3],
                    torch.cat([
                        map_Kd_rgb_acc[..., [2,1,0]], 
                        map_Kd_mask_inpainting.repeat_interleave(3, dim=-1),
                    ], dim=-2).clamp(0, 1).mul(255).detach().cpu().numpy(),
                ], axis=1),
            )
    
    # outpainting
    if use_outpainting:
        if visualize:
            map_Kd_rgb_acc_np = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
        map_Kd_mask_inpainting =(map_Kd_alpha <= 0.6)
        if use_cv_outpainting:
            map_Kd_rgb_acc[..., :3] = uv_dilation_cv(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        else:
            with torch.no_grad():
                map_Kd_rgb_outpainting, _ = pull_push(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), ~map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0))
                map_Kd_rgb_acc[..., :3] = map_Kd_rgb_outpainting.squeeze(0).permute(1, 2, 0)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_rgb_acc_outpainting.png', 
                np.concatenate([
                    map_Kd_rgb_acc_np[..., :3],
                    torch.cat([
                        map_Kd_rgb_acc[..., [2,1,0]], 
                        map_Kd_mask_inpainting.repeat_interleave(3, dim=-1),
                    ], dim=-2).clamp(0, 1).mul(255).detach().cpu().numpy(),
                ], axis=1),
            )
    
    map_Kd_rgb_acc = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd_alpha], dim=-1)
    return map_Kd_rgb_acc


def remapping_vertex_color(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    v_rgb: Optional[torch.Tensor]=None,
    weights: Optional[Union[List[float], np.ndarray]]=None,
    use_alpha=True, 
    overlay_confidence=0.2, 
    use_blending=True, blending_type="soft", blending_confidence=0.2,
    use_inpainting=True, inpainting_confidence=0.2,
    render_size=512, 
    visualize=False,
    perspective=True,
):
    batch_size = c2ws.shape[0]
    intrinsics = intrinsics.expand(batch_size, -1, -1)
    assert weights is None or len(weights) == batch_size, \
        f'size mismatch: length of weights should be {batch_size} but {len(weights)}'
    assert blending_type in ['soft', 'hard'], \
        f'value error: {blending_type} is not supported'
    if v_rgb is None:
        v_rgb = torch.zeros_like(mesh.v_pos)
        use_blending = False
    if weights is None:
        if batch_size == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif batch_size == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
        elif batch_size == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif batch_size == 2:
            weights = [1.0, 1.0]
        else:
            weights = [1.0] * batch_size
    weights = torch.as_tensor(weights, dtype=torch.float32, device=v_rgb.device)
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    # accumulate
    v_rgb_acc, v_weight_acc = project_vertex_color(
        mesh, c2ws, intrinsics, images, 
        v_rgb=v_rgb,
        weights=weights,
        use_alpha=use_alpha, 
        render_size=render_size, 
        visualize=visualize,
        perspective=perspective,
    )
    if visualize:
        Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc.obj', backend='trimesh')

    # overlay
    v_rgb_acc = torch.where(v_weight_acc > overlay_confidence, torch.div(v_rgb_acc, v_weight_acc), v_rgb_acc)
    if visualize:
        Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_overlay.obj', backend='trimesh')
    
    # blending
    if use_blending:
        if blending_type == "soft":
            v_rgb_acc = torch.where(v_weight_acc <= blending_confidence, (v_rgb * (blending_confidence - v_weight_acc) + v_rgb_acc) / blending_confidence, v_rgb_acc)
        elif blending_type == "hard":
            v_rgb_acc = torch.where(v_weight_acc <= blending_confidence, v_rgb, v_rgb_acc)
        else:
            raise NotImplementedError(blending_type)
        if visualize:
            Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_blending.obj', backend='trimesh')

    # inpainting
    if use_inpainting:
        v_idx_inpainting = torch.where(v_weight_acc[:, 0] <= inpainting_confidence)[0]
        v_rgb_acc_inpainting = inpainting_vertex_color(mesh, v_rgb_acc, v_idx_inpainting)
        if visualize:
            Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_inpainting_before.obj', backend='trimesh')
            Texture(mesh, v_rgb=v_rgb_acc_inpainting).export(f'./.cache/v_rgb_acc_inpainting_after.obj', backend='trimesh')
            Texture(mesh, v_rgb=(v_weight_acc < inpainting_confidence)[..., [0,0,0]]).export(f'./.cache/v_rgb_acc_inpainting_mask.obj', backend='trimesh')
        v_rgb_acc = v_rgb_acc_inpainting
    return v_rgb_acc


def initial_map_Kd_with_v_rgb(
    mesh: Mesh,
    v_rgb: torch.Tensor,
    texture_size=1024,
    use_pymeshlab=False,
    use_cv_outpainting=False,
    visualize=False,
    perspective=True,
):
    if visualize:
        os.makedirs('./.cache', exist_ok=True)
    map_Kd_alpha = mesh.compute_uv_mask(texture_size=texture_size).to(dtype=v_rgb.dtype, device=v_rgb.device)
    
    if not use_pymeshlab:
        renderer = NVDiffRendererBase()
        if not perspective:
            renderer.enable_orthogonal()
        ret = renderer.simple_inverse_rendering(
            mesh, v_rgb, None, None,
            None, None, texture_size, 
            render_v_attr=True,
            grid_interpolate_mode='bilinear', 
            enable_antialis=False,
        )
        map_Kd_rgb = ret['v_attr'].squeeze(0)
        
        # outpainting
        if visualize:
            map_Kd_rgb_np = torch.cat([map_Kd_rgb[..., :3], map_Kd_alpha], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy()
        map_Kd_mask_inpainting = (map_Kd_alpha <= 0.6)
        if use_cv_outpainting:
            map_Kd_rgb[..., :3] = uv_dilation_cv(map_Kd_rgb[..., :3].permute(2, 0, 1).unsqueeze(0), map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        else:
            with torch.no_grad():
                map_Kd_rgb_outpainting, _ = pull_push(map_Kd_rgb[..., :3].permute(2, 0, 1).unsqueeze(0), ~map_Kd_mask_inpainting.permute(2, 0, 1).unsqueeze(0))
                map_Kd_rgb[..., :3] = map_Kd_rgb_outpainting.squeeze(0).permute(1, 2, 0)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_from_v_rgb_outpainting.png', 
                np.concatenate([
                    map_Kd_rgb_np[..., :3],
                    torch.cat([
                        map_Kd_rgb[..., [2,1,0]], 
                        map_Kd_mask_inpainting.repeat_interleave(3, dim=-1),
                    ], dim=-2).clamp(0, 1).mul(255).detach().cpu().numpy(),
                ], axis=1),
            )
    else:
        mesh, v_rgb, _ = mesh.merge_faces(v_rgb)
        mesh_ml: ml.Mesh = ml.Mesh(
            vertex_matrix=mesh.v_pos.detach().cpu().numpy(),
            face_matrix=mesh.t_pos_idx.cpu().numpy(),
            v_color_matrix=torch.cat([v_rgb, torch.ones_like(v_rgb[..., [0]])], dim=-1).detach().cpu().numpy(),
            v_tex_coords_matrix=mesh.v_tex.detach().cpu().numpy(),
        )
        meshset_ml = ml.MeshSet()
        meshset_ml.add_mesh(mesh_ml, mesh_name='model', set_as_current=True)
        meshset_ml.apply_filter('compute_texcoord_transfer_vertex_to_wedge')
        meshset_ml.apply_filter(
            'compute_texmap_from_color', textname='material_0.png', 
            textw=texture_size, texth=texture_size,
            overwrite=False, pullpush=True,
        )
        mesh_ml = meshset_ml.current_mesh()
        map_Kd_rgb: ml.Image = mesh_ml.texture('material_0.png')
        os.makedirs('/tmp/pymeshlab', exist_ok=True)
        map_Kd_rgb.save('/tmp/pymeshlab/material_0.png')
        map_Kd_rgb = np.array(Image.open('/tmp/pymeshlab/material_0.png').convert('RGB'), dtype=np.uint8)
        map_Kd_rgb = torch.as_tensor(map_Kd_rgb, dtype=torch.float32).div(255.0).flip(-3).to(v_rgb)

    map_Kd = torch.cat([map_Kd_rgb[..., :3], map_Kd_alpha], dim=-1)
    if visualize:
        Texture(mesh, v_rgb, map_Kd).export(f'./.cache/map_Kd_from_v_rgb.obj', backend='trimesh')
        cv2.imwrite(f'./.cache/map_Kd_from_v_rgb.png', map_Kd.mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
    return map_Kd


def project_uv_texture(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    map_Kd: torch.Tensor,
    weights: torch.Tensor,
    use_alpha=True, 
    render_size=512, 
    texture_size=1024,
    visualize=False,
    perspective=True,
):
    '''
    c2ws: [B, 4, 4]
    intrinsics: [B, 3, 3]
    images: [B, 4, H, W], rgba
    map_Kd: [H, W, 4], UV map with alpha channel
    weights: [B,]

    map_Kd_rgb_acc: [H, W, 4]
    map_Kd_weight_acc: [H, W, 1]
    '''
    batch_size = c2ws.shape[0]
    renderer = NVDiffRendererBase()
    if not perspective:
        renderer.enable_orthogonal()
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    ret = renderer.simple_rendering(
        mesh, None, None, None,
        c2ws, intrinsics, render_size, 
        render_cos_ray_normal=True,
        render_camera_normal=not perspective,
    )
    images_cos = ret['cos_ray_normal'].clamp(-1.0, 0.0)
    images_arccos = ((torch.arccos(images_cos) - torch.pi / 2) / (torch.pi / 2)).clamp(0.0, 1.0)
    images_arccos_dy, images_arccos_dx = torch.gradient(images_arccos, dim=[1, 2])
    images_arccos_grad = torch.sqrt(images_arccos_dy.square() + images_arccos_dx.square())
    images_arccos_alpha = (images_arccos_grad > math.radians(10) / (math.pi / 2)).float()
    kernel_size = 2 * (render_size // 512) + 1
    if kernel_size > 1:
        images_arccos_alpha_dilate = torch.nn.functional.max_pool2d(images_arccos_alpha.permute(0, 3, 1, 2), kernel_size, 1, kernel_size // 2).permute(0, 2, 3, 1)
    else:
        images_arccos_alpha_dilate = images_arccos_alpha
    if not perspective:
        images_cos = ret['camera_normal'][:, :, :, [2]].clamp(0.0, 1.0)

    ret = renderer.simple_inverse_rendering(
        mesh, None, torch.cat([images.permute(0, 2, 3, 1), images_cos, images_arccos_alpha_dilate], dim=-1), None,
        c2ws, intrinsics, texture_size, 
        render_uv=True, render_map_attr=True, render_cos_ray_normal=True,
        grid_interpolate_mode='bilinear', enable_antialis=False
    )
    map_Kd_rgb, map_Kd_cos, map_Kd_arccos_alpha = torch.split(ret['map_attr'], [4, 1, 1], dim=-1)
    map_Kd_alpha = ret['uv_alpha']
    if not use_alpha:
        map_Kd_alpha = torch.ones_like(map_Kd_alpha)
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * map_Kd_cos.square().square() * map_Kd_alpha * (1 - map_Kd_arccos_alpha)

    if visualize:
        for i in range(batch_size):
            cv2.imwrite(f'./.cache/images_cos_{i:04d}.png', images_cos.square().square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_grad_{i:04d}.png', images_arccos_grad[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_{i:04d}.png', images_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_cos_{i:04d}.png', map_Kd_cos.square().square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_rgb_{i:04d}.png', map_Kd_rgb[i].mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_alpha_{i:04d}.png', map_Kd_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_arccos_alpha_{i:04d}.png', map_Kd_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_dilate_{i:04d}.png', images_arccos_alpha_dilate[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/weights_{i:04d}.png', weights[i].mul(255.0).detach().cpu().numpy())
    
    map_Kd_rgb_acc = ((map_Kd_alpha * map_Kd_rgb + (1 - map_Kd_alpha) * map_Kd) * weights).sum(0)
    map_Kd_weight_acc = weights.sum(0)
    return map_Kd_rgb_acc, map_Kd_weight_acc


def project_vertex_color(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    v_rgb: torch.Tensor,
    weights: torch.Tensor,
    use_alpha=True, 
    render_size=512, 
    visualize=False,
    perspective=True,
):
    '''
    c2ws: [B, 4, 4]
    intrinsics: [B, 3, 3]
    images: [B, 4, H, W], rgba
    v_rgb: [V, 3]
    weights: [B,]

    v_rgb_acc: [V, 3]
    v_weight_acc: [V, 1]
    '''
    batch_size = c2ws.shape[0]
    renderer = NVDiffRendererBase()
    if not perspective:
        renderer.enable_orthogonal()
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    v_visible_mask = renderer.get_visible_vertices(mesh, c2ws, intrinsics, render_size)
    ret = renderer.simple_rendering(
        mesh, None, None, None,
        c2ws, intrinsics, render_size, 
        render_cos_ray_normal=True,
        render_camera_normal=not perspective,
    )
    images_cos = ret['cos_ray_normal'].clamp(-1.0, 0.0)
    images_arccos = ((torch.arccos(images_cos) - torch.pi / 2) / (torch.pi / 2)).clamp(0.0, 1.0)
    images_arccos_dy, images_arccos_dx = torch.gradient(images_arccos, dim=[1, 2])
    images_arccos_grad = torch.sqrt(images_arccos_dy.square() + images_arccos_dx.square())
    images_arccos_alpha = (images_arccos_grad > math.radians(10) / (math.pi / 2)).float()
    kernel_size = 2 * (render_size // 512) + 1
    if kernel_size > 1:
        images_arccos_alpha_dilate = torch.nn.functional.max_pool2d(images_arccos_alpha.permute(0, 3, 1, 2), kernel_size, 1, kernel_size // 2).permute(0, 2, 3, 1)
    else:
        images_arccos_alpha_dilate = images_arccos_alpha
    if not perspective:
        images_cos = ret['camera_normal'][:, :, :, [2]].clamp(0.0, 1.0)

    v_rgb_acc = torch.zeros_like(v_rgb)
    v_weight_acc = torch.zeros_like(v_rgb[..., [0]])
    for b in range(batch_size):
        v_visible = mesh.v_pos[v_visible_mask[b], :]
        v_rgb_visible = v_rgb[v_visible_mask[b], :].clone()
        v_weight_visible = torch.zeros_like(v_rgb_visible[..., [0]])
        
        v_visible_homo = torch.cat([v_visible, torch.ones_like(v_visible[..., [0]])], dim=-1)
        v_visible_homo = apply_transform(v_visible_homo, [c2ws[b]])
        v_ndc, _ = project(v_visible_homo, intrinsics[b], perspective=perspective)
        v_ndc_mask_valid = torch.logical_and(v_ndc > -1, v_ndc < 1).prod(dim=-1, keepdim=False).to(dtype=torch.bool)
        v_ndc_valid = v_ndc[v_ndc_mask_valid, :]
        v_rgba_valid = torch.nn.functional.grid_sample(
            images[b].unsqueeze(0),  # [B, C, H, W]
            v_ndc_valid.unsqueeze(0).unsqueeze(-2),  # [B, V, 1, 2]
            padding_mode='reflection',
            mode='bilinear',
            align_corners=False,
        )  # [B, C, V, 1]
        v_cos_valid, v_arccos_alpha_valid = torch.split(torch.nn.functional.grid_sample(
            torch.cat([images_cos[[b]], images_arccos_alpha_dilate[[b]]], dim=-1).permute(0, 3, 1, 2),  # [B, C, H, W]
            v_ndc_valid.unsqueeze(0).unsqueeze(-2),  # [B, V, 1, 2]
            padding_mode='zeros',
            mode='bilinear',
            align_corners=False,
        ), [1, 1], dim=1)  # [B, C, V, 1]

        v_rgba_valid = v_rgba_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        v_rgb_valid, v_alpha_valid  = torch.split(v_rgba_valid, (3, 1), dim=-1)
        if not use_alpha:
            v_alpha_valid = torch.ones_like(v_alpha_valid)
        v_cos_valid = v_cos_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        v_arccos_alpha_valid = v_arccos_alpha_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        
        v_rgb_visible[v_ndc_mask_valid, :] = v_rgb_valid * v_alpha_valid + v_rgb_visible[v_ndc_mask_valid, :] * (1 - v_alpha_valid)
        v_weight_visible[v_ndc_mask_valid, :] = weights[b].unsqueeze(-1) * v_cos_valid.square().square() * (1 - v_arccos_alpha_valid)
        v_rgb_acc[v_visible_mask[b], :] = v_rgb_acc[v_visible_mask[b], :] + v_weight_visible * v_rgb_visible
        v_weight_acc[v_visible_mask[b], :] = v_weight_acc[v_visible_mask[b], :] + v_weight_visible
    
    if visualize:
        for i in range(batch_size):
            cv2.imwrite(f'./.cache/images_cos_{i:04d}.png', images_cos.square().square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_grad_{i:04d}.png', images_arccos_grad[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_{i:04d}.png', images_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_dilate_{i:04d}.png', images_arccos_alpha_dilate[i].mul(255.0).detach().cpu().numpy())
    return v_rgb_acc, v_weight_acc


def inpainting_vertex_color(mesh: Mesh, v_rgb: torch.Tensor, v_idx_inpainting: torch.Tensor, max_iters=1000):
    '''
    v_rgb: [V, 3]
    v_idx_inpainting: [V_inpainting,]
    '''
    device = v_rgb.device
    L = mesh.laplacian()  # [V, V]
    
    v_mask = torch.ones((v_rgb.shape[0], 1), dtype=torch.float32, device=device)
    v_mask[v_idx_inpainting, :] = 0
    v_mask_cnt = v_mask.sum()
    L_invalid = torch.index_select(L, 0, v_idx_inpainting)    # [V_inpainting, V]
    for _ in range(max_iters):
        v_rgb_mean = torch.matmul(L_invalid, v_rgb * v_mask)  # [V_inpainting, 3]
        v_mask_mean = torch.matmul(L_invalid, v_mask)  # [V_inpainting, 1]
        v_rgb[v_idx_inpainting, :] = torch.where(v_mask_mean > 0, v_rgb_mean / v_mask_mean, v_rgb[v_idx_inpainting, :])
        v_mask[v_idx_inpainting, :] = (v_mask_mean > 0).to(dtype=torch.float32)
        v_mask_cnt_cur = v_mask.sum()
        if v_mask_cnt_cur > v_mask_cnt:
            v_mask_cnt = v_mask_cnt_cur
        else:
            break
    return v_rgb

