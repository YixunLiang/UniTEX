'''
Base NVDiffrast Renderer
'''
import math
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import (
    NearestNDInterpolator,
    LinearNDInterpolator,
    CloughTocher2DInterpolator, 
)
from scipy.spatial import KDTree
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from ...camera.conversion import (
    intr_to_proj,
    c2w_to_w2c,
    discretize,
    undiscretize,
)
from ...geometry.triangle_topology.topology import erode_face

def draw_mask(v_pos_pix:torch.Tensor, H:int, W:int):
    pre_shape = v_pos_pix.shape[:-2]
    image_mask = torch.zeros((*pre_shape, H, W, 1), dtype=torch.bool, device=v_pos_pix.device)
    ui, vi = v_pos_pix.unbind(-1)
    points_mask = torch.logical_and(torch.logical_and(ui >= 0, ui <= W-1), torch.logical_and(vi >= 0, vi <= H-1))
    idx_spatial = vi * W + ui
    idx_spatial = torch.where(points_mask, idx_spatial, 0).reshape(*pre_shape, -1)
    image_mask = torch.scatter(image_mask.reshape(*pre_shape, -1, 1), -2, idx_spatial.unsqueeze(-1), 1).reshape(*pre_shape, H, W, 1)
    return image_mask


class NVDiffRendererBase(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.enable_nvdiffrast_cuda_ctx()
        self.enable_perspective()
        self.erode_neighbor = 0

    def enable_orthogonal(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=False)
        self.perspective = False

    def enable_perspective(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=True)
        self.perspective = True

    def enable_projection(self):
        self.intr_to_proj = lambda proj: proj
        self.perspective = None

    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def enable_nvdiffrast_opengl_ctx(self):
        self.ctx = dr.RasterizeGLContext(device=self.device)

    def get_visible_faces(self, mesh, c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]]):
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        device = c2ws.device

        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        w2cs_mtx = c2w_to_w2c(c2ws)
        proj_mtx = self.intr_to_proj(intrinsics)
        mvp_mtx = torch.matmul(proj_mtx, w2cs_mtx)
        v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (height, width))
        t_idx_visible_dummy = torch.as_tensor(rast[..., [3]], dtype=torch.int64).reshape(batch_size, -1)
        t_mask_visible_dummy = torch.zeros((batch_size, t_pos_idx.shape[0] + 1), dtype=torch.bool, device=device)
        t_mask_visible_dummy = torch.scatter(t_mask_visible_dummy, -1, t_idx_visible_dummy, 1)
        t_mask_visible = t_mask_visible_dummy[:, 1:]
        if self.erode_neighbor > 0:
            for b in range(batch_size):
                t_mask_visible[b] = erode_face(mesh.t_pos_idx, t_mask_visible[b], mesh.v_pos.shape[0], self.erode_neighbor)
        return t_mask_visible

    def get_visible_vertices(self, mesh, c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]]):
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        device = c2ws.device

        t_pos_idx = mesh.t_pos_idx
        t_mask_visible = self.get_visible_faces(mesh, c2ws, intrinsics, (height, width))
        v_pos_mask_visible = torch.zeros((batch_size, mesh.v_pos.shape[0]), dtype=torch.bool, device=device)
        for b in range(batch_size):
            t_pos_idx_visible = torch.masked_select(t_pos_idx, t_mask_visible[b].unsqueeze(-1))
            t_pos_idx_visible_unique = torch.unique(t_pos_idx_visible, return_inverse=False, return_counts=False)
            v_pos_mask_visible[b] = torch.index_fill(v_pos_mask_visible[b], -1, t_pos_idx_visible_unique, 1)
        return v_pos_mask_visible

    def simple_rendering(
        self, mesh, v_attr:Tensor, map_attr:Union[Tensor, Tuple[Tensor], Callable], voxel_attr:Union[Tensor, Tuple[Tensor], Callable], 
        c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]], 
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_voxel_attr=False,
        render_voxel_network=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        render_v_attr=False,
        render_uv=False,
        render_map_attr=False,
        render_map_network=False,
        background=None,
        grid_interpolate_mode="bilinear",
        enable_antialis=True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        '''
        v_attr: [V, C] or [M, V, C], vertex attribute of mesh
        map_attr: [H, W, C] or [M, H, W, C], UV map attribute of mesh
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        background: [M, H, W, C] or [H, W, C] or [C,] or float in range(0, 1) or None
        '''
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size

        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        w2cs_mtx = c2w_to_w2c(c2ws)
        proj_mtx = self.intr_to_proj(intrinsics)
        mvp_mtx = torch.matmul(proj_mtx, w2cs_mtx)
        v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (height, width))
        mask = rast[..., [3]] > 0
        if enable_antialis:
            alpha = dr.antialias(mask.float(), rast, v_pos_clip, t_pos_idx)
        else:
            alpha = mask.float()
        out = {"mask": mask, "alpha": alpha}

        if render_all_point_cloud or render_visible_point_cloud:
            v_pos_ndc = v_pos_clip[:, :, :2] / v_pos_clip[:, :, [3]]
            v_pos_pix = discretize(v_pos_ndc, height, width)
            if render_all_point_cloud:
                all_point_cloud = draw_mask(v_pos_pix, height, width)
                out.update({"all_point_cloud": all_point_cloud})
            if render_visible_point_cloud:  # FIXME: batch version
                t_pos_visible = torch.masked_select(rast[0, :, :, 3].to(dtype=torch.int64), mask[0, :, :, 0]).sub(1)
                t_pos_visible_unique = torch.unique(t_pos_visible, return_inverse=False, return_counts=False)
                v_pos_idx_visible = torch.index_select(t_pos_idx, 0, t_pos_visible_unique).reshape(-1)
                v_pos_idx_visible_unique = torch.unique(v_pos_idx_visible, return_inverse=False, return_counts=False)
                v_pos_pix_visible = v_pos_pix[0, v_pos_idx_visible_unique, :]
                visible_point_cloud = draw_mask(v_pos_pix_visible, height, width).unsqueeze(0)
                out.update({"visible_point_cloud": visible_point_cloud})

        if render_z_depth:
            z_depth, _ = dr.interpolate(v_pos_clip[:, :, [3]].contiguous(), rast, t_pos_idx)
            z_depth = torch.lerp(torch.zeros_like(z_depth), z_depth, alpha)
            if enable_antialis:
                z_depth = dr.antialias(z_depth, rast, v_pos_clip, t_pos_idx)
            out.update({"z_depth": z_depth})

        if render_world_normal:
            world_normal, _ = dr.interpolate(mesh.v_nrm.contiguous(), rast, t_pos_idx)
            world_normal = F.normalize(world_normal, dim=-1)
            world_normal = torch.lerp(torch.full_like(world_normal, fill_value=-1.0), world_normal, alpha)
            if enable_antialis:
                world_normal = dr.antialias(world_normal, rast, v_pos_clip, t_pos_idx)
            out.update({"world_normal": world_normal})

        if render_camera_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm.contiguous(), c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            camera_normal = torch.lerp(torch.full_like(camera_normal, fill_value=-1.0), camera_normal, alpha)
            if enable_antialis:
                camera_normal = dr.antialias(camera_normal, rast, v_pos_clip, t_pos_idx)
            out.update({"camera_normal": camera_normal})

        if render_world_position:
            gb_ccm, _ = dr.interpolate(mesh.v_pos, rast, t_pos_idx)
            gb_bg = torch.full_like(gb_ccm, fill_value=-1.0)
            if enable_antialis:
                gb_ccm_aa = torch.lerp(gb_bg, gb_ccm, alpha)
                gb_ccm_aa = dr.antialias(gb_ccm_aa, rast, v_pos_clip, t_pos_idx)
                gb_ccm = torch.lerp(gb_bg, gb_ccm, mask.float())
                out.update({"world_position": gb_ccm_aa})
            else:
                gb_ccm = torch.lerp(gb_bg, gb_ccm, mask.float())
                out.update({"world_position": gb_ccm})
            
            if render_voxel_attr:
                voxel_attr = voxel_attr.expand(batch_size, *voxel_attr.shape[-4:])
                gb_voxel_attr = torch.nn.functional.grid_sample(voxel_attr.permute(0, 3, 1, 2).contiguous(), gb_ccm.unsqueeze(-4), mode=grid_interpolate_mode, align_corners=False).squeeze(-3).permute(0, 2, 3, 1)
                if isinstance(voxel_attr, torch.Tensor):
                    voxel_attr = voxel_attr.expand(batch_size, *voxel_attr.shape[-4:])
                    gb_voxel_attr = torch.nn.functional.grid_sample(voxel_attr.permute(0, 3, 1, 2).contiguous(), gb_ccm.unsqueeze(-4), mode=grid_interpolate_mode, align_corners=False).squeeze(-3).permute(0, 2, 3, 1)
                elif isinstance(voxel_attr, Tuple):
                    gb_voxel_attr = []
                    for v in voxel_attr:
                        v = v.expand(batch_size, *v.shape[-4:])
                        gb_v = torch.nn.functional.grid_sample(v.permute(0, 3, 1, 2).contiguous(), gb_ccm.unsqueeze(-4), mode=grid_interpolate_mode, align_corners=False).squeeze(-3).permute(0, 2, 3, 1)
                        gb_voxel_attr.append(gb_v)
                    gb_voxel_attr = torch.cat(gb_voxel_attr, dim=-1)
                else:
                    raise NotImplementedError(f'gb_voxel_attr type {type(gb_voxel_attr)} is not supported')
                if background is not None:
                    if isinstance(background, float):
                        gb_voxel_attr = torch.lerp(torch.full_like(gb_voxel_attr, fill_value=background), gb_voxel_attr, alpha)
                    elif isinstance(background, torch.Tensor):
                        gb_voxel_attr = torch.lerp(background.to(gb_voxel_attr).expand_as(gb_voxel_attr), gb_voxel_attr, alpha)
                    else:
                        raise NotImplementedError
                if enable_antialis:
                    gb_voxel_attr = dr.antialias(gb_voxel_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                out.update({"gb_voxel_attr": gb_voxel_attr})
            elif render_voxel_network:
                assert isinstance(voxel_attr, Callable), 'voxel_attr is not callable'
                gb_voxel_attr = voxel_attr(gb_ccm)
                if background is not None:
                    if isinstance(background, float):
                        gb_voxel_attr = torch.lerp(torch.full_like(gb_voxel_attr, fill_value=background), gb_voxel_attr, alpha)
                    elif isinstance(background, torch.Tensor):
                        gb_voxel_attr = torch.lerp(background.to(gb_voxel_attr).expand_as(gb_voxel_attr), gb_voxel_attr, alpha)
                    else:
                        raise NotImplementedError
                if enable_antialis:
                    gb_voxel_attr = dr.antialias(gb_voxel_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                out.update({"voxel_attr": gb_voxel_attr})

        if render_camera_position or render_distance or render_ray_direction:
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast, t_pos_idx)
            if render_camera_position:
                camera_position = torch.lerp(torch.full_like(camera_position, fill_value=0.0), camera_position, alpha)
                if enable_antialis:
                    camera_position = dr.antialias(camera_position, rast, v_pos_clip, t_pos_idx)
                out.update({"camera_position": camera_position})
            if render_distance:
                distance = torch.norm(camera_position, p=2, dim=-1, keepdim=True)
                distance = torch.lerp(torch.full_like(distance, fill_value=0.0), distance, alpha)
                if enable_antialis:
                    distance = dr.antialias(distance, rast, v_pos_clip, t_pos_idx)
                out.update({"distance": distance})
            if render_ray_direction:
                ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
                ray_direction = torch.lerp(torch.full_like(ray_direction, fill_value=-1.0), ray_direction, alpha)
                if enable_antialis:
                    ray_direction = dr.antialias(ray_direction, rast, v_pos_clip, t_pos_idx)
                out.update({"ray_direction": ray_direction})

        if render_cos_ray_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast, t_pos_idx)
            ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
            cos_ray_normal = torch.sum(camera_normal * ray_direction, dim=-1, keepdim=True)
            cos_ray_normal = torch.lerp(torch.full_like(cos_ray_normal, fill_value=-1.0), cos_ray_normal, alpha)
            if enable_antialis:
                cos_ray_normal = dr.antialias(cos_ray_normal, rast, v_pos_clip, t_pos_idx)
            out.update({"cos_ray_normal": cos_ray_normal})

        if render_v_attr:
            gb_v_attr, _ = dr.interpolate(v_attr, rast, t_pos_idx)
            if background is not None:
                if isinstance(background, float):
                    gb_v_attr = torch.lerp(torch.full_like(gb_v_attr, fill_value=background), gb_v_attr, alpha)
                elif isinstance(background, torch.Tensor):
                    gb_v_attr = torch.lerp(background.to(gb_v_attr).expand_as(gb_v_attr), gb_v_attr, alpha)
                else:
                    raise NotImplementedError
            if enable_antialis:
                gb_v_attr = dr.antialias(gb_v_attr, rast, v_pos_clip, t_pos_idx)
            out.update({"v_attr": gb_v_attr})

        if render_uv:
            v_tex_ndc = mesh.v_tex * 2.0 - 1.0
            t_tex_idx = mesh.t_tex_idx.to(torch.int32)
            # NOTE: rast is defined on faces, 
            # so there is no need merge/unmerge face and update rast.
            gb_uv, _ = dr.interpolate(v_tex_ndc, rast, t_tex_idx)
            gb_bg = torch.full_like(gb_uv, fill_value=-1.0)
            if enable_antialis:
                gb_uv_aa = torch.lerp(gb_bg, gb_uv, alpha)
                gb_uv_aa = dr.antialias(gb_uv_aa, rast, v_pos_clip, t_pos_idx)
                gb_uv = torch.lerp(gb_bg, gb_uv, mask.float())
                out.update({"uv": gb_uv_aa})
            else:
                gb_uv = torch.lerp(gb_bg, gb_uv, mask.float())
                out.update({"uv": gb_uv})

            if render_map_attr:
                if isinstance(map_attr, torch.Tensor):
                    map_attr = map_attr.expand(batch_size, *map_attr.shape[-3:])
                    if grid_interpolate_mode == 'nvdiffrast':
                        gb_map_attr = dr.texture(map_attr.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
                    else:
                        gb_map_attr = torch.nn.functional.grid_sample(map_attr.permute(0, 3, 1, 2).contiguous(), gb_uv, mode=grid_interpolate_mode, align_corners=False).permute(0, 2, 3, 1)
                elif isinstance(map_attr, Tuple):
                    gb_map_attr = []
                    if grid_interpolate_mode == 'nvdiffrast':
                        for m in map_attr:
                            m = m.expand(batch_size, *m.shape[-3:])
                            gb_m = dr.texture(m.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
                            gb_map_attr.append(gb_m)
                    else:
                        for m in map_attr:
                            m = m.expand(batch_size, *m.shape[-3:])
                            gb_m = torch.nn.functional.grid_sample(m.permute(0, 3, 1, 2).contiguous(), gb_uv, mode=grid_interpolate_mode, align_corners=False).permute(0, 2, 3, 1)
                            gb_map_attr.append(gb_m)
                    gb_map_attr = torch.cat(gb_map_attr, dim=-1)
                else:
                    raise NotImplementedError(f'map_attr type {type(map_attr)} is not supported')
                if background is not None:
                    if isinstance(background, float):
                        gb_map_attr = torch.lerp(torch.full_like(gb_map_attr, fill_value=background), gb_map_attr, alpha)
                    elif isinstance(background, torch.Tensor):
                        gb_map_attr = torch.lerp(background.to(gb_map_attr).expand_as(gb_map_attr), gb_map_attr, alpha)
                    else:
                        raise NotImplementedError
                if enable_antialis:
                    gb_map_attr = dr.antialias(gb_map_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                out.update({"map_attr": gb_map_attr})
            elif render_map_network:
                assert isinstance(map_attr, Callable), 'map_attr is not callable'
                gb_map_attr = map_attr(gb_uv)
                if background is not None:
                    if isinstance(background, float):
                        gb_map_attr = torch.lerp(torch.full_like(gb_map_attr, fill_value=background), gb_map_attr, alpha)
                    elif isinstance(background, torch.Tensor):
                        gb_map_attr = torch.lerp(background.to(gb_map_attr).expand_as(gb_map_attr), gb_map_attr, alpha)
                    else:
                        raise NotImplementedError
                if enable_antialis:
                    gb_map_attr = dr.antialias(gb_map_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                out.update({"map_attr": gb_map_attr})
        return out

    def simple_inverse_rendering(
        self, mesh, 
        v_attr:Tensor, map_attr:Tensor, voxel_attr:Tensor,
        c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]], 
        render_all_point_cloud=False,  # prepare for multi texture map
        render_visible_point_cloud=False,  # prepare for multi texture map
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_voxel_attr=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        render_v_attr=False,
        render_uv=False,
        render_map_attr=False,
        background=None,
        grid_interpolate_mode="bilinear",
        enable_antialis=True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        
        v_tex_ndc = mesh.v_tex * 2.0 - 1.0
        v_tex_clip = torch.cat([v_tex_ndc, torch.zeros_like(v_tex_ndc[:, [0]]), torch.ones_like(v_tex_ndc[:, [0]])], dim=-1).unsqueeze(0)
        t_tex_idx = mesh.t_tex_idx.to(dtype=torch.int32)
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

        # NOTE: rast is defined on faces, 
        # so there is no need merge/unmerge face and update rast.
        rast, _ = dr.rasterize(self.ctx, v_tex_clip, t_tex_idx, (height, width))
        mask = rast[..., [3]] > 0
        if enable_antialis:
            alpha = dr.antialias(mask.float(), rast, v_tex_clip, t_tex_idx)
        else:
            alpha = mask.float()
        out = {"mask": mask, "alpha": alpha}

        if c2ws is not None:
            batch_size = c2ws.shape[0]
            rast_duplicated = rast.tile(batch_size, 1, 1, 1)
            v_tex_clip_duplicated = v_tex_clip.tile(batch_size, 1, 1)
        else:
            batch_size = None
            rast_duplicated = None
            v_tex_clip_duplicated = None

        if render_world_normal:
            world_normal, _ = dr.interpolate(mesh.v_nrm, rast, t_pos_idx)
            world_normal = F.normalize(world_normal, dim=-1)
            world_normal = torch.lerp(torch.full_like(world_normal, fill_value=-1.0), world_normal, alpha)
            if enable_antialis:
                world_normal = dr.antialias(world_normal, rast, v_tex_clip, t_tex_idx)
            out.update({"world_normal": world_normal})

        if render_camera_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = F.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast_duplicated, t_pos_idx)
            camera_normal = F.normalize(camera_normal, dim=-1)
            camera_normal = torch.lerp(torch.full_like(camera_normal, fill_value=-1.0), camera_normal, alpha)
            if enable_antialis:
                camera_normal = dr.antialias(camera_normal, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
            out.update({"camera_normal": camera_normal})

        if render_world_position:
            gb_ccm, _ = dr.interpolate(mesh.v_pos, rast, t_pos_idx)
            gb_bg = torch.full_like(gb_ccm, fill_value=-1.0)
            if enable_antialis:
                gb_ccm_aa = torch.lerp(gb_bg, gb_ccm, alpha)
                gb_ccm_aa = dr.antialias(gb_ccm_aa, rast, v_tex_clip, t_tex_idx)
                gb_ccm = torch.lerp(gb_bg, gb_ccm, mask.float())
                out.update({"world_position": gb_ccm_aa})
            else:
                gb_ccm = torch.lerp(gb_bg, gb_ccm, mask.float())
                out.update({"world_position": gb_ccm})

            if render_voxel_attr:
                voxel_attr = voxel_attr.expand(batch_size, *voxel_attr.shape[-4:])
                gb_voxel_attr = torch.nn.functional.grid_sample(voxel_attr.permute(0, 3, 1, 2).contiguous(), gb_ccm.unsqueeze(-4), mode=grid_interpolate_mode, align_corners=False).squeeze(-3).permute(0, 2, 3, 1)
                if background is not None:
                    if isinstance(background, float):
                        gb_voxel_attr = torch.lerp(torch.full_like(gb_voxel_attr, fill_value=background), gb_voxel_attr, alpha)
                    elif isinstance(background, torch.Tensor):
                        gb_voxel_attr = torch.lerp(background.to(gb_voxel_attr).expand_as(gb_voxel_attr), gb_voxel_attr, alpha)
                    else:
                        raise NotImplementedError
                if enable_antialis:
                    gb_voxel_attr = dr.antialias(gb_voxel_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                out.update({"gb_voxel_attr": gb_voxel_attr})

        if render_camera_position or render_distance or render_z_depth or render_ray_direction:
            batch_size = c2ws.shape
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast_duplicated, t_pos_idx)
            if render_camera_position:
                camera_position = torch.lerp(torch.full_like(camera_position, fill_value=0.0), camera_position, alpha)
                if enable_antialis:
                    camera_position = dr.antialias(camera_position, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
                out.update({"camera_position": camera_position})
            if render_distance:
                distance = torch.norm(camera_position, p=2, dim=-1, keepdim=True)
                distance = torch.lerp(torch.full_like(distance, fill_value=0.0), distance, alpha)
                if enable_antialis:
                    distance = dr.antialias(distance, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
                out.update({"distance": distance})
            if render_z_depth:
                z_depth = camera_position[:, :, :, [-1]]
                z_depth = torch.lerp(torch.zeros_like(z_depth), z_depth, alpha)
                if enable_antialis:
                    z_depth = dr.antialias(z_depth, rast_duplicated, v_tex_clip_duplicated, t_pos_idx)
                out.update({"z_depth": z_depth})
            if render_ray_direction:
                ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
                ray_direction = torch.lerp(torch.full_like(ray_direction, fill_value=-1.0), ray_direction, alpha)
                if enable_antialis:
                    ray_direction = dr.antialias(ray_direction, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
                out.update({"ray_direction": ray_direction})

        if render_cos_ray_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast_duplicated, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast_duplicated, t_pos_idx)
            ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
            cos_ray_normal = torch.sum(camera_normal * ray_direction, dim=-1, keepdim=True)
            cos_ray_normal = torch.lerp(torch.full_like(cos_ray_normal, fill_value=-1.0), cos_ray_normal, alpha)
            if enable_antialis:
                cos_ray_normal = dr.antialias(cos_ray_normal, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
            out.update({"cos_ray_normal": cos_ray_normal})

        if render_v_attr:
            gb_v_attr, _ = dr.interpolate(v_attr, rast, t_pos_idx)
            if background is not None:
                if isinstance(background, float):
                    gb_v_attr = torch.lerp(torch.full_like(gb_v_attr, fill_value=background), gb_v_attr, alpha)
                elif isinstance(background, torch.Tensor):
                    gb_v_attr = torch.lerp(background.to(gb_v_attr).expand_as(gb_v_attr), gb_v_attr, alpha)
                else:
                    raise NotImplementedError
            if enable_antialis:
                gb_v_attr = dr.antialias(gb_v_attr, rast, v_tex_clip, t_tex_idx)
            out.update({"v_attr": gb_v_attr})
        
        if render_uv:
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            proj_mtx = self.intr_to_proj(intrinsics)
            mvp_mtx = torch.matmul(proj_mtx, w2cs_mtx)
            v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
            v_pos_ndc = v_pos_clip[:, :, :2] / v_pos_clip[:, :, [3]]

            # t_mask_visible = self.get_visible_faces(mesh, c2ws, intrinsics, render_size)
            t_mask_visible = mesh.get_visible_faces(c2ws, perspective=self.perspective)
            v_tex_clip_visible = v_tex_clip.tile(batch_size, 1, 1)
            rast_visible = rast[:, :, :, [3]].clone().to(dtype=torch.int64).sub(1).tile(batch_size, 1, 1, 1)
            for b in range(batch_size):
                t_idx_visible = torch.where(t_mask_visible[b, :])[0]
                rast_visible[b, :, :, :] = torch.where(torch.isin(rast_visible[b, :, :, :], t_idx_visible), rast_visible[b, :, :, :], -1)
            rast_visible = torch.cat([rast[:, :, :, :3].tile(batch_size, 1, 1, 1), rast_visible.add(1).to(dtype=rast.dtype)], dim=-1)
            mask_visible = rast_visible[..., [3]] > 0
            if enable_antialis:
                alpha_visible = dr.antialias(mask_visible.float(), rast_visible, v_tex_clip_visible, t_tex_idx)
            else:
                alpha_visible = mask_visible.float()
            out.update({"uv_alpha": alpha_visible})
            
            gb_uv, _ = dr.interpolate(v_pos_ndc, rast_visible, t_pos_idx)
            gb_bg = torch.full_like(gb_uv, fill_value=-1.0)
            if enable_antialis:
                gb_uv_aa = torch.lerp(gb_bg, gb_uv, alpha_visible)
                gb_uv_aa = dr.antialias(gb_uv_aa, rast_visible, v_tex_clip_visible, t_tex_idx)
                gb_uv = torch.lerp(gb_bg, gb_uv, mask_visible.float())
                out.update({"uv": gb_uv_aa})
            else:
                gb_uv = torch.lerp(gb_bg, gb_uv, mask_visible.float())
                out.update({"uv": gb_uv})

            if render_map_attr:
                map_attr = map_attr.expand(batch_size, *map_attr.shape[-3:])
                if grid_interpolate_mode == 'nvdiffrast':
                    gb_map_attr = dr.texture(map_attr.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
                else:
                    gb_map_attr = torch.nn.functional.grid_sample(map_attr.permute(0, 3, 1, 2).contiguous(), gb_uv, mode=grid_interpolate_mode, align_corners=False).permute(0, 2, 3, 1)
                rast_map, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (map_attr.shape[-3], map_attr.shape[-2]))
                map_alpha = (rast_map[..., [3]] > 0).float()
                gb_map_alpha = torch.nn.functional.grid_sample(map_alpha.permute(0, 3, 1, 2).contiguous(), gb_uv, mode='nearest', align_corners=False).permute(0, 2, 3, 1)
                alpha_visible = torch.where(gb_map_alpha < 1.0, 0.0, alpha_visible)
                if background is not None:
                    if isinstance(background, float):
                        gb_map_attr = torch.lerp(torch.full_like(gb_map_attr, fill_value=background), gb_map_attr, alpha_visible)
                    elif isinstance(background, torch.Tensor):
                        gb_map_attr = torch.lerp(background.to(gb_map_attr).expand_as(gb_map_attr), gb_map_attr, alpha_visible)
                    else:
                        raise NotImplementedError
                else:
                    gb_map_attr = torch.where(gb_map_alpha < 1.0, map_attr[:, 0, 0, :].unsqueeze(1).unsqueeze(1), gb_map_attr)
                if enable_antialis:
                    gb_map_attr = dr.antialias(gb_map_attr.contiguous(), rast_visible, v_tex_clip_visible, t_tex_idx)
                out.update({"map_attr": gb_map_attr, "uv_alpha": alpha_visible})
        return out

    def global_inverse_rendering(
        self, mesh, image_attr:Tensor, c2ws:Tensor, intrinsics:Tensor, 
        render_size:Union[int, Tuple[int]], 
        texture_size:Union[int, Tuple[int]], 
        image_mask:Optional[Tensor]=None,
        image_mask_exclude_boundary=True,
        texture_attr:Optional[Tensor]=None,
        texture_mask:Optional[Tensor]=None,
        ray_angle_degree_threshold:Optional[float]=None,
        normal_angle_degree_threshold:Optional[float]=None,
        grid_interpolate_mode:str='linear',  # nearest, linear, barycentric
        inpainting_occlusion=True,
        n_neighbors=10,
        n_iters_max=1000,
        **kwargs,
    ) -> Tensor:
        assert grid_interpolate_mode in ['nearest', 'linear', 'barycentric']
        assert not ((texture_attr is None) ^ (texture_mask is None))
        batch_size = c2ws.shape[0]
        device = mesh.device
        image_height, image_width = (render_size, render_size) if isinstance(render_size, int) else render_size
        texture_height, texture_width = (texture_size, texture_size) if isinstance(texture_size, int) else texture_size

        # transform mesh to clip/uv space
        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        mvps = torch.matmul(self.intr_to_proj(intrinsics), c2w_to_w2c(c2ws))
        v_pos_clip = torch.matmul(v_pos_homo, mvps.permute(0, 2, 1))
        v_pos_ndc = v_pos_clip[:, :, :2] / v_pos_clip[:, :, [3]]
        v_pos_clip = v_pos_clip.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        v_pos_ndc = v_pos_ndc.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32, memory_format=torch.contiguous_format)

        v_tex_ndc = mesh.v_tex * 2.0 - 1.0
        v_tex_clip = torch.cat([v_tex_ndc, torch.zeros_like(v_tex_ndc[:, [0]]), torch.ones_like(v_tex_ndc[:, [0]])], dim=-1).unsqueeze(0)
        v_tex_clip = v_tex_clip.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        v_tex_ndc = v_tex_ndc.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        t_tex_idx = mesh.t_tex_idx.to(dtype=torch.int32, memory_format=torch.contiguous_format)

        # rasterize and raycast
        rast_image, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (image_height, image_width))
        rast_texture, _ = dr.rasterize(self.ctx, v_tex_clip, t_tex_idx, (texture_height, texture_width))
        t_mask_visible = mesh.get_visible_faces(c2ws, perspective=self.perspective)
        rast_texture_visible = rast_texture[:, :, :, [3]].to(dtype=torch.int64).sub(1).repeat(batch_size, 1, 1, 1)
        for b in range(batch_size):
            rast_texture_visible[b, :, :, :] = torch.where(
                torch.isin(
                    rast_texture_visible[b, :, :, :], 
                    torch.where(t_mask_visible[b, :])[0],
                ), 
                rast_texture_visible[b, :, :, :], 
                -1,
            )
        rast_texture_visible = torch.cat([
            rast_texture[:, :, :, :3].repeat(batch_size, 1, 1, 1), 
            rast_texture_visible.add(1).to(dtype=rast_texture.dtype),
        ], dim=-1)

        # interpolate
        pos_to_tex_mask = (rast_texture_visible[:, :, :, [-1]] > 0)
        # pos_to_tex, _ = dr.interpolate(v_pos_ndc, rast_texture_visible, t_pos_idx)
        # dv, du = torch.gradient(pos_to_tex, dim=(1, 2))
        # jacobian = (du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]).unsqueeze(-1)
        tex_to_pos_mask = (rast_image[:, :, :, [-1]] > 0)
        tex_to_pos, _ = dr.interpolate(v_tex_ndc, rast_image, t_tex_idx)
        tex_to_pos_cam, _ = dr.interpolate(torch.cat([
            torch.nn.functional.normalize(torch.matmul(v_pos_homo, c2w_to_w2c(c2ws).permute(0, 2, 1))[:, :, :3], dim=-1),
            torch.nn.functional.normalize(torch.matmul(mesh.v_nrm, c2ws[:, :3, :3]), dim=-1),
        ], dim=-1), rast_image, t_pos_idx)
        tex_to_tex = undiscretize(
            torch.cartesian_prod(
                torch.arange(texture_height, dtype=torch.float32, device=device), 
                torch.arange(texture_width, dtype=torch.float32, device=device),
            ).reshape(texture_height, texture_width, 2).flip(-1), 
            H=texture_height, 
            W=texture_width,
        )
        world_to_tex_mask = (rast_texture[:, :, :, [-1]] > 0)
        world_to_tex, _ = dr.interpolate(torch.cat([mesh.v_pos, mesh.v_nrm], dim=-1), rast_texture, t_pos_idx)
        world_to_tex_mask = world_to_tex_mask[0]
        world_to_tex = world_to_tex[0]

        # sample points on mesh
        samples_mask = tex_to_pos_mask
        if image_mask_exclude_boundary:
            samples_mask = samples_mask * \
            torch.roll(tex_to_pos_mask, 1, dims=1) * torch.roll(tex_to_pos_mask, -1, dims=1) * \
            torch.roll(tex_to_pos_mask, 1, dims=2) * torch.roll(tex_to_pos_mask, -1, dims=2)
        if normal_angle_degree_threshold is not None:
            samples_mask = samples_mask.logical_and(tex_to_pos_cam[:, :, :, [5]] > math.cos(math.radians(normal_angle_degree_threshold)))
        if ray_angle_degree_threshold is not None:
            samples_mask = samples_mask.logical_and(torch.sum(tex_to_pos_cam[:, :, :, :3] * tex_to_pos_cam[:, :, :, 3:6], dim=-1, keepdim=True) > math.cos(math.radians(ray_angle_degree_threshold)))
        if image_mask is not None:
            samples_mask = samples_mask.logical_and(image_mask)
        samples_u = torch.masked_select(tex_to_pos, samples_mask).reshape(-1, tex_to_pos.shape[-1])
        samples_v = torch.masked_select(image_attr, samples_mask).reshape(-1, image_attr.shape[-1])
        predicts_mask = pos_to_tex_mask.any(dim=0)
        if texture_attr is not None and texture_mask is not None:
            predicts_mask = predicts_mask.logical_and(texture_mask.logical_not())
        predicts_u = torch.masked_select(tex_to_tex, predicts_mask).reshape(-1, tex_to_tex.shape[-1])

        # interpolate
        if grid_interpolate_mode == 'nearest':
            interp_cls = NearestNDInterpolator
        elif grid_interpolate_mode == 'linear':
            interp_cls = LinearNDInterpolator
        elif grid_interpolate_mode == 'barycentric':
            interp_cls = CloughTocher2DInterpolator
        else:
            raise NotImplementedError(f'grid_interpolate_mode {grid_interpolate_mode} is not supported.')

        print(f"Interpolating, M={samples_u.shape[0]}, N={math.prod(predicts_u.shape[:-1])}, may take a while ...")
        t = perf_counter()
        interp = interp_cls(samples_u.detach().cpu().numpy(), samples_v.detach().cpu().numpy())
        predicts_v = interp(predicts_u.detach().cpu().numpy())
        predicts_v = torch.as_tensor(predicts_v, dtype=torch.float32, device=device)
        print(f"Interpolating wastes {perf_counter() - t} sec")

        map_attr = torch.zeros((*tex_to_tex.shape[:-1], predicts_v.shape[-1]), device=predicts_v.device, dtype=predicts_v.dtype)
        map_attr = map_attr.masked_scatter(predicts_mask, predicts_v)
        map_mask = predicts_mask.logical_and(torch.isnan(map_attr).any(dim=-1, keepdim=True).logical_not())
        map_attr = torch.nan_to_num(map_attr, nan=0.0).clamp(0.0, 1.0)
        if texture_attr is not None and texture_mask is not None:
            map_attr = torch.where(texture_mask, texture_attr, map_attr)
            map_mask = torch.logical_and(world_to_tex_mask, torch.logical_or(map_mask, texture_mask))
        map_alpha = world_to_tex_mask.to(dtype=torch.float32)

        # inpainting self-occlusion region
        if inpainting_occlusion:
            samples_mask = world_to_tex_mask
            samples_x = torch.masked_select(world_to_tex, samples_mask).reshape(-1, world_to_tex.shape[-1])

            print(f"Building tree, M=N={samples_x.shape[0]}, may take a while ...")
            t = perf_counter()
            kdtree = KDTree(samples_x[:, :3].detach().cpu().numpy(), leafsize=10)
            samples_d, samples_idx = kdtree.query(samples_x[:, :3].detach().cpu().numpy(), k=n_neighbors, workers=-1)
            samples_d = torch.as_tensor(samples_d, dtype=torch.float32, device=device)
            samples_idx = torch.as_tensor(samples_idx, dtype=torch.int64, device=device)
            print(f"Building tree wastes {perf_counter() - t} sec")

            # NOTE: check out the section 3.2 of the original paper https://arxiv.org/abs/2411.02336v1
            samples_d_score = torch.nn.functional.normalize(samples_d.reciprocal().nan_to_num(nan=0.0), p=1, dim=-1)
            samples_cos_nrm = torch.nn.functional.cosine_similarity(samples_x[samples_idx, 3:], samples_x[:, None, 3:], dim=-1)
            samples_weight = samples_d_score * samples_cos_nrm
            samples_y = torch.masked_select(map_attr, samples_mask).reshape(-1, map_attr.shape[-1])
            predicts_mask = map_mask.logical_not().logical_and(samples_mask)
            predicts_mask = torch.masked_select(predicts_mask, samples_mask).reshape(-1, predicts_mask.shape[-1])
            samples_neighbor_y = samples_y[samples_idx, :]
            samples_neighbor_mask = predicts_mask[samples_idx, :]
            predicts_mask_cur = predicts_mask.logical_and(samples_neighbor_mask.sum(dim=1) < n_neighbors - 1)
            for _ in range(n_iters_max):
                if predicts_mask_cur.sum() == 0:
                    break
                selected_neighbor_y = samples_neighbor_y[predicts_mask_cur[:, 0], :, :]
                selected_neighbor_mask_inv = ~samples_neighbor_mask[predicts_mask_cur[:, 0], :, :]
                selected_neighbor_weight = samples_weight[predicts_mask_cur[:, 0], :, None]
                selected_rgb = (selected_neighbor_mask_inv * selected_neighbor_weight * selected_neighbor_y).sum(dim=1) / (selected_neighbor_mask_inv * selected_neighbor_weight).sum(dim=1)
                samples_y = torch.masked_scatter(samples_y, predicts_mask_cur, selected_rgb)
                predicts_mask = predicts_mask.logical_and(predicts_mask_cur.logical_not())
                samples_neighbor_y = samples_y[samples_idx, :]
                samples_neighbor_mask = predicts_mask[samples_idx, :]
                predicts_mask_cur = predicts_mask.logical_and(samples_neighbor_mask.sum(dim=1) < n_neighbors - 1)
            map_attr = torch.zeros_like(map_attr)
            map_attr = map_attr.masked_scatter(samples_mask, samples_y)
            map_attr = torch.nan_to_num(map_attr, nan=0.0).clamp(0.0, 1.0)
        else:
            samples_x = None
            samples_y = None
        out = {
            'map_attr': map_attr,
            'map_alpha': map_alpha,
            'map_mask': map_mask,
            'samples_u': samples_u,
            'samples_v': samples_v,
            'predicts_u': predicts_u,
            'predicts_v': predicts_v,
            'samples_x': samples_x,
            'samples_y': samples_y,
            'tex_to_pos': tex_to_pos,
            'tex_to_pos_cam': tex_to_pos_cam,
            'world_to_tex': world_to_tex,
        }
        return out

    def geometry_rendering(
        self, mesh, 
        c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]],
        render_all_point_cloud=True,
        render_visible_point_cloud=True,
        render_z_depth=True,
        render_distance=True,
        render_world_normal=True,
        render_camera_normal=True,
        render_world_position=True,
        render_camera_position=True,
        render_ray_direction=True,
        render_cos_ray_normal=True,
        render_uv=True,
        background=None,
        **kwargs,
    ):
        return self.simple_rendering(
            mesh, None, None, None,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=False,
            render_uv=render_uv,
            render_map_attr=False,
            background=background,
            **kwargs,
        )
    
    def vertex_rendering(
        self, mesh, v_rgb:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]], 
        render_rgb=True,
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        background=None,
        **kwargs,
    ):
        render_result = self.simple_rendering(
            mesh, v_rgb, None, None,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=render_rgb,
            render_uv=False,
            render_map_attr=False,
            background=background,
            **kwargs,
        )
        if render_rgb:
            render_result['rgb'] = render_result['v_attr']
        return render_result
    
    def uv_rendering(
        self, mesh, map_Kd:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:Union[int, Tuple[int]], 
        render_rgb=True,
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        background=None,
        **kwargs,
    ):
        render_result = self.simple_rendering(
            mesh, None, map_Kd, None,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=False,
            render_uv=True,
            render_map_attr=render_rgb,
            background=background,
            **kwargs,
        )
        if render_rgb:
            render_result['rgb'] = render_result['map_attr']
        return render_result


