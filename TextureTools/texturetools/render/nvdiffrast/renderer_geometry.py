'''
Geometry Renderer for MV Texture
'''
import math
import os
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple, Union
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import trimesh

from ...camera.generator import generate_orbit_views_c2ws, generate_intrinsics
from ...camera.conversion import (
    intr_to_proj,
    c2w_to_w2c,
)
from ...io.mesh_loader import load_whole_mesh, convert_to_whole_mesh
from ...mesh.structure_v2 import PBRMesh, PBRScene, trimesh_to_pbr_mesh, pbr_mesh_to_trimesh
from ...pcd.knn import knn


class NVDiffRendererGeometry(nn.Module):
    def __init__(self, device='cuda', pbr_mesh:Optional[PBRMesh]=None):
        super().__init__()
        # NOTE: no check here
        # assert pbr_mesh is not None
        self.device = torch.device(device)
        self.kernel_dict = {k: torch.nn.functional.pad(torch.full((1, 1, k-2, k-2), fill_value=-1.0, dtype=torch.float32, device=self.device), (1, 1, 1, 1), mode='constant', value=k ** 2) for k in [3, 5, 7, 9]}
        self.pbr_mesh:PBRMesh = pbr_mesh
        self.enable_nvdiffrast_cuda_ctx()

    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def enable_nvdiffrast_opengl_ctx(self):
        self.ctx = dr.RasterizeGLContext(device=self.device)

    def clear(self):
        self.pbr_mesh = None

    @classmethod
    def from_file(cls, device='cuda', pbr_mesh_path:Optional[str]=None):
        assert pbr_mesh_path is not None
        if pbr_mesh_path is not None:
            pbr_mesh = trimesh_to_pbr_mesh(load_whole_mesh(pbr_mesh_path))
        else:
            pbr_mesh = None
        return cls(device=device, pbr_mesh=pbr_mesh)

    @classmethod
    def from_trimesh(cls, device='cuda', pbr_mesh:Optional[Union[trimesh.Trimesh, trimesh.Scene]]=None):
        assert pbr_mesh is not None
        if pbr_mesh is not None:
            pbr_mesh = trimesh_to_pbr_mesh(convert_to_whole_mesh(pbr_mesh))
        else:
            pbr_mesh = None
        return cls(device=device, pbr_mesh=pbr_mesh)

    def update_from_file(self, pbr_mesh_path:Optional[str]=None):
        assert pbr_mesh_path is not None
        if pbr_mesh_path is not None:
            pbr_mesh = trimesh_to_pbr_mesh(load_whole_mesh(pbr_mesh_path))
        else:
            pbr_mesh = None
        self.pbr_mesh = pbr_mesh
        return self

    def update_from_trimesh(self, pbr_mesh:Optional[Union[trimesh.Trimesh, trimesh.Scene]]=None):
        assert pbr_mesh is not None
        if pbr_mesh is not None:
            pbr_mesh = trimesh_to_pbr_mesh(convert_to_whole_mesh(pbr_mesh))
        else:
            pbr_mesh = None
        self.pbr_mesh = pbr_mesh
        return self

    def geometry_rendering(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor,
        render_size:Union[int, Tuple[int]],
        render_size_2d:Union[int, Tuple[int]],
        perspective=True,
        render_mesh=False,
        render_uv=False,
        enable_antialis=False,
    ) -> Dict[str, torch.Tensor]:
        '''
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        NOTE: if H != W, render image will be resized directly
        '''
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        height_2d, width_2d = (render_size_2d, render_size_2d) if isinstance(render_size_2d, int) else render_size_2d
        out = dict()

        ## rasterize 3D mesh
        if render_mesh:
            vertices_homo = torch.cat([self.pbr_mesh.vertices, torch.ones_like(self.pbr_mesh.vertices[:, [0]])], dim=-1)
            vertices_clip = torch.matmul(vertices_homo, torch.matmul(intr_to_proj(intrinsics, perspective=perspective), c2w_to_w2c(c2ws)).permute(0, 2, 1))
            faces = self.pbr_mesh.faces.to(dtype=torch.int32)
            rast_out, _ = dr.rasterize(self.ctx, vertices_clip, faces, (height, width))
            mask = (rast_out[..., [3]] > 0)
            alpha = mask.to(dtype=torch.float32)
            vertices_attrs = torch.cat([self.pbr_mesh.vertices, self.pbr_mesh.vertex_normals], dim=-1)
            attrs, _ = dr.interpolate(vertices_attrs.contiguous(), rast_out, faces)
            if enable_antialis:
                alpha = dr.antialias(alpha, rast_out, vertices_clip, faces)
                attrs = dr.antialias(attrs, rast_out, vertices_clip, faces)
            out.update({
                'mask': mask,
                'alpha': alpha,
                'ccm': attrs[:, :, :, 0:3],
                'normal': attrs[:, :, :, 3:6],
            })

        ## rasterize 2D mesh
        if render_uv:
            faces = self.pbr_mesh.faces.to(dtype=torch.int32)
            faces_2d = self.pbr_mesh.faces_2d.to(dtype=torch.int32)
            uvs_2d_clip = torch.cat([self.pbr_mesh.uvs_2d, torch.zeros_like(self.pbr_mesh.uvs_2d[:, [0]]), torch.ones_like(self.pbr_mesh.uvs_2d[:, [0]])], dim=-1).unsqueeze(0)
            rast_2d_out, _ = dr.rasterize(self.ctx, uvs_2d_clip, faces_2d, (height_2d, width_2d))
            mask_2d = (rast_2d_out[..., [3]] > 0)
            alpha_2d = mask_2d.to(dtype=torch.float32)
            vertices_attrs = torch.cat([self.pbr_mesh.vertices, self.pbr_mesh.vertex_normals], dim=-1)
            attrs_2d, _ = dr.interpolate(vertices_attrs.contiguous(), rast_2d_out, faces)
            if enable_antialis:
                alpha_2d = dr.antialias(alpha_2d, rast_2d_out, uvs_2d_clip, faces_2d)
                attrs_2d = dr.antialias(attrs_2d, rast_2d_out, uvs_2d_clip, faces_2d)
            out.update({
                'mask_2d': mask_2d,
                'alpha_2d': alpha_2d,
                'ccm_2d': attrs_2d[:, :, :, 0:3],
                'normal_2d': attrs_2d[:, :, :, 3:6],
            })
        return out

    def geometry_inverse_rendering(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor,
        image_attrs:torch.Tensor,
        render_size:Union[int, Tuple[int]],
        render_size_2d:Union[int, Tuple[int]],
        perspective=True,
        enable_antialis=False,
        grid_interpolate_mode='torch',
        kernel_mode=7,
        grad_norm_threhold=0.20,
        ray_normal_angle_threhold=115.0,
        kernel_size_seamless=9,
        n_neighbors_seamless=8,
        n_neighbors_full=32,
    ) -> Dict[str, torch.Tensor]:
        '''
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        image_attrs: [M, H, W, C]
        '''
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        height_2d, width_2d = (render_size_2d, render_size_2d) if isinstance(render_size_2d, int) else render_size_2d
        assert kernel_mode in self.kernel_dict.keys(), f'kernel_mode should be one of {list(self.kernel_dict.keys())}'
        assert 0.0 <= ray_normal_angle_threhold <= 180.0, f'degree of ray_normal_angle_threhold should be in [0.0, 180.0], but {ray_normal_angle_threhold}'

        ## rasterize 3D mesh
        vertices_homo = torch.cat([self.pbr_mesh.vertices, torch.ones_like(self.pbr_mesh.vertices[:, [0]])], dim=-1)
        vertices_clip = torch.matmul(vertices_homo, torch.matmul(intr_to_proj(intrinsics, perspective=perspective), c2w_to_w2c(c2ws)).permute(0, 2, 1))
        vertices_ndc = vertices_clip[..., :2] / vertices_clip[..., [3]]
        faces = self.pbr_mesh.faces.to(dtype=torch.int32)
        rast_out, _ = dr.rasterize(self.ctx, vertices_clip, faces, (height, width))
        mask = (rast_out[..., [3]] > 0)
        alpha = mask.to(dtype=torch.float32)
        tid = rast_out[..., [3]].to(dtype=torch.int64).sub(1)
        face_normal = self.pbr_mesh.normals.gather(0, torch.where(mask, tid, 0).reshape(-1, 1).repeat(1, 3)).reshape(batch_size, height, width, 3)
        vertices_attrs = torch.cat([self.pbr_mesh.vertices, self.pbr_mesh.vertex_normals], dim=-1)
        attrs, _ = dr.interpolate(vertices_attrs.contiguous(), rast_out, faces)
        # NOTE: if grad of ccm or normal is greater than threhold, then grad of alpha is greater than threhold
        attrs_dy, attrs_dx = torch.gradient(attrs, dim=(1, 2))
        attrs_grad_norm = (attrs_dx.square() + attrs_dy.square()).sum(dim=-1, keepdim=True).sqrt()
        if perspective:
            rays_o = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
            rays_d = attrs[:, :, :, 0:3] - rays_o
        else:
            rays_d = c2ws[:, :3, 2].neg().unsqueeze(1).unsqueeze(1)
            rays_o = attrs[:, :, :, 0:3] - (2.0 * math.sqrt(3.0)) * rays_d
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        rays_o, rays_d = torch.broadcast_tensors(rays_o, rays_d)

        ## process visibility and gradient
        # NOTE: vertex normal fails for large faces
        mask_visiable = torch.logical_and(mask, torch.nn.functional.cosine_similarity(rays_d, face_normal, dim=-1).unsqueeze(-1) < math.cos(math.radians(ray_normal_angle_threhold)))
        mask_visiable = torch.logical_and(mask_visiable, attrs_grad_norm < grad_norm_threhold)
        alpha_visiable = mask_visiable.to(dtype=torch.float32)

        ## rasterize 2D mesh
        faces_2d = self.pbr_mesh.faces_2d.to(dtype=torch.int32)
        uvs_2d_clip = torch.cat([self.pbr_mesh.uvs_2d, torch.zeros_like(self.pbr_mesh.uvs_2d[:, [0]]), torch.ones_like(self.pbr_mesh.uvs_2d[:, [0]])], dim=-1).unsqueeze(0)
        rast_2d_out, _ = dr.rasterize(self.ctx, uvs_2d_clip, faces_2d, (height_2d, width_2d))
        mask_2d = (rast_2d_out[..., [3]] > 0)
        alpha_2d = mask_2d.to(dtype=torch.float32)
        tid_2d = rast_2d_out[..., [3]].to(dtype=torch.int64).sub(1)
        ccm_2d, _ = dr.interpolate(self.pbr_mesh.vertices.contiguous(), rast_2d_out, faces)
        face_normal_2d = self.pbr_mesh.normals.gather(0, torch.where(mask_2d, tid_2d, 0).reshape(-1, 1).repeat(1, 3)).reshape(1, height_2d, width_2d, 3)
        if perspective:
            rays_o_2d = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
            rays_d_2d = ccm_2d - rays_o_2d
        else:
            rays_d_2d = c2ws[:, :3, 2].neg().unsqueeze(1).unsqueeze(1)
            rays_o_2d = ccm_2d - (2.0 * math.sqrt(3.0)) * rays_d_2d
        rays_d_2d = torch.nn.functional.normalize(rays_d_2d, dim=-1)
        rays_o_2d, rays_d_2d = torch.broadcast_tensors(rays_o_2d, rays_d_2d)
        ray_normal_cos_2d = torch.nn.functional.cosine_similarity(rays_d_2d, face_normal_2d, dim=-1).unsqueeze(-1)

        ## image to uv (partial)
        image_to_uv, _ = dr.interpolate(vertices_ndc.contiguous(), rast_2d_out.repeat(batch_size, 1, 1, 1), faces)
        image_attrs_all = torch.cat([image_attrs, attrs_grad_norm], dim=-1)
        if grid_interpolate_mode in ['torch', 'pytorch']:
            attrs_2d_all = torch.nn.functional.grid_sample(
                image_attrs_all.permute(0, 3, 1, 2),
                image_to_uv,
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)
        elif grid_interpolate_mode in ['nvdiff', 'nvdiffrast']:
            attrs_2d_all = dr.texture(
                image_attrs_all.contiguous(),
                image_to_uv.mul(0.5).add(0.5),
                filter_mode='linear',
            )
        else:
            raise NotImplementedError(f'grid_interpolate_mode {grid_interpolate_mode} is not supported')
        if enable_antialis:
            attrs_2d_all = dr.antialias(attrs_2d_all.contiguous(), rast_2d_out, uvs_2d_clip, faces_2d)
        attrs_2d, attrs_grad_norm_2d = attrs_2d_all.split([image_attrs.shape[-1], attrs_grad_norm.shape[-1]], dim=-1)

        ## process visibility and gradient
        tid_2d_valid = torch.masked_select(tid_2d.squeeze(-1), mask_2d.squeeze(-1))
        face_normal_2d_valid = torch.masked_select(face_normal_2d, mask_2d).reshape(-1, face_normal_2d.shape[-1])
        rays_o_2d_valid = torch.masked_select(rays_o_2d, mask_2d).reshape(batch_size, -1, rays_o_2d.shape[-1])
        rays_d_2d_valid = torch.masked_select(rays_d_2d, mask_2d).reshape(batch_size, -1, rays_d_2d.shape[-1])
        _, _, rays_tid_2d_valid, _, _ = self.pbr_mesh.optix.intersects_closest(rays_o_2d_valid, rays_d_2d_valid)
        rays_tid_2d_valid = rays_tid_2d_valid.to(dtype=torch.int64)
        # NOTE: points on mesh near edges may cause misjudgments here
        rays_mask_valid = torch.logical_and(rays_tid_2d_valid == tid_2d_valid, rays_tid_2d_valid != -1)
        # NOTE: vertex normal fails for large faces
        rays_mask_valid = torch.logical_and(rays_mask_valid, torch.nn.functional.cosine_similarity(rays_d_2d_valid, face_normal_2d_valid, dim=-1) < math.cos(math.radians(ray_normal_angle_threhold)))
        mask_2d_visiable = torch.zeros((batch_size, height_2d, width_2d, 1), dtype=mask_2d.dtype, device=mask_2d.device)
        mask_2d_visiable.reshape(batch_size, height_2d * width_2d).masked_scatter_(mask_2d.reshape(mask_2d.shape[0], height_2d * width_2d), rays_mask_valid)
        # NOTE: postprocess ray tracing misjudgments 
        kernel_list = list(self.kernel_dict.keys())
        for i in range(len(kernel_list)):
            k = kernel_list.pop(0)
            if kernel_mode in kernel_list:
                mask_2d_visiable = torch.logical_or(
                    mask_2d_visiable, 
                    torch.nn.functional.conv2d(
                        mask_2d_visiable.to(dtype=torch.float32).permute(0, 3, 1, 2), 
                        weight=self.kernel_dict[k], stride=1, padding=k//2, 
                    ).permute(0, 2, 3, 1) >= ((k - 1) ** 2 - 1) * ((k - 2) ** 2),
                )
        mask_2d_visiable = torch.logical_and(mask_2d_visiable, mask_2d)
        mask_2d_visiable = torch.logical_and(mask_2d_visiable, ray_normal_cos_2d < ray_normal_angle_threhold)
        mask_2d_visiable = torch.logical_and(mask_2d_visiable, attrs_grad_norm_2d < grad_norm_threhold)
        alpha_2d_visiable = mask_2d_visiable.to(dtype=torch.float32)
        ray_normal_cos_2d = ray_normal_cos_2d * alpha_2d_visiable
        attrs_2d = attrs_2d * alpha_2d_visiable

        ## image to uv (full)
        image_attrs_valid = torch.masked_select(image_attrs, mask_visiable).reshape(-1, image_attrs.shape[-1])
        attrs_valid = torch.masked_select(attrs, mask_visiable).reshape(-1, attrs.shape[-1])
        face_normal_valid = torch.masked_select(face_normal, mask_visiable).reshape(-1, face_normal.shape[-1])
        '''
        # NOTE: for debug
        torch.save({
            'src': attrs_valid[:, 0:3],
            'dst': ccm_2d.reshape(-1, 3),
            'k': n_neighbors_full,
        }, 'kdtree_states.pth')
        '''
        score, index = knn(attrs_valid[:, 0:3], ccm_2d.reshape(-1, 3), k=n_neighbors_full, batch_size=1024*1024)
        # NOTE: https://arxiv.org/abs/2411.02336v1, Section 3.2
        weight = torch.nn.functional.normalize(score.reciprocal().nan_to_num(nan=0.0), p=1, dim=-1) * \
            torch.nn.functional.cosine_similarity(face_normal_valid[index, :], face_normal_2d.reshape(-1, 3).unsqueeze(-2), dim=-1)
        weight = weight.unsqueeze(-1)
        weight_sum = weight.sum(dim=-2, keepdim=False)
        image_attrs_valid_full = image_attrs_valid[index, :]
        attrs_2d_full = (image_attrs_valid_full * weight).sum(dim=-2) / weight_sum
        attrs_2d_full = torch.where(weight_sum > 0.0, attrs_2d_full, image_attrs_valid_full.mean(dim=-2))
        attrs_2d_full = attrs_2d_full.reshape(1, height_2d, width_2d, image_attrs_valid_full.shape[-1])

        ## process seamless
        # NOTE: inner boundary of uv charts/atlas/islands
        kernel_size_seamless = 2 * (kernel_size_seamless // 2) + 1
        alpha_2d_boundary = alpha_2d - (1.0 - torch.nn.functional.max_pool2d(1.0 - alpha_2d.permute(0, 3, 1, 2), kernel_size_seamless, 1, kernel_size_seamless // 2)).permute(0, 2, 3, 1)
        mask_2d_boundary = (alpha_2d_boundary > 0.0)
        alpha_2d_boundary = mask_2d_boundary.to(dtype=torch.float32)
        ccm_2d_boundary_valid = torch.masked_select(ccm_2d, mask_2d_boundary).reshape(-1, ccm_2d.shape[-1])
        score, index = knn(ccm_2d_boundary_valid, ccm_2d_boundary_valid, k=n_neighbors_seamless+1)
        score = score[:, 1:]
        index = index[:, 1:]
        index_spatial = torch.arange(height_2d * width_2d, dtype=torch.int64, device=self.device).reshape(1, height_2d, width_2d, 1).masked_select(mask_2d_boundary).reshape(-1)[index]
        ccm_2d_boundary = torch.where(mask_2d_boundary, ccm_2d, 0.0)
        ccm_2d_boundary_inv = torch.zeros_like(ccm_2d)
        # NOTE: select from ccm_2d_boundary_valid with index
        # ccm_2d_boundary_inv.masked_scatter_(mask_2d_boundary, ccm_2d_boundary_valid[index, :].mean(dim=1, keepdim=False))
        # NOTE: select from ccm_2d with index_spatial
        ccm_2d_boundary_inv.masked_scatter_(mask_2d_boundary, ccm_2d.reshape(1, height_2d * width_2d, 3)[:, index_spatial, :].mean(dim=2, keepdim=False).reshape(-1, 3))

        out = {
            'mask_visiable': mask_visiable,
            'alpha_visiable': alpha_visiable,
            'mask_2d': mask_2d,
            'alpha_2d': alpha_2d,
            'mask_2d_visiable': mask_2d_visiable,
            'alpha_2d_visiable': alpha_2d_visiable,
            'ray_normal_cos_2d': ray_normal_cos_2d,
            'attrs_2d': attrs_2d,
            'attrs_2d_full': attrs_2d_full,
            'mask_2d_boundary': mask_2d_boundary,
            'alpha_2d_boundary': alpha_2d_boundary,
            'index_spatial': index_spatial,
            'ccm_2d_boundary': ccm_2d_boundary,
            'ccm_2d_boundary_inv': ccm_2d_boundary_inv,
        }
        return out


#### some test functions ####


def test_renderer():
    from .renderer_utils import image_to_tensor, tensor_to_image, tensor_to_video

    pbr_mesh_path = '/home/chenxiao/code/MVDiffusion/test_data/online_cases/d2b57ca9-f299-44d8-b265-b28d4db5f8d7_outputs/cache/processed_mesh.obj'
    result_dir = 'test_result/test_geometry_renderer'

    renderer = NVDiffRendererGeometry.from_file(pbr_mesh_path=pbr_mesh_path)
    renderer.pbr_mesh.scale_to_bbox()

    N, H, W, H2D, W2D = 250, 512, 512, 2048, 2048
    c2ws = generate_orbit_views_c2ws(N+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:N]
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    for _ in range(3):
        t = perf_counter()
        render_out = renderer.geometry_rendering(c2ws, intrinsics, (H, W), (H2D, W2D), perspective=True, render_mesh=True, render_uv=True)
        print('geometry_rendering', perf_counter() - t)

    os.makedirs(result_dir, exist_ok=True)
    imageio.mimsave(
        os.path.join(result_dir, 'alpha.mp4'),
        tensor_to_video(render_out['alpha'])[..., [0,0,0]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'ccm.mp4'),
        tensor_to_video(render_out['ccm'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'normal.mp4'),
        tensor_to_video(render_out['normal'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )
    tensor_to_image(render_out['alpha_2d'])[0].save(os.path.join(result_dir, 'alpha_2d.png'))
    tensor_to_image(render_out['ccm_2d'] * 0.5 + 0.5)[0].save(os.path.join(result_dir, 'ccm_2d.png'))
    tensor_to_image(render_out['normal_2d'] * 0.5 + 0.5)[0].save(os.path.join(result_dir, 'normal_2d.png'))


def test_inverse_renderer():
    from .renderer_utils import image_to_tensor, tensor_to_image, tensor_to_video

    pbr_mesh_path = '/home/chenxiao/code/MVDiffusion/test_data/online_cases/d2b57ca9-f299-44d8-b265-b28d4db5f8d7_outputs/cache/processed_mesh.obj'
    result_dir = 'test_result/test_geometry_renderer'

    renderer = NVDiffRendererGeometry.from_file(pbr_mesh_path=pbr_mesh_path)
    renderer.pbr_mesh.scale_to_bbox()

    N, H, W, H2D, W2D = 250, 512, 512, 512, 512
    c2ws = generate_orbit_views_c2ws(N+1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:N]
    intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    render_out = renderer.geometry_rendering(c2ws, intrinsics, (H, W), (H2D, W2D), perspective=True, render_mesh=True, render_uv=True)
    image_attrs = torch.cat([render_out['alpha'], render_out['ccm'], render_out['normal']], dim=-1)

    for _ in range(3):
        t = perf_counter()
        inverse_render_out = renderer.geometry_inverse_rendering(c2ws, intrinsics, image_attrs, (H, W), (H2D, W2D), perspective=True)
        print('geometry_inverse_rendering', perf_counter() - t)

    uv_attrs = inverse_render_out['attrs_2d']
    inverse_render_out['alpha'], inverse_render_out['ccm'], inverse_render_out['normal'] = uv_attrs.split([1, 3, 3], dim=-1)

    os.makedirs(result_dir, exist_ok=True)
    imageio.mimsave(
        os.path.join(result_dir, 'alpha.mp4'),
        tensor_to_video(render_out['alpha'])[..., [0,0,0]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'ccm.mp4'),
        tensor_to_video(render_out['ccm'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'normal.mp4'),
        tensor_to_video(render_out['normal'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'alpha_inv.mp4'),
        tensor_to_video(inverse_render_out['alpha'])[..., [0,0,0]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'ccm_inv.mp4'),
        tensor_to_video(inverse_render_out['ccm'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )
    imageio.mimsave(
        os.path.join(result_dir, 'normal_inv.mp4'),
        tensor_to_video(inverse_render_out['normal'] * 0.5 + 0.5)[..., [0,1,2]],
        fps=25,
    )

