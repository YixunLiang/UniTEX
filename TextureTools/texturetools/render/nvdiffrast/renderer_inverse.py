'''
Geometry Renderer for MV Texture
'''
from glob import glob
import math
import os
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
import nvdiffrast.torch as dr
from tqdm import tqdm
import trimesh

from ...camera.conversion import (
    intr_to_proj,
    c2w_to_w2c,
)
from ...io.mesh_loader import load_whole_mesh, convert_to_whole_mesh
from ...pcd.structure import PointCloud
from ...mesh.structure_v2 import PBRMesh, PBRScene, trimesh_to_pbr_mesh, pbr_mesh_to_trimesh
from ...pcd.knn import knn
from ...image.image_fusion import image_fusion
from ...image.image_outpainting import image_outpainting
from ...texture.stitching.mip import pull_push
from ...image.gaussian_blur import gaussian_blur
from ...image.lens_blur import lens_blur_torch
from .renderer_utils import image_to_tensor, tensor_to_image
from ...io.link_pbr_to_mesh import link_rgb_to_mesh, link_pbr_to_mesh


class NVDiffRendererInverse(nn.Module):
    def __init__(self, device='cuda', pbr_mesh:Optional[PBRMesh]=None):
        super().__init__()
        # NOTE: no check here
        # assert pbr_mesh is not None
        self.device = torch.device(device)
        self.kernel_dict = {k: torch.nn.functional.pad(torch.full((1, 1, k-2, k-2), fill_value=-1.0, dtype=torch.float32, device=self.device), (1, 1, 1, 1), mode='constant', value=k ** 2) for k in [3, 5, 7, 9]}
        self.pbr_mesh:PBRMesh = pbr_mesh
        self.enable_nvdiffrast_cuda_ctx()
        self.index = [0, 3, 4, 1, 2, 5]  # frtbld ==> fblrtd
        self.query_field_function = None

    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def enable_nvdiffrast_opengl_ctx(self):
        self.ctx = dr.RasterizeGLContext(device=self.device)

    def clear(self):
        self.pbr_mesh = None
        self.query_field_function = None

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

    def register_query_field(self, query_field:Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]=None):
        '''
        Args:
            vertices_visiable: [V_vis, 3], float32
            colors_visiable: [V_vis, C], float32
            vertices_invisiable: [V_invis, 3], float32
        Returns:
            colors_invisiable: [V_invis, C], float32
        '''
        self.query_field_function = query_field
        return self

    def test_query_field(self, blank_mesh:Union[trimesh.Trimesh, str]) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        # NOTE: create a new renderer here to avoid inplace operators
        if isinstance(blank_mesh, str):
            renderer = self.__class__.from_file(device=self.device, pbr_mesh_path=blank_mesh)
        else:
            renderer = self.__class__.from_trimesh(device=self.device, pbr_mesh_path=blank_mesh)  # NOTE: not safe
        vertices_attrs = torch.cat([renderer.pbr_mesh.vertices, renderer.pbr_mesh.vertex_normals], dim=-1)
        uvs_2d_clip = torch.cat([renderer.pbr_mesh.uvs_2d, torch.zeros_like(renderer.pbr_mesh.uvs_2d[:, [0]]), torch.ones_like(renderer.pbr_mesh.uvs_2d[:, [0]])], dim=-1).unsqueeze(0)
        faces = renderer.pbr_mesh.faces.to(dtype=torch.int32)
        faces_2d = renderer.pbr_mesh.faces_2d.to(dtype=torch.int32)
        rast_2d_out, _ = dr.rasterize(renderer.ctx, uvs_2d_clip, faces_2d, (2048, 2048))
        mask_2d = (rast_2d_out[..., [3]] > 0)
        mask_2d_sum = mask_2d.sum().item()
        attr_2d, _ = dr.interpolate(vertices_attrs.contiguous(), rast_2d_out, faces)
        attr_2d[..., 3:] = attr_2d[..., 3:] * 0.5 + 0.5  # NOTE: using global normal as fake gt
        attr_2d_valid = torch.masked_select(attr_2d, mask_2d).reshape(-1, attr_2d.shape[-1])
        random_index = torch.randperm(mask_2d_sum, dtype=torch.int64, device='cuda')
        attr_2d_visiable = attr_2d_valid[random_index[:mask_2d_sum//2], :]
        color_2d_valid = self.query_field(
            vertices_visiable=attr_2d_visiable[:, :3],
            colors_visiable=attr_2d_visiable[:, 3:],
            vertices_invisiable=attr_2d_valid[:, :3],
        )
        color_2d:torch.Tensor = attr_2d.clone()
        color_2d[..., 3:].masked_scatter_(mask_2d, color_2d_valid)
        textured_mesh_gt = link_rgb_to_mesh(
            src_path=blank_mesh,
            rgb_path=tensor_to_image(attr_2d[0, :, :, 3:]),
            dst_path=None,
        )
        textured_mesh_pred = link_rgb_to_mesh(
            src_path=blank_mesh,
            rgb_path=tensor_to_image(color_2d[0, :, :, 3:]),
            dst_path=None,
        )
        return textured_mesh_gt, textured_mesh_pred

    def query_field(
        self,
        vertices_visiable: torch.Tensor,
        colors_visiable: torch.Tensor,
        vertices_invisiable: torch.Tensor,
    ) -> torch.Tensor:
        '''
        vertices_visiable: [V_vis, 3], float32
        colors_visiable: [V_vis, C], float32
        vertices_invisiable: [V_invis, 3], float32
        colors_invisiable: [V_invis, C], float32
        '''
        if self.query_field_function is None:
            raise NotImplementedError(f'using register_query_field before query')
        colors_invisiable = self.query_field_function(vertices_visiable, colors_visiable, vertices_invisiable)
        return colors_invisiable

    def mv_to_pcd(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor,
        render_size:Union[int, Tuple[int]],
        image_attrs:Optional[torch.Tensor]=None,
        perspective=True,
        grad_norm_threhold=0.20,
        ray_normal_angle_threhold=115.0,
        filt_gradient_points=False
    ):
        '''
        convert mv images to point cloud
        '''
        assert c2ws.shape[0] == len(self.index)
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
        assert 0.0 <= ray_normal_angle_threhold <= 180.0, f'degree of ray_normal_angle_threhold should be in [0.0, 180.0], but {ray_normal_angle_threhold}'

        ## compute on mesh
        vertices_homo = torch.cat([self.pbr_mesh.vertices, torch.ones_like(self.pbr_mesh.vertices[:, [0]])], dim=-1)
        vertices_clip = torch.matmul(vertices_homo, torch.matmul(intr_to_proj(intrinsics, perspective=perspective), c2w_to_w2c(c2ws)).permute(0, 2, 1))
        vertices_attrs = torch.cat([self.pbr_mesh.vertices, self.pbr_mesh.vertex_normals], dim=-1)
        faces = self.pbr_mesh.faces.to(dtype=torch.int32)

        ## rasterize 3D mesh
        rast_out, _ = dr.rasterize(self.ctx, vertices_clip, faces, (height, width))
        mask = (rast_out[..., [3]] > 0)
        alpha = (mask > 0).to(dtype=torch.float32)
        tid = rast_out[..., [3]].to(dtype=torch.int64).sub(1)
        face_normal = self.pbr_mesh.normals.gather(0, torch.where(mask, tid, 0).reshape(-1, 1).repeat(1, 3)).reshape(batch_size, height, width, 3)
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
        ray_normal_cos = torch.nn.functional.cosine_similarity(rays_d, face_normal, dim=-1).unsqueeze(-1)

        ## process visibility and gradient
        # NOTE: vertex normal fails for large faces

        ####dilation!
        if filt_gradient_points:
            visible_mask = attrs_grad_norm < grad_norm_threhold
            max_pool = nn.MaxPool2d(kernel_size=31, stride=1, padding=15)
            visible_mask_dilate_grad =(1. -max_pool(1.-visible_mask.float())).bool()
            mask_visiable = torch.logical_and(mask, ray_normal_cos < math.cos(math.radians(ray_normal_angle_threhold)))
            mask_visiable = torch.logical_and(mask_visiable, visible_mask_dilate_grad)
            alpha_visiable = mask_visiable.to(dtype=torch.float32)
        else:
            mask_visiable = mask  
            alpha_visiable = mask_visiable.to(dtype=torch.float32)      
        # mask_visiable = mask

        ## reshape image to point cloud
        attrs_valid = torch.masked_select(attrs, mask).reshape(-1, attrs.shape[-1])
        face_normal_valid = torch.masked_select(face_normal, mask).reshape(-1, face_normal.shape[-1])
        if image_attrs is not None:
            image_attrs_valid = torch.masked_select(image_attrs, mask).reshape(-1, image_attrs.shape[-1])
        else:
            image_attrs_valid = None
        alpha_visiable_valid = torch.masked_select(mask_visiable, mask).reshape(-1, mask_visiable.shape[-1]).to(dtype=torch.float32)
        point_cloud = PointCloud(vertices=attrs_valid[..., :3], normals=face_normal_valid, colors=image_attrs_valid, alphas=alpha_visiable_valid)
        attrs_visiable = torch.masked_select(attrs, mask_visiable).reshape(-1, attrs.shape[-1])
        face_normal_visiable = torch.masked_select(face_normal, mask_visiable).reshape(-1, face_normal.shape[-1])
        if image_attrs is not None:
            image_attrs_visiable = torch.masked_select(image_attrs, mask_visiable).reshape(-1, image_attrs.shape[-1])
        else:
            image_attrs_visiable = None
        point_cloud_visiable = PointCloud(vertices=attrs_visiable[..., :3], normals=face_normal_visiable, colors=image_attrs_visiable)
        return {
            'mask': mask,
            'alpha': alpha,
            'mask_visiable': mask_visiable,
            'alpha_visiable': alpha_visiable,
            'point_cloud': point_cloud,
            'point_cloud_visiable': point_cloud_visiable,
        }

    def uv_to_pcd(
        self, c2ws:torch.Tensor, intrinsics:torch.Tensor,
        render_size_2d:Union[int, Tuple[int]],
        image_attrs:Optional[torch.Tensor]=None,
        alpha_attrs:Optional[torch.Tensor]=None,
        perspective=True,
        ray_normal_angle_threhold=115.0,
        grid_interpolate_mode='torch',
        kernel_mode=7,
    ) -> Dict[str, torch.Tensor]:
        '''
        convert uv map to point cloud
        '''
        assert c2ws.shape[0] == len(self.index)
        batch_size = c2ws.shape[0]
        height_2d, width_2d = (render_size_2d, render_size_2d) if isinstance(render_size_2d, int) else render_size_2d
        assert 0.0 <= ray_normal_angle_threhold <= 180.0, f'degree of ray_normal_angle_threhold should be in [0.0, 180.0], but {ray_normal_angle_threhold}'
        assert grid_interpolate_mode in ['torch', 'pytorch', 'nvdiff', 'nvdiffrast']
        assert kernel_mode in self.kernel_dict.keys(), f'kernel_mode should be one of {list(self.kernel_dict.keys())}'

        ## compute on mesh
        vertices_homo = torch.cat([self.pbr_mesh.vertices, torch.ones_like(self.pbr_mesh.vertices[:, [0]])], dim=-1)
        vertices_clip = torch.matmul(vertices_homo, torch.matmul(intr_to_proj(intrinsics, perspective=perspective), c2w_to_w2c(c2ws)).permute(0, 2, 1))
        vertices_ndc = vertices_clip[..., :2] / vertices_clip[..., [3]]
        vertices_attrs = torch.cat([self.pbr_mesh.vertices, self.pbr_mesh.vertex_normals], dim=-1)
        uvs_2d_clip = torch.cat([self.pbr_mesh.uvs_2d, torch.zeros_like(self.pbr_mesh.uvs_2d[:, [0]]), torch.ones_like(self.pbr_mesh.uvs_2d[:, [0]])], dim=-1).unsqueeze(0)
        faces = self.pbr_mesh.faces.to(dtype=torch.int32)
        faces_2d = self.pbr_mesh.faces_2d.to(dtype=torch.int32)

        ## rasterize 2D mesh
        rast_2d_out, _ = dr.rasterize(self.ctx, uvs_2d_clip, faces_2d, (height_2d, width_2d))
        mask_2d = (rast_2d_out[..., [3]] > 0)
        alpha_2d = (mask_2d > 0).to(dtype=torch.float32)
        tid_2d = rast_2d_out[..., [3]].to(dtype=torch.int64).sub(1)
        attrs_2d, _ = dr.interpolate(vertices_attrs.contiguous(), rast_2d_out, faces)
        face_normal_2d = self.pbr_mesh.normals.gather(0, torch.where(mask_2d, tid_2d, 0).reshape(-1, 1).repeat(1, 3)).reshape(1, height_2d, width_2d, 3)
        if perspective:
            rays_o_2d = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
            rays_d_2d = attrs_2d[..., :3] - rays_o_2d
        else:
            rays_d_2d = c2ws[:, :3, 2].neg().unsqueeze(1).unsqueeze(1)
            rays_o_2d = attrs_2d[..., :3] - (2.0 * math.sqrt(3.0)) * rays_d_2d
        rays_d_2d = torch.nn.functional.normalize(rays_d_2d, dim=-1)
        rays_o_2d, rays_d_2d = torch.broadcast_tensors(rays_o_2d, rays_d_2d)
        ray_normal_cos_2d = torch.nn.functional.cosine_similarity(rays_d_2d, face_normal_2d, dim=-1).unsqueeze(-1)
        ndc_2d, _ = dr.interpolate(vertices_ndc.contiguous(), rast_2d_out.repeat(batch_size, 1, 1, 1), faces)
        if image_attrs is not None:
            if alpha_attrs is not None:
                image_attrs = torch.cat([image_attrs, alpha_attrs], dim=-1)
            if grid_interpolate_mode in ['torch', 'pytorch']:
                image_attrs_2d = torch.nn.functional.grid_sample(
                    image_attrs.permute(0, 3, 1, 2),
                    ndc_2d,
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            elif grid_interpolate_mode in ['nvdiff', 'nvdiffrast']:
                image_attrs_2d = dr.texture(
                    image_attrs.contiguous(),
                    ndc_2d.mul(0.5).add(0.5),
                    filter_mode='linear',
                )
            else:
                raise NotImplementedError(f'grid_interpolate_mode {grid_interpolate_mode} is not supported')
            if alpha_attrs is not None:
                image_attrs_2d, image_alpha_2d = image_attrs_2d.split([image_attrs_2d.shape[-1] - 1, 1], dim=-1)
            else:
                image_alpha_2d = None
        else:
            image_attrs_2d = None
            image_alpha_2d = None

        ## process visibility and gradient
        rays_o_2d_valid = torch.masked_select(rays_o_2d, mask_2d).reshape(batch_size, -1, rays_o_2d.shape[-1])
        rays_d_2d_valid = torch.masked_select(rays_d_2d, mask_2d).reshape(batch_size, -1, rays_d_2d.shape[-1])
        tid_2d_valid = torch.masked_select(tid_2d.squeeze(-1), mask_2d.squeeze(-1))
        face_normal_2d_valid = torch.masked_select(face_normal_2d, mask_2d).reshape(-1, face_normal_2d.shape[-1])
        # NOTE: transmit rays from camera to mesh may cause misjudgments near mesh edges
        _, _, rays_tid_2d_valid, _, _ = self.pbr_mesh.optix.intersects_closest(rays_o_2d_valid, rays_d_2d_valid)
        rays_tid_2d_valid = rays_tid_2d_valid.to(dtype=torch.int64)
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
        if image_alpha_2d is not None:
            mask_2d_visiable = torch.logical_and(mask_2d_visiable, image_alpha_2d > 0.999)
        alpha_2d_visiable = mask_2d_visiable.to(dtype=torch.float32)

        ## reshape image to point cloud
        attrs_2d_valid = torch.masked_select(attrs_2d, mask_2d).reshape(-1, attrs_2d.shape[-1])
        face_normal_2d_valid = torch.masked_select(face_normal_2d, mask_2d).reshape(-1, face_normal_2d.shape[-1])
        alpha_2d_visiable_valid = torch.masked_select(mask_2d_visiable.any(dim=0, keepdim=True), mask_2d).reshape(-1, mask_2d_visiable.shape[-1]).to(dtype=torch.float32)
        point_cloud_2d = PointCloud(vertices=attrs_2d_valid[..., :3], normals=face_normal_2d_valid, alphas=alpha_2d_visiable_valid)
        attrs_2d_visiable = torch.masked_select(attrs_2d, mask_2d_visiable).reshape(-1, attrs_2d.shape[-1])
        face_normal_2d_visiable = torch.masked_select(face_normal_2d, mask_2d_visiable).reshape(-1, face_normal_2d.shape[-1])
        if image_attrs is not None:
            image_attrs_2d_visiable = torch.masked_select(image_attrs_2d, mask_2d_visiable).reshape(-1, image_attrs_2d.shape[-1])
        else:
            image_attrs_2d_visiable = None
        point_cloud_2d_visiable = PointCloud(vertices=attrs_2d_visiable[..., :3], normals=face_normal_2d_visiable, colors=image_attrs_2d_visiable)
        return {
            'mask_2d': mask_2d,
            'alpha_2d': alpha_2d,
            'mask_2d_visiable': mask_2d_visiable,
            'alpha_2d_visiable': alpha_2d_visiable,
            'point_cloud_2d': point_cloud_2d,
            'point_cloud_2d_visiable': point_cloud_2d_visiable,
        }

    def bake_mv_to_uv_kdtree(
        self,
        point_cloud_visiable: PointCloud,
        point_cloud_2d: PointCloud,
        mask_2d: torch.Tensor,
        mask_visiable:torch.Tensor,
        mask_2d_visiable:torch.Tensor,
        n_neighbors=32,
        n_neighbors_visiable=1,
        n_neighbors_invisiable=32,
        method='order_mean',
        inpainting=False,
    ):
        assert method in ['mean', 'mvpaint', 'order_mean']
        if method == 'mean':
            if not inpainting:
                score, index = knn(point_cloud_visiable.vertices, point_cloud_2d.vertices, k=n_neighbors, batch_size=1024*1024)
                colors = point_cloud_visiable.colors[index, :]
                weighted_colors = colors.mean(dim=-2)
            else:
                weighted_colors = self.query_field(point_cloud_visiable.vertices, point_cloud_visiable.colors, point_cloud_2d.vertices)
        elif method == 'mvpaint':
            score, index = knn(point_cloud_visiable.vertices, point_cloud_2d.vertices, k=n_neighbors, batch_size=1024*1024)
            # NOTE: https://arxiv.org/abs/2411.02336v1, Section 3.2
            weight = torch.nn.functional.normalize(score.reciprocal().nan_to_num(nan=0.0), p=1, dim=-1) * \
                torch.nn.functional.cosine_similarity(point_cloud_visiable.normals[index, :], point_cloud_2d.normals.unsqueeze(-2), dim=-1)
            weight = weight.unsqueeze(-1)
            weight_sum = weight.sum(dim=-2, keepdim=False)
            colors = point_cloud_visiable.colors[index, :]
            weighted_colors = (colors * weight).sum(dim=-2) / weight_sum
            weighted_colors = torch.nan_to_num(weighted_colors, nan=0.0, posinf=0.0, neginf=0.0)
        elif method == 'order_mean':
            point_clouds_visiable = point_cloud_visiable.split(mask_visiable.sum([-3, -2, -1], keepdim=False))
            weighted_colors = torch.zeros((point_cloud_2d.vertices.shape[0], point_cloud_visiable.colors.shape[-1]), dtype=point_cloud_2d.vertices.dtype, device=point_cloud_2d.vertices.device)
            mask_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], mask_2d_visiable.shape[-1]), dtype=mask_2d_visiable.dtype, device=mask_2d_visiable.device)
            for i in self.index:
                mask_2d_extra = torch.logical_and(mask_2d_current.logical_not(), mask_2d_visiable[[i]])
                visiable_vertices_mask = torch.masked_select(mask_2d_extra, mask_2d).reshape(-1, mask_2d_extra.shape[-1])
                visiable_vertices = torch.masked_select(point_cloud_2d.vertices, visiable_vertices_mask).reshape(-1, point_cloud_2d.vertices.shape[-1])
                score, index = knn(point_clouds_visiable[i].vertices, visiable_vertices, k=n_neighbors_visiable, batch_size=1024*1024)
                colors = point_clouds_visiable[i].colors[index, :]
                weighted_colors_current = colors.mean(dim=-2)
                weighted_colors.masked_scatter_(visiable_vertices_mask, weighted_colors_current)
                mask_2d_current = torch.logical_or(mask_2d_current, mask_2d_extra)
            visiable_vertices_mask = torch.masked_select(mask_2d_current, mask_2d).reshape(-1, mask_2d_current.shape[-1])
            visiable_vertices = torch.masked_select(point_cloud_2d.vertices, visiable_vertices_mask).reshape(-1, point_cloud_2d.vertices.shape[-1])
            visiable_colors = torch.masked_select(weighted_colors, visiable_vertices_mask).reshape(-1, weighted_colors_current.shape[-1])
            invisiable_vertices_mask = torch.masked_select(mask_2d_current.logical_not(), mask_2d).reshape(-1, mask_2d_current.shape[-1])
            invisiable_vertices = torch.masked_select(point_cloud_2d.vertices, invisiable_vertices_mask).reshape(-1, point_cloud_2d.vertices.shape[-1])
            if not inpainting:
                score, index = knn(visiable_vertices, invisiable_vertices, k=n_neighbors_invisiable, batch_size=1024*1024)
                colors = visiable_colors[index, :]
                weighted_colors_current = colors.mean(dim=-2)
            else:
                weighted_colors_current = self.query_field(visiable_vertices, visiable_colors, invisiable_vertices)
            weighted_colors.masked_scatter_(invisiable_vertices_mask, weighted_colors_current)
        else:
            raise NotImplementedError(f'method {method} is not supported')
        point_cloud_2d.colors = weighted_colors
        color_2d = torch.zeros((mask_2d.shape[0], mask_2d.shape[1], mask_2d.shape[2], weighted_colors.shape[-1]), dtype=weighted_colors.dtype, device=weighted_colors.device)
        color_2d.masked_scatter_(mask_2d, weighted_colors)
        color_2d = pull_push(color_2d.permute(0, 3, 1, 2), mask_2d.permute(0, 3, 1, 2))[0].permute(0, 2, 3, 1)
        bake_out = {
            'point_cloud_2d': point_cloud_2d,
            'color_2d': color_2d,
        }
        return bake_out

    def get_boundary_mask(
        self,
        mask_2d:torch.Tensor,
        kernel_size=3,
    ):
        alpha_2d = mask_2d.to(dtype=torch.float32)
        mask_2d_boundary_inner = (alpha_2d - (1.0 - torch.nn.functional.max_pool2d(1.0 - alpha_2d.permute(0, 3, 1, 2), 2 * (kernel_size // 2) + 1, 1, kernel_size // 2)).permute(0, 2, 3, 1) > 0)
        mask_2d_boundary_outer = (torch.nn.functional.max_pool2d(alpha_2d.permute(0, 3, 1, 2), 2 * (kernel_size // 2) + 1, 1, kernel_size // 2).permute(0, 2, 3, 1) - alpha_2d > 0)
        mask_2d_boundary = torch.logical_or(mask_2d_boundary_inner, mask_2d_boundary_outer)
        return mask_2d_boundary

    def get_boundary_mask_and_index(
        self,
        point_cloud_2d: PointCloud,
        mask_2d:torch.Tensor,
        kernel_size=3,
        n_neighbors_2d=1,
        n_neighbors_3d=8,
    ):
        '''
        index_boundary_3d: [N_outer, N_2d, N_3d]
        '''
        height_2d, width_2d, _ = mask_2d.shape[-3:]
        device = mask_2d.device
        alpha_2d = mask_2d.to(dtype=torch.float32)
        mask_2d_boundary_inner = (alpha_2d - (1.0 - torch.nn.functional.max_pool2d(1.0 - alpha_2d.permute(0, 3, 1, 2), 2 * (kernel_size // 2) + 1, 1, kernel_size // 2)).permute(0, 2, 3, 1) > 0)
        mask_2d_boundary_outer = (torch.nn.functional.max_pool2d(alpha_2d.permute(0, 3, 1, 2), 2 * (kernel_size // 2) + 1, 1, kernel_size // 2).permute(0, 2, 3, 1) - alpha_2d > 0)
        vertices_mask_boundary_inner = torch.masked_select(mask_2d_boundary_inner, mask_2d).reshape(-1, mask_2d_boundary_inner.shape[-1])
        vertices_boundary_inner = torch.masked_select(point_cloud_2d.vertices, vertices_mask_boundary_inner).reshape(-1, point_cloud_2d.vertices.shape[-1])
        score_boundary, index_boundary = knn(vertices_boundary_inner, vertices_boundary_inner, k=n_neighbors_3d+1, batch_size=1024)
        score_boundary = score_boundary[:, 1:]
        index_boundary = index_boundary[:, 1:]
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / height_2d, 1.0 - 1.0 / height_2d, height_2d, dtype=torch.float32, device=device),
            torch.linspace(-1.0 + 1.0 / width_2d, 1.0 - 1.0 / width_2d, width_2d, dtype=torch.float32, device=device),
            indexing='ij',
        )
        gxy = torch.stack([gx, gy], dim=-1)
        vertices_2d = torch.masked_select(gxy, mask_2d).reshape(-1, gxy.shape[-1])
        vertices_2d_boundary_inner = torch.masked_select(vertices_2d, vertices_mask_boundary_inner).reshape(-1, vertices_2d.shape[-1])
        vertices_2d_boundary_outer = torch.masked_select(gxy, mask_2d_boundary_outer).reshape(-1, gxy.shape[-1])
        score_boundary_2d, index_boundary_2d = knn(vertices_2d_boundary_inner, vertices_2d_boundary_outer, k=n_neighbors_2d, batch_size=1024)
        index_boundary_3d = index_boundary[index_boundary_2d, :]
        return mask_2d_boundary_inner, mask_2d_boundary_outer, index_boundary_3d

    def bake_mv_to_uv_reproject_blending(
        self,
        point_cloud_2d_visiable:PointCloud,
        point_cloud_2d: PointCloud,
        mask_2d_visiable:torch.Tensor,
        mask_2d:torch.Tensor,
        n_erode_pix_blending=0,
        n_erode_pix_seamless=0,
        method='3D',
        n_radius=1,
        n_neighbors=8,
        kernel_size_blending=3,
        kernel_size_seamless=11,
        inpainting=False,
    ):
        '''
        bake mv to uv with reprojection + blending
        '''
        assert mask_2d_visiable.shape[0] == len(self.index)
        assert method in ['2D', '3D']
        colors = point_cloud_2d_visiable.colors
        colors_2d = torch.zeros((mask_2d_visiable.shape[0], mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], colors.shape[-1]), dtype=colors.dtype, device=colors.device)
        colors_2d.masked_scatter_(mask_2d_visiable, colors)

        ## create base color
        color_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], colors.shape[-1]), dtype=colors.dtype, device=colors.device)
        mask_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], mask_2d_visiable.shape[-1]), dtype=mask_2d_visiable.dtype, device=mask_2d_visiable.device)
        for i in self.index:
            mask_2d_extra = torch.logical_and(mask_2d_current.logical_not(), mask_2d_visiable[[i]])
            color_2d_current.masked_scatter_(mask_2d_extra, torch.masked_select(colors_2d[[i]], mask_2d_extra))
            mask_2d_current = torch.logical_or(mask_2d_current, mask_2d_extra)
        vertices_mask_visiable = torch.masked_select(mask_2d_current, mask_2d).reshape(-1, mask_2d_current.shape[-1])
        vertices_visiable = torch.masked_select(point_cloud_2d.vertices, vertices_mask_visiable).reshape(-1, point_cloud_2d.vertices.shape[-1])
        colors_visiable = torch.masked_select(color_2d_current, mask_2d_current).reshape(-1, color_2d_current.shape[-1])
        vertices_invisiable = torch.masked_select(point_cloud_2d.vertices, vertices_mask_visiable.logical_not()).reshape(-1, point_cloud_2d.vertices.shape[-1])
        if not inpainting:
            score_invisiable, index_invisiable = knn(vertices_visiable, vertices_invisiable, k=1, batch_size=1024)
            color_invisiable = colors_visiable[index_invisiable.squeeze(-1), :]
        else:
            color_invisiable = self.query_field(vertices_visiable, colors_visiable, vertices_invisiable)
        color_2d_current.masked_scatter_(torch.logical_and(mask_2d_current.logical_not(), mask_2d), color_invisiable)

        ## blending colors with base color
        if method == '3D':
            mask_2d_boundary_inner, mask_2d_boundary_outer, index_boundary_3d = self.get_boundary_mask_and_index(
                point_cloud_2d=point_cloud_2d,
                mask_2d=mask_2d,
                kernel_size=kernel_size_blending,
                n_neighbors_2d=1,
                n_neighbors_3d=n_neighbors,
            )
        for i in self.index[::-1]:
            color_2d_current = torch.where(mask_2d_current, color_2d_current, 0.0)
            if method == '2D':
                color_2d_current = image_outpainting(color_2d_current, mask_2d.logical_not(), n_radius=n_radius)
            elif method == '3D':
                color_2d_boundary_inner = torch.masked_select(color_2d_current, mask_2d_boundary_inner).reshape(-1, color_2d_current.shape[-1])
                color_2d_boundary_outer = color_2d_boundary_inner[index_boundary_3d.squeeze(-2), :]
                color_2d_boundary_outer = color_2d_boundary_outer.mean(dim=1, keepdim=False)
                color_2d_current.masked_scatter_(mask_2d_boundary_outer, color_2d_boundary_outer)
            else:
                raise NotImplementedError(f'method {method} is not supported')
            color_2d_current = image_fusion(colors_2d[[i]], color_2d_current, mask_2d_visiable[[i]], n_erode_pix=n_erode_pix_blending)
            # color_2d_result = torch.where(mask_2d_visiable[[i]], color_2d_result, color_2d_current)
            color_visiable = torch.masked_select(color_2d_current, mask_2d_current).reshape(-1, color_2d_current.shape[-1])
            color_invisiable = color_visiable[index_invisiable.squeeze(-1), :]
            color_2d_current.masked_scatter_(torch.logical_and(mask_2d_current.logical_not(), mask_2d), color_invisiable)

        ## blending seamless colors
        color_2d_current = torch.where(mask_2d, color_2d_current, 0.0)
        color_2d_current_2d = image_outpainting(color_2d_current, mask_2d.logical_not(), n_radius=n_radius)
        mask_2d_boundary_inner, mask_2d_boundary_outer, index_boundary_3d = self.get_boundary_mask_and_index(
            point_cloud_2d=point_cloud_2d,
            mask_2d=mask_2d,
            kernel_size=kernel_size_seamless,
            n_neighbors_2d=1,
            n_neighbors_3d=n_neighbors,
        )
        color_2d_boundary_inner = torch.masked_select(color_2d_current, mask_2d_boundary_inner).reshape(-1, color_2d_current.shape[-1])
        color_2d_boundary_outer = color_2d_boundary_inner[index_boundary_3d.squeeze(-2), :]
        color_2d_boundary_outer = color_2d_boundary_outer.mean(dim=1, keepdim=False)
        color_2d_current.masked_scatter_(mask_2d_boundary_outer, color_2d_boundary_outer)
        color_2d_current = image_outpainting(color_2d_current, mask_2d, n_radius=-1)
        color_2d_current = image_fusion(color_2d_current_2d, color_2d_current, mask_2d, n_erode_pix=n_erode_pix_seamless)

        point_cloud_2d.colors = torch.masked_select(color_2d_current, mask_2d).reshape(-1, color_2d_current.shape[-1])
        color_2d = pull_push(color_2d_current.permute(0, 3, 1, 2), mask_2d.permute(0, 3, 1, 2))[0].permute(0, 2, 3, 1)
        bake_out = {
            'point_cloud_2d': point_cloud_2d,
            'colors_2d': colors_2d,
            'color_2d': color_2d,
        }
        return bake_out

    def bake_mv_to_uv_reproject_blur(
        self,
        point_cloud_2d_visiable:PointCloud,
        point_cloud_2d: PointCloud,
        mask_2d_visiable:torch.Tensor,
        mask_2d:torch.Tensor,
        method='lens',
        kernel_size_boundary=3,
        kernel_size_boundary_blur=3,
        kernel_size_blur=5,
        inpainting=False,
    ):
        '''
        bake mv to uv with reprojection + blur
        '''
        assert mask_2d_visiable.shape[0] == len(self.index)
        assert method in ['gaussian', 'lens']
        colors = point_cloud_2d_visiable.colors
        colors_2d = torch.zeros((mask_2d_visiable.shape[0], mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], colors.shape[-1]), dtype=colors.dtype, device=colors.device)
        colors_2d.masked_scatter_(mask_2d_visiable, colors)

        ## create base color
        color_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], colors.shape[-1]), dtype=colors.dtype, device=colors.device)
        mask_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], mask_2d_visiable.shape[-1]), dtype=mask_2d_visiable.dtype, device=mask_2d_visiable.device)
        boundary_2d_current = torch.zeros((1, mask_2d_visiable.shape[1], mask_2d_visiable.shape[2], mask_2d_visiable.shape[-1]), dtype=mask_2d_visiable.dtype, device=mask_2d_visiable.device)
        for i in self.index:
            mask_2d_extra = torch.logical_and(mask_2d_current.logical_not(), mask_2d_visiable[[i]])
            color_2d_current.masked_scatter_(mask_2d_extra, torch.masked_select(colors_2d[[i]], mask_2d_extra))
            mask_2d_current = torch.logical_or(mask_2d_current, mask_2d_extra)
            boundary_2d_current = torch.logical_or(boundary_2d_current, self.get_boundary_mask(mask_2d_extra, kernel_size=kernel_size_boundary))
        boundary_2d_current = (torch.max_pool2d(boundary_2d_current.to(dtype=torch.float32).permute(0, 3, 1, 2), 2 * (kernel_size_boundary_blur // 2) + 1, 1, kernel_size_boundary_blur // 2).permute(0, 2, 3, 1) > 0)
        boundary_2d_current = torch.logical_and((1.0 - torch.max_pool2d(1.0 - mask_2d.to(dtype=torch.float32).permute(0, 3, 1, 2), 2 * (kernel_size_boundary_blur // 2) + 5, 1, kernel_size_boundary_blur // 2 + 2).permute(0, 2, 3, 1) > 0), boundary_2d_current)
        vertices_mask_visiable = torch.masked_select(mask_2d_current, mask_2d).reshape(-1, mask_2d_current.shape[-1])
        vertices_visiable = torch.masked_select(point_cloud_2d.vertices, vertices_mask_visiable).reshape(-1, point_cloud_2d.vertices.shape[-1])
        colors_visiable = torch.masked_select(color_2d_current, mask_2d_current).reshape(-1, color_2d_current.shape[-1])
        vertices_invisiable = torch.masked_select(point_cloud_2d.vertices, vertices_mask_visiable.logical_not()).reshape(-1, point_cloud_2d.vertices.shape[-1])
        if not inpainting:
            score_invisiable, index_invisiable = knn(vertices_visiable, vertices_invisiable, k=1, batch_size=1024)
            color_invisiable = colors_visiable[index_invisiable.squeeze(-1), :]
        else:
            color_invisiable = self.query_field(vertices_visiable, colors_visiable, vertices_invisiable)
        color_2d_current.masked_scatter_(torch.logical_and(mask_2d_current.logical_not(), mask_2d), color_invisiable)

        ## blur mv seams
        if method == 'gaussian':
            color_2d_current_blur = gaussian_blur(color_2d_current.permute(0, 3, 1, 2), (kernel_size_blur, kernel_size_blur)).permute(0, 2, 3, 1)
        elif method == 'lens':
            color_2d_current_blur = lens_blur_torch(color_2d_current.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            raise NotImplementedError(f'method {method} is not supported')
        color_2d_current = torch.where(boundary_2d_current, color_2d_current_blur, color_2d_current)

        point_cloud_2d.colors = torch.masked_select(color_2d_current, mask_2d).reshape(-1, color_2d_current.shape[-1])
        color_2d = pull_push(color_2d_current.permute(0, 3, 1, 2), mask_2d.permute(0, 3, 1, 2))[0].permute(0, 2, 3, 1)
        bake_out = {
            'point_cloud_2d': point_cloud_2d,
            'colors_2d': colors_2d,
            'color_2d': color_2d,
        }
        return bake_out

    def infer(
        self, 
        blank_mesh:Union[trimesh.Trimesh, str],
        c2ws:torch.Tensor, 
        intrinsics:torch.Tensor, 
        image_attrs:torch.Tensor, 
        H=512, W=512, H2D=2048, W2D=2048, 
        perspective=True,
        grad_norm_threhold=0.20,
        ray_normal_angle_threhold=115.0,
        grid_interpolate_mode='torch',
        method='reproject',
        kdtree_n_neighbors=32,
        kdtree_n_neighbors_visiable=1,
        kdtree_n_neighbors_invisiable=32,
        kdtree_method='order_mean',
        kdtree_inpainting=False,
        reproject_method='lens',
        reproject_kernel_size_boundary=3,
        reproject_kernel_size_boundary_blur=3,
        reproject_kernel_size_blur=5,
        reproject_inpainting = False,
        return_mv_reproject_uv = False,
        filt_gradient_points = True
    ) -> trimesh.Trimesh:
        assert method in ['kdtree', 'reproject']
        assert image_attrs.shape[-1] in [3, 9]
        render_mv_out = self.mv_to_pcd(
            c2ws, 
            intrinsics, 
            (H, W), 
            image_attrs=image_attrs, 
            perspective=perspective, 
            grad_norm_threhold=grad_norm_threhold,
            ray_normal_angle_threhold=ray_normal_angle_threhold,
            filt_gradient_points=filt_gradient_points,
        )
        render_uv_out = self.uv_to_pcd(
            c2ws, 
            intrinsics, 
            (H2D, W2D), 
            image_attrs=image_attrs, 
            alpha_attrs=render_mv_out['alpha_visiable'], 
            perspective=perspective, 
            ray_normal_angle_threhold=ray_normal_angle_threhold,
            grid_interpolate_mode=grid_interpolate_mode,
        )
        if method == 'kdtree':
            bake_out = self.bake_mv_to_uv_kdtree(
                render_mv_out['point_cloud_visiable'], 
                render_uv_out['point_cloud_2d'], 
                render_uv_out['mask_2d'], 
                render_mv_out['mask_visiable'], 
                render_uv_out['mask_2d_visiable'],
                n_neighbors=kdtree_n_neighbors,
                n_neighbors_visiable=kdtree_n_neighbors_visiable,
                n_neighbors_invisiable=kdtree_n_neighbors_invisiable,
                method=kdtree_method,
                inpainting=kdtree_inpainting,
            )
        elif method == 'reproject':
            bake_out = self.bake_mv_to_uv_reproject_blur(
                render_uv_out['point_cloud_2d_visiable'], 
                render_uv_out['point_cloud_2d'], 
                render_uv_out['mask_2d_visiable'], 
                render_uv_out['mask_2d'],
                method=reproject_method,
                kernel_size_boundary=reproject_kernel_size_boundary,
                kernel_size_boundary_blur=reproject_kernel_size_boundary_blur,
                kernel_size_blur=reproject_kernel_size_blur,
                inpainting=reproject_inpainting,
            )
        else:
            raise NotImplementedError(f'method {method} is not supported')
        if image_attrs.shape[-1] == 3:
            textured_mesh = link_rgb_to_mesh(
                src_path=blank_mesh, 
                rgb_path=tensor_to_image(bake_out['color_2d'][0]),
                dst_path=None,
            )
        elif image_attrs.shape[-1] == 9:
            textured_mesh = link_pbr_to_mesh(
                src_path=blank_mesh, 
                albedo_path=tensor_to_image(bake_out['color_2d'][0, :, :, 0:3]),
                metallic_roughness_path=tensor_to_image(bake_out['color_2d'][0, :, :, 3:6]),
                bump_path=tensor_to_image(bake_out['color_2d'][0, :, :, 6:9]),
                dst_path=None,
            )
        else:
            raise NotImplementedError(f'shape {image_attrs.shape} is not supported')

        return textured_mesh, render_uv_out['mask_2d_visiable'], render_uv_out['mask_2d'], bake_out['color_2d']


#### some test functions ####


def test_gt():
    from ...camera.generator import generate_orbit_views_c2ws, generate_box_views_c2ws, generate_intrinsics
    from ...geometry.uv.uv_atlas import preprocess_textured_mesh
    from ...geometry.uv.uv_kernel import preprocess_blank_mesh
    from ...video.export_nvdiffrast_video import VideoExporter

    tetured_mesh_path = 'test_data/texture_reproject_gt/26f7a22490324a25b0bc7601e3929dc8.glb'
    processed_blank_mesh_path = 'test_data/texture_reproject_gt/processed_blank_mesh.obj'
    processed_textured_mesh_path = 'test_data/texture_reproject_gt/processed_textured_mesh.obj'
    render_results_path = 'test_data/texture_reproject_gt/render_results.mp4'
    render_results_dir = 'test_data/texture_reproject_gt/render_results_frames'
    os.makedirs(render_results_dir, exist_ok=True)
    result_dir = 'test_result/test_geometry_renderer/gt'
    os.makedirs(result_dir, exist_ok=True)
    video_exporter = VideoExporter()

    N, H, W, H2D, W2D = 6, 512, 512, 2048, 2048
    perspective = False
    c2ws = generate_box_views_c2ws(radius=2.8)
    if perspective:
        intrinsics = generate_intrinsics(49.1, 49.1, fov=True, degree=True)
    else:
        intrinsics = generate_intrinsics(0.85, 0.85, fov=False, degree=False)
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')

    preprocess_blank_mesh(tetured_mesh_path, processed_blank_mesh_path)
    preprocess_textured_mesh(tetured_mesh_path, processed_textured_mesh_path)
    video_exporter.export_orbit_video(
        mesh_obj=processed_textured_mesh_path,
        video_path=render_results_path,
        enhance_mode='box',
        perspective=perspective,
        save_frames=True,
    )
    image_attrs = image_to_tensor([Image.open(os.path.join(render_results_dir, f'{i:04d}.png')).convert('RGB').resize((W, H)) for i in range(6)])
    
    renderer = NVDiffRendererInverse.from_file(pbr_mesh_path=processed_blank_mesh_path)
    renderer.pbr_mesh.scale_to_bbox()
    textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='kdtree')
    textured_mesh.export(os.path.join(result_dir, f'textured_mesh_kdtree.glb'))
    textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='reproject')
    textured_mesh.export(os.path.join(result_dir, f'textured_mesh_reproject.glb'))

def test_pred():
    test_root = '/home/chenxiao/code/MVDiffusion/test_data/online_cases/d2b57ca9-f299-44d8-b265-b28d4db5f8d7_outputs_wo_inpainting/cache'
    processed_blank_mesh_path = os.path.join(test_root, 'processed_mesh.obj')
    camera_info_path = os.path.join(test_root, 'camera_info.pth')
    render_results_path = os.path.join(test_root, 'mv_rgb.png')
    result_dir = 'test_result/test_geometry_renderer/pred'
    os.makedirs(result_dir, exist_ok=True)

    N, H, W, H2D, W2D = 6, 512, 512, 2048, 2048
    camera_info = torch.load(camera_info_path, weights_only=False)
    perspective = camera_info['perspective']
    c2ws = camera_info['c2ws']
    intrinsics = camera_info['intrinsics']
    c2ws = c2ws.to(device='cuda')
    intrinsics = intrinsics.to(device='cuda')
    image_attrs = image_to_tensor(Image.open(render_results_path).convert('RGB'))
    image_attrs = image_attrs.reshape(2, H, 3, W, 3).permute(0, 2, 1, 3, 4).reshape(6, H, W, 3)
    image_attrs = image_attrs.repeat(1, 1, 1, 3)  # NOTE: debug for albedo/metallic-roughness/bump

    renderer = NVDiffRendererInverse.from_file(pbr_mesh_path=processed_blank_mesh_path)
    # renderer.pbr_mesh.scale_to_bbox()
    textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='kdtree')
    textured_mesh.export(os.path.join(result_dir, f'textured_mesh_kdtree.glb'))
    textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='reproject')
    textured_mesh.export(os.path.join(result_dir, f'textured_mesh_reproject.glb'))

def batch_test_pred():
    test_root = '/mnt/jfs/chenxiao/mvdiffusion_test_results/20250321/outputs_rgb_6_views/*/cache'
    for test_root in tqdm(glob(test_root)):
        processed_blank_mesh_path = os.path.join(test_root, 'processed_mesh.obj')
        camera_info_path = os.path.join(test_root, 'camera_info.pth')
        render_results_path = os.path.join(test_root, 'mv_rgb.png')
        result_dir = os.path.join(os.path.dirname(test_root), 'test_texture')
        os.makedirs(result_dir, exist_ok=True)

        N, H, W, H2D, W2D = 6, 512, 512, 2048, 2048
        camera_info = torch.load(camera_info_path, weights_only=False)
        perspective = camera_info['perspective']
        c2ws = camera_info['c2ws']
        intrinsics = camera_info['intrinsics']
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        image_attrs = image_to_tensor(Image.open(render_results_path).convert('RGB'))
        image_attrs = image_attrs.reshape(2, H, 3, W, 3).permute(0, 2, 1, 3, 4).reshape(6, H, W, 3)

        renderer = NVDiffRendererInverse.from_file(pbr_mesh_path=processed_blank_mesh_path)
        # renderer.pbr_mesh.scale_to_bbox()
        textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='kdtree')
        textured_mesh.export(os.path.join(result_dir, f'textured_mesh_kdtree.glb'))
        textured_mesh = renderer.infer(processed_blank_mesh_path, c2ws, intrinsics, image_attrs, perspective=perspective, method='reproject')
        textured_mesh.export(os.path.join(result_dir, f'textured_mesh_reproject.glb'))

