from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import trimesh

from ..geometry.utils import to_tensor_f, to_tensor_i, to_array_f


class PointCloud:
    def __init__(
        self, 
        vertices:torch.Tensor, 
        normals:Optional[torch.Tensor]=None, 
        colors:Optional[torch.Tensor]=None, 
        alphas:Optional[torch.Tensor]=None,
        attributes:Optional[torch.Tensor]=None, 
    ):
        self.vertices = to_tensor_f(vertices)
        self.normals = to_tensor_f(normals) if normals is not None else None
        self.colors = to_tensor_f(colors) if colors is not None else None
        self.alphas = to_tensor_f(alphas) if alphas is not None else None
        self.attributes = to_tensor_f(attributes) if attributes is not None else None

    @classmethod
    def from_trimesh(cls, mesh:Union[trimesh.PointCloud, trimesh.Trimesh], ignore_alpha=True) -> "PointCloud":
        vertices = to_tensor_f(mesh.vertices)
        normals = mesh._cache.get('vertex_normals', None)
        if normals is not None:
            normals = to_tensor_f(normals)
        colors = to_tensor_f(mesh.visual.vertex_colors).div(255.0)
        if colors.shape[-1] == 3:
            alphas = None
        elif colors.shape[-1] == 3:
            colors, alphas = colors.split([3, 1], dim=-1)
            if ignore_alpha:
                alphas = None
        else:
            raise NotImplementedError(f'shape of color {colors.shape} is not supported')
        return cls(vertices=vertices, normals=normals, colors=colors, alphas=alphas)

    def to_trimesh(self, ignore_alpha=True) -> Union[trimesh.PointCloud, trimesh.Trimesh]:
        vertices = to_array_f(self.vertices)
        if self.normals is not None:
            normals = to_array_f(self.normals)
        else:
            normals = None
        if not ignore_alpha:
            if self.colors is not None and self.alphas is not None:
                colors = torch.cat([self.colors, self.alphas], dim=-1).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
            elif self.colors is not None and self.alphas is None:
                colors = self.colors.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
            elif self.colors is None and self.alphas is not None:
                colors = self.alphas.repeat(1, 3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
            elif self.colors is None and self.alphas is None:
                colors = None
        else:
            if self.colors is not None:
                colors = self.colors.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
            else:
                colors = None
        if normals is None:
            return trimesh.PointCloud(vertices=vertices, colors=colors, process=False)
        else:
            return trimesh.Trimesh(vertices=vertices, vertex_normals=normals, vertex_colors=colors, process=False)

    def split(self, split_size:Tuple[int]) -> List["PointCloud"]:
        '''
        split_size: [N,]
        '''
        split_size = tuple(split_size)
        N = len(split_size)
        vertices_list = self.vertices.split(split_size, dim=0)
        normals_list = self.normals.split(split_size, dim=0) if self.normals is not None else [None] * N
        colors_list = self.colors.split(split_size, dim=0) if self.colors is not None else [None] * N
        alphas_list = self.alphas.split(split_size, dim=0) if self.alphas is not None else [None] * N
        attributes_list = self.attributes.split(split_size, dim=0) if self.attributes is not None else [None] * N
        return [
            PointCloud(
                vertices=vertices,
                normals=normals,
                colors=colors,
                alphas=alphas,
                attributes=attributes,
            )
            for (
                vertices, normals, colors, alphas, attributes,
            ) in zip(
                vertices_list, normals_list, colors_list, alphas_list, attributes_list,
            )
        ]


class PointClouds:
    def __init__(self, point_cloud_list:Union[List[PointCloud], List["PointClouds"]]):
        vertices_list = []
        vertices_idx_list = []
        V_interval = []
        V_idx = 0
        for idx, point_cloud in enumerate(point_cloud_list):
            vertices = point_cloud.vertices
            V_cnt = vertices.shape[0]
            vertices_list += [vertices]
            V_interval.append([V_idx, V_idx + V_cnt, V_cnt])
            V_idx += V_cnt
        self.point_cloud_list = point_cloud_list
        self.vertices = torch.cat(vertices_list, dim=0)
        self.V_interval = to_tensor_i(V_interval)
        self.vertices_idx = torch.cat(vertices_idx_list, dim=0)


