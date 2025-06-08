'''
Scene/Multi-Mesh Data Structure based on pytorch
'''

import os
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial

from ..raytracing import RayTracing
from .trimesh_utils import parse_texture_visuals
from ..geometry.utils import to_tensor_f, to_tensor_i


class PBRDefault:
    albedo:torch.Tensor = to_tensor_f([0.0, 0.0, 0.0])
    metallic:torch.Tensor = to_tensor_f([0.0])
    roughness:torch.Tensor = to_tensor_f([1.0])
    bump:torch.Tensor = to_tensor_f([0.0, 0.0, 0.0])


class PBRMesh:
    def __init__(
        self,
        vertices:torch.Tensor,
        faces:torch.Tensor,
        uvs_2d:Optional[torch.Tensor]=None,
        faces_2d:Optional[torch.Tensor]=None,
        albedo:Optional[torch.Tensor]=None,
        metallic:Optional[torch.Tensor]=None,
        roughness:Optional[torch.Tensor]=None,
        bump:Optional[torch.Tensor]=None,
    ):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        uvs_2d: [V_2d, 2], float32, optional
        faces_2d: [F, 3], int64, optional
        albedo: [H, W, 3] or [V, 3] or [3,], float32, optional
        metallic: [H, W, 1] or [V, 1] or [1,], float32, optional
        roughness: [H, W, 1] or [V, 1] or [1,], float32, optional
        bump: [H, W, 3] or [V, 3] or [3,], float32, optional
        '''
        self.vertices = to_tensor_f(vertices)
        self.faces = to_tensor_i(faces)
        self.areas = torch.linalg.cross(self.vertices[self.faces[:, 1], :] - self.vertices[self.faces[:, 0], :], self.vertices[self.faces[:, 2], :] - self.vertices[self.faces[:, 0], :], dim=-1)
        self.normals = torch.nn.functional.normalize(self.areas, dim=-1)
        self._vertex_normals = None
        self._optix = None
        self.uvs_2d = to_tensor_f(uvs_2d) if uvs_2d is not None else None
        if faces_2d is not None:
            self.faces_2d = to_tensor_i(faces_2d)
            assert self.faces.shape[0] == self.faces_2d.shape[0], f'invaid faces: {self.faces.shape[0]}, {self.faces_2d.shape[0]}'
        else:
            self.faces_2d = None
        self.albedo = to_tensor_f(albedo) if albedo is not None else None
        self.metallic = to_tensor_f(metallic) if metallic is not None else None
        self.roughness = to_tensor_f(roughness) if roughness is not None else None
        self.bump = to_tensor_f(bump) if bump is not None else None

    @property
    def vertex_normals(self):
        if self._vertex_normals is None:
            vertex_normals = torch.zeros((self.vertices.shape[0], 3, 3),dtype=self.vertices.dtype, device=self.vertices.device)
            vertex_normals.scatter_add_(0, self.faces.unsqueeze(-1).expand(-1, -1, 3), self.areas.unsqueeze(1).expand(-1, 3, -1))
            vertex_normals = torch.nn.functional.normalize(vertex_normals.mean(dim=1), dim=-1)
            self._vertex_normals = vertex_normals
        return self._vertex_normals

    @property
    def optix(self):
        if self._optix is None:
            self._optix = RayTracing(self.vertices, self.faces)
        return self._optix

    def scale_to_bbox(self, largest=True, scale=1.0):
        '''
        largest: align largest or shortest length of bbox to 1
        scale: do extra scaling wrt origin after aligned to bbox
        '''
        aaa = self.vertices.min(dim=0).values
        bbb = self.vertices.max(dim=0).values
        if largest:
            scale = (bbb - aaa).max() / (2.0 * scale)
        else:
            scale = (bbb - aaa).min() / (2.0 * scale)
        transform = torch.eye(4, dtype=self.vertices.dtype, device=self.vertices.device)
        transform[[0, 1, 2], [0, 1, 2]] = 1.0 / scale
        transform[:3, 3] = - (aaa + bbb) / (2.0 * scale)
        vertices_homo = torch.cat([self.vertices, torch.ones_like(self.vertices[:, [0]])], dim=-1)
        vertices_homo = torch.matmul(vertices_homo, transform.T)
        self.vertices = vertices_homo[:, :3].contiguous()
        normals = torch.matmul(self.normals, transform[:3, :3].T)
        self.normals = torch.nn.functional.normalize(normals, dim=-1)
        if self._vertex_normals is not None:
            _vertex_normals = torch.matmul(self._vertex_normals, transform[:3, :3].T)
            self._vertex_normals = torch.nn.functional.normalize(_vertex_normals, dim=-1)
        if self._optix is not None:
            self._optix.update_raw(vertices=self.vertices, faces=self.faces)
        return self

    def __call__(self, face_idx:torch.Tensor, face_uvw:torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        face_idx: [N,], int64
        face_uvw: [N, 3], float32
        out: Dict[str, [N, C]], float32
        '''
        N = face_idx.shape[0]
        faces, uvs = None, None
        out = dict()
        for k in ['albedo', 'metallic', 'roughness', 'bump']:
            v = getattr(self, k, None)
            if v is None:
                v = getattr(PBRDefault, k).clone()
            if v.ndim == 1:
                out[k] = v.unsqueeze(0).expand(N, -1)
            elif v.ndim == 2:
                if faces is None:
                    faces = self.faces[face_idx, :]
                out[k] = torch.sum(v[faces, :] * face_uvw.unsqueeze(-1), dim=1)
            elif v.ndim == 3:
                if uvs is None:
                    uvs = torch.sum(self.uvs_2d[self.faces_2d[face_idx, :], :] * face_uvw.unsqueeze(-1), dim=1)
                out[k] = torch.nn.functional.grid_sample(
                    v.unsqueeze(0).permute(0, 3, 1, 2),
                    uvs.unsqueeze(0).unsqueeze(0),
                    align_corners=False,
                    padding_mode='reflection',
                ).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            else:
                raise NotImplementedError(f'invalid shape of {k}: {v.shape}')
        return out

    def __repr__(self):
        info = f'PBRMesh\n'
        info += f'\t3D: V={self.vertices.shape[0]}, F={self.faces.shape[0]}\n'
        if self.uvs_2d is not None:
            info += f'\t2D: V={self.uvs_2d.shape[0]}, F={self.faces_2d.shape[0]}\n'
        else:
            info += f'\t2D: V={None}, F={None}\n'
        includes = []
        excludes = []
        for k in ['albedo', 'metallic', 'roughness', 'bump']:
            if getattr(self, k, None) is not None:
                includes.append(k)
            else:
                excludes.append(k)
        info += f'\tinclues: ' + ', '.join(includes) + '\n'
        info += f'\texcludes: ' + ', '.join(excludes) + '\n'
        return info


class PBRScene:
    def __init__(self, pbr_mesh_list:Union[List[PBRMesh], List["PBRScene"]]):
        vertices_list, faces_list = [], []
        areas_list, normals_list = [], []
        vertices_idx_list, faces_idx_list = [], []
        V_interval, F_interval = [], []
        V_idx, F_idx = 0, 0
        for idx, pbr_mesh in enumerate(pbr_mesh_list):
            vertices, faces = pbr_mesh.vertices, pbr_mesh.faces
            areas, normals = pbr_mesh.areas, pbr_mesh.normals
            V_cnt, F_cnt = vertices.shape[0], faces.shape[0]
            vertices_list += [vertices]
            faces_list += [faces + V_idx]
            areas_list += [areas]
            normals_list += [normals]
            V_interval.append([V_idx, V_idx + V_cnt, V_cnt])
            F_interval.append([F_idx, F_idx + F_cnt, F_cnt])
            vertices_idx_list += [to_tensor_i([idx]).expand(V_cnt)]
            faces_idx_list += [to_tensor_i([idx]).expand(F_cnt)]
            V_idx += V_cnt
            F_idx += F_cnt
        self.pbr_mesh_list = pbr_mesh_list
        self.vertices = torch.cat(vertices_list, dim=0)
        self.faces = torch.cat(faces_list, dim=0)
        self.areas = torch.cat(areas_list, dim=0)
        self.normals = torch.cat(normals_list, dim=0)
        self._vertex_normals = None
        self._optix = None

        self.V_interval = to_tensor_i(V_interval)
        self.F_interval = to_tensor_i(F_interval)
        self.vertices_idx = torch.cat(vertices_idx_list, dim=0)
        self.faces_idx = torch.cat(faces_idx_list, dim=0)

    @property
    def vertex_normals(self):
        if self._vertex_normals is None:
            self._vertex_normals = torch.cat([pbr_mesh.vertex_normals for pbr_mesh in self.pbr_mesh_list], dim=0)
        return self._vertex_normals

    @property
    def optix(self):
        if self._optix is None:
            self._optix = RayTracing(self.vertices, self.faces)
        return self._optix

    def scale_to_bbox(self, largest=True, scale=1.0):
        '''
        largest: align largest or shortest length of bbox to 1
        scale: do extra scaling wrt origin after aligned to bbox
        '''
        aaa = self.vertices.min(dim=0)
        bbb = self.vertices.max(dim=0)
        if largest:
            scale = (bbb - aaa).max() / (2.0 * scale)
        else:
            scale = (bbb - aaa).min() / (2.0 * scale)
        transform = torch.eye(4, dtype=self.vertices.dtype, device=self.vertices.device)
        transform[[0, 1, 2], [0, 1, 2]] = 1.0 / scale
        transform[:3, 3] = - (aaa + bbb) / (2.0 * scale)
        vertices_homo = torch.cat([self.vertices, torch.ones_like(self.vertices[:, [0]])], dim=-1)
        vertices_homo = torch.matmul(vertices_homo, transform.T)
        self.vertices = vertices_homo[:, :3].contiguous()
        normals = torch.matmul(self.normals, transform[:3, :3].T)
        self.normals = torch.nn.functional.normalize(normals, dim=-1)
        if self._vertex_normals is not None:
            _vertex_normals = torch.matmul(self._vertex_normals, transform[:3, :3].T)
            self._vertex_normals = torch.nn.functional.normalize(_vertex_normals, dim=-1)
        if self._optix is not None:
            self._optix.update_raw(vertices=self.vertices, faces=self.faces)
        return self

    def __call__(self, face_idx:torch.Tensor, face_uvw:torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        face_idx: [N,], int64
        face_uvw: [N, 3], float32
        out: Dict[str, [N, C]], float32
        '''
        N = face_idx.shape[0]
        indices = self.faces_idx[face_idx]
        out = dict()
        for k in ['albedo', 'metallic', 'roughness', 'bump']:
            out[k] = getattr(PBRDefault, k).clone().unsqueeze(0).expand(N, -1).clone()
        for idx, pbr_mesh in enumerate(self.pbr_mesh_list):
            mask = (indices == idx)
            if not mask.any():
                continue
            _out = pbr_mesh(face_idx[mask] - self.F_interval[idx, 0], face_uvw[mask, :])
            mask = mask.unsqueeze(-1)
            for k in ['albedo', 'metallic', 'roughness', 'bump']:
                out[k].masked_scatter_(mask, _out[k])
        return out

    def __repr__(self):
        info = f'PBRScene\n'
        info += f'\tV={self.V_interval[:, 2].tolist()}\n'
        info += f'\tF={self.F_interval[:, 2].tolist()}\n'
        for i, pbr_mesh in enumerate(self.pbr_mesh_list):
            info += f'{i}-{pbr_mesh.__repr__()}'
        return info


def trimesh_to_pbr_mesh(mesh:Union[trimesh.Trimesh, trimesh.Scene], safe_convert=True) -> Union[PBRMesh, PBRScene]:
    '''
    safe_convert: whether deepcopy mesh in processing
    '''
    if isinstance(mesh, trimesh.Trimesh):
        if safe_convert:
            m = mesh.copy(include_cache=True)
            if isinstance(mesh.visual, TextureVisuals):
                m.visual.vertex_attributes.update(mesh.visual.vertex_attributes.data)
        else:
            m = mesh
        m.merge_vertices(merge_tex=False, merge_norm=True)
        uvs_2d = None
        faces_2d = None
        albedo = None
        metallic = None
        roughness = None
        bump = None
        if isinstance(m.visual, ColorVisuals):
            v_rgb = m.visual.vertex_colors
            if v_rgb is not None:
                albedo = to_tensor_f(v_rgb)[..., :3]
            else:
                albedo = None
        elif isinstance(m.visual, TextureVisuals):
            if m.visual.uv is not None:
                uvs_2d = torch.as_tensor(m.visual.uv, dtype=torch.float32)
                # NOTE: padding_mode='reflection'
                # uvs_2d = torch.abs(uvs_2d.add(1.0).fmod(2.0).sub(1.0))
                uvs_2d = uvs_2d.mul(2.0).sub(1.0)
                faces_2d = torch.as_tensor(m.faces, dtype=torch.int64)
            map_Kd, map_Ks, map_normal = parse_texture_visuals(m.visual)
            if map_Kd is not None:
                albedo = to_tensor_f(np.array(map_Kd.convert('RGB'))).div(255.0).flip(0)
            else:
                v_rgb = m.visual.vertex_attributes.data.get('color', None)
                if v_rgb is not None:
                    albedo = to_tensor_f(v_rgb)[..., :3]
                else:
                    albedo = None
            if map_Ks is not None:
                ones_roughness_metallic = to_tensor_f(np.array(map_Ks.convert('RGB'))).div(255.0).flip(0)
                metallic = ones_roughness_metallic[..., [2]]
                roughness = ones_roughness_metallic[..., [1]]
            if map_normal is not None:
                bump = to_tensor_f(np.array(map_normal.convert('RGB'))).div(255.0).flip(0)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        vertices = to_tensor_f(m.vertices)
        faces = to_tensor_i(m.faces)
        out = PBRMesh(
            vertices=vertices,
            faces=faces,
            uvs_2d=uvs_2d,
            faces_2d=faces_2d,
            albedo=albedo,
            metallic=metallic,
            roughness=roughness,
            bump=bump,
        )
        return out
    elif isinstance(mesh, trimesh.Scene):
        out = []
        for i, m in enumerate(mesh.dump()):
            out += [trimesh_to_pbr_mesh(m, safe_convert=safe_convert)]
        out = PBRScene(out)
        return out
    else:
        raise NotImplementedError(f'mesh type {type(mesh)} is not supported')


def pbr_mesh_to_trimesh(mesh:Union[PBRMesh, PBRScene]) -> Union[trimesh.Trimesh, trimesh.Scene]:
    if isinstance(mesh, PBRMesh):  # TODO
        out = trimesh.Trimesh(
            vertices=...,
            faces=...,
            visual=TextureVisuals(
                uv=...,
                material=PBRMaterial(
                    baseColorTexture=...,
                    metallicRoughnessTexture=...,
                    normalTexture=...,
                ),
            ),
            process=False,
        )
    elif isinstance(mesh, PBRScene):
        out = trimesh.Scene()
        for i, m in enumerate(mesh.pbr_mesh_list):
            o = pbr_mesh_to_trimesh(m)
            out.add_geometry(o, node_name=str(i), geom_name=str(i))
        return out
    else:
        raise NotImplementedError(f'mesh type {type(mesh)} is not supported')


