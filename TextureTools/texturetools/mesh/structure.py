'''
Mesh Data Structure based on pytorch
'''

import math
import os
from time import perf_counter
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.texture import unmerge_faces
from trimesh.visual.material import PBRMaterial
import pymeshlab as ml
import open3d as o3d
import gpytoolbox as gtb

from ..geometry.triangle_topology.topology import get_boundary_tex, reverse_triangle_group_2d
from .utils import dot
from ..io.obj_saver import ObjSaver
from ..render.nvdiffrast.renderer_base import NVDiffRendererBase
from ..texture.stitching.mip import pull_push
from .trimesh_utils import parse_texture_visuals


class DeviceMixin:
    attr_list = []
    def to(self, device):
        device = torch.device(device)
        for key in self.attr_list:
            value = getattr(self, key, None)
            if value is not None:
                if isinstance(value, torch.Tensor) or (hasattr(value, 'device') and hasattr(value, 'to')):
                    if value.device != device:
                        setattr(self, key, value.to(device))
        return self


class ExporterMixin:
    def to_trimesh(self) -> trimesh.Trimesh:
        raise NotImplementedError

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        raise NotImplementedError
    
    def _to_pymeshlab(self) -> Dict:
        raise NotImplementedError

    def to_pymeshlab(self) -> ml.Mesh:
        return ml.Mesh(**self._to_pymeshlab())
    
    def _to_custom(self) -> Dict:
        raise NotImplementedError
    
    def to_custom(self) -> ObjSaver:
        return ObjSaver()._cache_func('add_mesh', **self._to_custom())

    def export(self, obj_path: str, backend='auto'):
        '''
        NOTE: backend
        * auto: assign obj to custom, assign glb to trimesh
        * trimesh: obj(missing pbr materials), glb
        * open3d: obj(missing pbr materials), glb(missing uv and texture)
        * pymeshlab: obj(missing pbr materials), glb(not supported)
        * custom: obj, glb(not supported)
        '''
        assert backend in ['auto', 'trimesh', 'open3d', 'pymeshlab', 'custom']
        ext = os.path.splitext(obj_path)[1]
        if backend == 'auto':
            if ext == '.obj':
                backend = 'custom'
            elif ext == '.glb':
                backend = 'trimesh'
            else:
                backend = 'trimesh'
            return self.export(obj_path=obj_path, backend=backend)
        obj_path = os.path.abspath(obj_path)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        if backend == 'trimesh':
            kwargs = {'include_normals': True} if ext in ['.obj', '.glb'] else {}
            self.to_trimesh().export(obj_path, **kwargs)
        elif backend == 'open3d':
            o3d.io.write_triangle_mesh(
                obj_path, 
                self.to_open3d(), 
                write_ascii=False, 
                compressed=False, 
                write_vertex_normals=True, 
                write_vertex_colors=True, 
                write_triangle_uvs=True, 
                print_progress=False,
            )
        elif backend == 'pymeshlab':
            ms = ml.MeshSet()
            ms.add_mesh(self.to_pymeshlab(), mesh_name='model', set_as_current=True)
            ms.save_current_mesh(
                obj_path,
                save_vertex_color = True,
                save_vertex_coord = True,
                save_vertex_normal = True,
                save_face_color = False, 
                save_wedge_texcoord = True,
                save_wedge_normal = False,
                save_polygonal = False,
            )
            map_Kd = getattr(self, 'map_Kd', None)
            if map_Kd is not None:
                image = Image.fromarray(map_Kd.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
                image.save(os.path.join(os.path.dirname(obj_path), 'material_0.png'))
                mtl_path = os.path.join(os.path.dirname(obj_path), os.path.splitext(os.path.basename(obj_path))[0]+'.mtl')
                if os.path.isfile(mtl_path):
                    with open(mtl_path, 'r+') as f:
                        lines = f.readlines()
                        has_mtl = False
                        for i in range(len(lines)):
                            if lines[i].split(' ')[0] == 'map_Kd':
                                lines[i] = f'map_Kd material_0.png'
                                has_mtl = True
                                break
                        if not has_mtl:
                            lines.append(f'map_Kd material_0.png')
                        f.seek(0, 0)
                        f.writelines(lines)
                else:
                    with open(mtl_path, 'w') as f:
                        f.write(f'newmtl material_0\n')
                        f.write(f'Ka 1.00000000 1.00000000 1.00000000\n')
                        f.write(f'Kd 1.00000000 1.00000000 1.00000000\n')
                        f.write(f'Ks 0.50000000 0.00000000 0.50000000\n')
                        f.write(f'map_Kd material_0.png\n')
                    with open(obj_path, "r+") as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(f'mtllib {os.path.splitext(os.path.basename(obj_path))[0]}.mtl\n')
                        f.write(f'usemtl material_0\n')
                        f.write(content)
        elif backend == 'custom':
            mesh = self.to_custom()
            for i, m in enumerate(mesh._kwargs_queue):
                m.update({
                    'mtllibname': os.path.splitext(os.path.basename(obj_path))[0],
                    'mtlname': f'material_{i}',
                })
            mesh._apply_func()
            mesh.write_mesh(obj_path)
        else:
            raise NotImplementedError(f'export backend {backend} is not supported yet')


class MeasureMixin:  # disable lazy api
    def bbox(self):  # [2, 3]
        return torch.stack([
            self.v_pos.min(dim=0).values,
            self.v_pos.max(dim=0).values,
        ], dim=0)
    
    def bbox_center(self):  # [3,]
        return self.bbox().mean(dim=0)
    
    def bbox_radius(self):  # [3,]
        bbox = self.bbox()
        return (bbox[1, :] - bbox[0, :]).square().sum().sqrt()
    
    def center(self):  # [3,]
        return self.v_pos.mean(dim=0)
    
    def radius(self):
        d2 = self.v_pos.square().sum(dim=1)
        return d2.max().sqrt()


class TransformMixin(DeviceMixin):
    def __init__(self) -> None:
        super().__init__()
        self._identity = torch.eye(n=4, dtype=torch.float32)
        self._transform = None
        self.attr_list.extend(['_identity', '_transform'])
    
    @property
    def identity(self) -> torch.Tensor:  # read only
        return self._identity
    
    def init_transform(self, transform:Optional[torch.Tensor]=None):
        self._transform = self.identity.clone() if transform is None else transform
        return self
    
    def apply_transform(self, clear_transform=True):
        if self._transform is not None:
            v_pos_homo = torch.cat([self.v_pos, torch.ones_like(self.v_pos[:, [0]])], dim=-1)
            v_pos_homo = torch.matmul(v_pos_homo, self._transform.T.to(v_pos_homo))
            self.v_pos = v_pos_homo[:, :3].contiguous()
            v_nrm = getattr(self, '_v_nrm', None)
            if v_nrm is not None:
                v_nrm = torch.matmul(v_nrm, self._transform[:3, :3].T.to(v_nrm))
                v_nrm = torch.nn.functional.normalize(v_nrm, dim=-1)
                self._v_nrm = v_nrm.contiguous()
            if clear_transform:
                self._transform = None
        return self
    
    def compose_transform(self, transform:torch.Tensor, after=True):
        if self._transform is None:
            self.init_transform()
        if after:
            self._transform = torch.matmul(transform.to(self._transform), self._transform)
        else:
            self._transform = torch.matmul(self._transform, transform.to(self._transform))
        return self


class CoordinateSystemMixin(MeasureMixin, TransformMixin):
    def flip_x(self):
        transform = self.identity.clone()
        transform[0, 0] = -1
        self.compose_transform(transform)
        return self

    def flip_y(self):
        transform = self.identity.clone()
        transform[1, 1] = -1
        self.compose_transform(transform)
        return self

    def flip_z(self):
        transform = self.identity.clone()
        transform[2, 2] = -1
        self.compose_transform(transform)
        return self

    def swap_xy(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 1] = 1
        transform[1, 0] = 1
        transform[2, 2] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self

    def swap_yz(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 0] = 1
        transform[1, 2] = 1
        transform[2, 1] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self

    def swap_zx(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 2] = 1
        transform[1, 1] = 1
        transform[2, 0] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self
    
    def translate(self, offset:torch.Tensor):
        transform = self.identity.clone()
        transform[:3, 3] = offset
        self.compose_transform(transform)
        return self

    def scale(self, scale:torch.Tensor):
        transform = self.identity.clone()
        transform[[0, 1, 2], [0, 1, 2]] = scale
        self.compose_transform(transform)
        return self

    def align_to(self, src:torch.Tensor, dst:Optional[torch.Tensor]=None):
        if dst is None:
            return self.translate(src.neg())
        else:
            return self.translate(dst.sub(src))
    
    def scale_to(self, src:torch.Tensor, dst:Optional[torch.Tensor]=None):
        if dst is None:
            return self.scale(src.reciprocal())
        else:
            return self.scale(dst.div(src))

    def align_to_bbox_center(self):
        return self.align_to(self.bbox_center())
    
    def align_center(self):
        return self.align_to(self.center())
    
    def scale_to_bbox(self, largest=True, scale=1.0):
        '''
        largest: align largest or shortest length of bbox to 1
        scale: do extra scaling wrt origin after aligned to bbox
        '''
        bbox = self.bbox()
        ccc = bbox.mean(dim=0, keepdim=False)
        sss = (bbox[1, :] - bbox[0, :]) / (2.0 * scale)
        sss =  sss.max() if largest else sss.min()
        transform = self.identity.clone()
        transform[[0, 1, 2], [0, 1, 2]] = 1 / sss
        transform[:3, 3] = - ccc / sss
        self.compose_transform(transform)
        return self


class Mesh(ExporterMixin, CoordinateSystemMixin):
    def __init__(
        self, 
        v_pos:torch.Tensor, 
        t_pos_idx:torch.Tensor, 
        v_tex:Optional[torch.Tensor]=None, 
        t_tex_idx:Optional[torch.Tensor]=None, 
        **kwargs,
    ):
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self._v_nrm = None
        self._v_tng = None
        self._v_tex = v_tex
        self._t_tex_idx = t_tex_idx
        self._e_pos_idx = None
        self._t_pos_e_idx = None
        self._e_t_pos_idx = None

        assert v_pos.device == t_pos_idx.device, \
            f'v_pos is on {v_pos.device} but t_pos_idx is on {t_pos_idx.device}'
        assert v_tex is None or v_pos.device == v_tex.device, \
            f'v_pos is on {v_pos.device} but v_tex is on {v_tex.device}'
        assert t_tex_idx is None or v_pos.device == t_tex_idx.device, \
            f'v_pos is on {v_pos.device} but t_tex_idx is on {t_tex_idx.device}'
        self.attr_list.extend([
            'v_pos',
            't_pos_idx',
            '_v_nrm',
            '_v_tng',
            '_v_tex',
            '_t_tex_idx',
            '_e_pos_idx',
            '_t_pos_e_idx',
            '_e_t_pos_idx',
        ])
        CoordinateSystemMixin.__init__(self)
    
    @property
    def device(self):
        return self.v_pos.device
    
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        mesh = mesh.copy(include_cache=True)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        v_pos = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        t_pos_idx = torch.as_tensor(mesh.faces, dtype=torch.int64)
        m = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
        if mesh.vertex_normals is not None:
            v_nrm = torch.as_tensor(np.asarray(mesh.vertex_normals, dtype=np.float32), dtype=torch.float32)
        else:
            v_nrm = None
        m._v_nrm = v_nrm
        return m
    
    @classmethod
    def from_open3d(cls, mesh: Union[o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh]):
        if isinstance(mesh, o3d.t.geometry.TriangleMesh):
            mesh = mesh.to_legacy()
        v_pos = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), dtype=torch.float32)
        t_pos_idx = torch.as_tensor(np.asarray(mesh.triangles, dtype=np.int64), dtype=torch.int64)
        m = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
        if mesh.vertex_normals is not None:
            v_nrm = np.asarray(mesh.vertex_normals, dtype=np.float32)
            # NOTE: std::vector<Eigen::Vector3d> with 0 elements.
            if v_nrm.shape[0] == 0:
                v_nrm = None
            else:
                v_nrm = torch.as_tensor(v_nrm, dtype=torch.float32)
        else:
            v_nrm = None
        m._v_nrm = v_nrm
        return m
    
    @classmethod
    def from_pymeshlab(cls, mesh: ml.Mesh):
        v_pos = torch.as_tensor(np.asarray(mesh.vertex_matrix(), dtype=np.float32), dtype=torch.float32)
        t_pos_idx = torch.as_tensor(np.asarray(mesh.face_matrix(), dtype=np.int64), dtype=torch.int64)
        m = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
        try:
            v_nrm = torch.as_tensor(np.asarray(mesh.vertex_normal_matrix(), dtype=np.float32), dtype=torch.float32)
        except ml.MissingCompactnessException:
            v_nrm = None
        m._v_nrm = v_nrm
        return m
    
    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(), 
            faces=self.t_pos_idx.cpu().numpy(), 
            vertex_normals=self.v_nrm.detach().cpu().numpy(),
            process=False,
        )
        return mesh
    
    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.v_pos.detach().cpu().numpy()), 
            triangles=o3d.utility.Vector3iVector(self.t_pos_idx.cpu().numpy()),
        )
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.v_nrm.detach().cpu().numpy())
        return mesh
    
    def _to_pymeshlab(self) -> Dict:
        return dict(
            vertex_matrix=self.v_pos.detach().cpu().numpy().astype(np.float64),
            face_matrix=self.t_pos_idx.cpu().numpy().astype(np.int32),
            v_normals_matrix=self.v_nrm.detach().cpu().numpy().astype(np.float64),
        )

    def _to_custom(self) -> Dict:
        mesh = dict(
            mtllibname = None,
            mtlname = None,
            v_pos = self.v_pos.detach().cpu().numpy(),
            t_pos_idx = self.t_pos_idx.cpu().numpy(),
            v_tex = self.v_tex.detach().cpu().numpy(),
            t_tex_idx = self.t_tex_idx.cpu().numpy(),
            v_nrm = self.v_nrm.detach().cpu().numpy(),
            t_nrm_idx = self.t_pos_idx.cpu().numpy(),
            Ka = (1.0, 1.0, 1.0), 
            Kd = (1.0, 1.0, 1.0), 
            Ks = (0.5, 0.5, 0.5),  # specular, default is 0.5
            map_Kd = None, 
            map_Ks = None,  # specular, default is 0.5
            map_Bump = None, 
            map_Pm = None,  # B channel of map_Ks
            map_Pr = None,  # G channel of map_Ks
            map_format = "png",
        )
        return mesh

    def remove_outlier(self, outlier_n_faces_threshold):
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )
        components = mesh.split(only_watertight=False)
        if isinstance(outlier_n_faces_threshold, float):
            n_faces_threshold = int(max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold)
        else:
            n_faces_threshold = outlier_n_faces_threshold
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]
        mesh = trimesh.util.concatenate(components)
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)
        clean_mesh = Mesh(v_pos, t_pos_idx)
        return clean_mesh

    def merge_faces(self, v_pos_attr=None, v_tex_attr=None):
        if self._v_tex is not None and self._t_tex_idx is not None:
            t_joint_idx, v_pos_idx, v_tex_idx = unmerge_faces(self.t_pos_idx.cpu().numpy(), self.t_tex_idx.cpu().numpy(), maintain_faces=False)
            t_joint_idx = torch.as_tensor(t_joint_idx, dtype=torch.int64, device=self.device)
            v_pos_idx = torch.as_tensor(v_pos_idx, dtype=torch.int64, device=self.device)
            v_tex_idx = torch.as_tensor(v_tex_idx, dtype=torch.int64, device=self.device)
            m = Mesh(v_pos=self.v_pos[v_pos_idx], t_pos_idx=t_joint_idx)
            m._v_nrm = self.v_nrm[v_pos_idx]
            m._v_tex = self._v_tex[v_tex_idx]
            m._t_tex_idx = t_joint_idx
            if v_pos_attr is None and v_tex_attr is None:
                return m
            else:
                if v_pos_attr is not None:
                    v_pos_attr = v_pos_attr[v_pos_idx]
                if v_tex_attr is not None:
                    v_tex_attr = v_tex_attr[v_tex_idx]
                return m, v_pos_attr, v_tex_attr
        else:
            if v_pos_attr is None and v_tex_attr is None:
                return self
            else:
                return self, v_pos_attr, v_tex_attr

    @property
    def v_nrm(self) -> torch.Tensor:
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self) -> torch.Tensor:
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self) -> torch.Tensor:
        if self._v_tex is None:
            self.unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self) -> torch.Tensor:
        if self._t_tex_idx is None:
            self.unwrap_uv()
        return self._t_tex_idx

    @property
    def e_pos_idx(self) -> torch.Tensor:
        if self._e_pos_idx is None:
            self._compute_edges()
        return self._e_pos_idx
    
    @property
    def t_pos_e_idx(self) -> torch.Tensor:
        if self._t_pos_e_idx is None:
            self._compute_edges()
        return self._t_pos_e_idx

    @property
    def e_t_pos_idx(self) -> torch.Tensor:
        if self._e_t_pos_idx is None:
            self._compute_edges(compute_e_t_pos_idx=True)
        return self._e_t_pos_idx

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0].long()
        i1 = self.t_pos_idx[:, 1].long()
        i2 = self.t_pos_idx[:, 2].long()

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv_v1(self, chart_config:Dict=dict(), pack_config:Dict=dict()):
        import xatlas

        atlas = xatlas.Atlas()
        v_pos = self.laplacian_func(self.v_pos, depth=3)
        atlas.add_mesh(
            v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        _chart_config = {
            'max_chart_area': 0.0,
            'max_boundary_length': 0.0,
            'normal_deviation_weight': 2.0,
            'roundness_weight': 0.01,
            'straightness_weight': 6.0,
            'normal_seam_weight': 4.0,
            'texture_seam_weight': 0.5,
            'max_cost': 16.0,  # avoid small charts
            'max_iterations': 1,
            'use_input_mesh_uvs': False,
            'fix_winding': False,
        }
        _pack_config = {
            'max_chart_size': 0,
            'padding': 4,  # avoid adjoint
            'texels_per_unit': 0.0,
            'resolution': 2048,
            'bilinear': True,
            'blockAlign': False,
            'bruteForce': False,
            'create_image': False,
            'rotate_charts_to_axis': True,
            'rotate_charts': True,
        }
        _chart_config.update(chart_config)
        _pack_config.update(pack_config)
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in _chart_config.items():
            setattr(co, k, v)
        for k, v in _pack_config.items():
            setattr(po, k, v)
        
        print(f"UV unwrapping, V={self.v_pos.shape[0]}, F={self.t_pos_idx.shape[0]}, may take a while ...")
        t = perf_counter()
        atlas.generate(co, po, verbose=False)
        print(f"UV unwrapping wastes {perf_counter() - t} sec")

        _, indices, uvs = atlas.get_mesh(0)
        uvs = torch.as_tensor(uvs.astype(np.float32), dtype=self.v_pos.dtype, device=self.v_pos.device)
        indices = torch.as_tensor(indices.astype(np.int64), dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device)
        return uvs, indices
    
    def _unwrap_uv_v2(self, config:Dict=dict()):
        device = o3d.core.Device('CPU:0')
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int64
        mesh = o3d.t.geometry.TriangleMesh(device=device)
        v_pos = self.laplacian_func(self.v_pos, depth=3)
        mesh.vertex.positions = o3d.core.Tensor(v_pos.detach().cpu().numpy(), dtype=dtype_f, device=device)
        mesh.triangle.indices = o3d.core.Tensor(self.t_pos_idx.cpu().numpy(), dtype=dtype_i, device=device)
        _config = {
            'size': 2048,
            'gutter': 4.0,
            'max_stretch': 0.1667,
            'parallel_partitions': 4,
            'nthreads': 0,
        }
        _config.update(config)
        
        print(f"UV unwrapping, V={self.v_pos.shape[0]}, F={self.t_pos_idx.shape[0]}, may take a while ...")
        t = perf_counter()
        mesh.compute_uvatlas(**_config)
        print(f"UV unwrapping wastes {perf_counter() - t} sec")

        triangle_uvs = mesh.triangle.texture_uvs.numpy().astype(np.float32).reshape(-1, 2)  # [F*3, 2]
        t_tex = torch.as_tensor(triangle_uvs, dtype=self.v_pos.dtype, device=self.v_pos.device)
        v_tex, t_tex_idx = torch.unique(t_tex, dim=0, sorted=False, return_inverse=True, return_counts=False)
        t_tex_idx = t_tex_idx.reshape(-1, 3)
        return v_tex, t_tex_idx
        
    def unwrap_uv(self, **kwargs):
        self._v_tex, self._t_tex_idx = self._unwrap_uv_v2(**kwargs)

    def remesh(self, n_steps=1, scale=1.0):        
        e_pos = self.v_pos[self.e_pos_idx]  # [E, 2, 3]
        e_dis = torch.norm(e_pos[:, 1, :] - e_pos[:, 0, :], dim=-1)
        e_dis_mean = e_dis.mean()
        v_pos, t_pos_idx = gtb.remesh_botsch(
            V=self.v_pos.detach().cpu().numpy().astype(np.float64),
            F=self.t_pos_idx.detach().cpu().numpy().astype(np.int32),
            i=n_steps,
            h=e_dis_mean.item() * scale,
            project=True,
        )
        v_pos = torch.as_tensor(v_pos, dtype=torch.float32, device=self.device)
        t_pos_idx = torch.as_tensor(t_pos_idx, dtype=torch.int64, device=self.device)
        return Mesh(v_pos, t_pos_idx)

    def _compute_edges(self, compute_e_t_pos_idx=False):
        e_pos_idx_full = torch.cat([self.t_pos_idx[:, [0, 1]], self.t_pos_idx[:, [1, 2]], self.t_pos_idx[:, [2, 0]]], dim=0)  # [3*F, 2]
        e_pos_idx_sorted, e_pos_idx_sorted_idx = torch.sort(e_pos_idx_full, dim=-1)  # [3*F, 2], [3*F, 2](value: [0, 1] or [1, 0])
        e_pos_idx, _t_pos_e_idx, e_pos_count = torch.unique(e_pos_idx_sorted, dim=0, sorted=False, return_inverse=True, return_counts=True)  # [E, 2], [F*3,], [E,]
        t_pos_e_idx = _t_pos_e_idx.reshape(3, -1).permute(1, 0)  # [F, 3]
        if compute_e_t_pos_idx:
            t_idx = torch.arange(self.t_pos_idx.shape[0], dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device).repeat((3,))  # [3*F,]
            t_pos_e_type_1 = (e_pos_idx_sorted_idx[..., 0] == 0)  # [3*F,]
            t_pos_e_type_2 = (e_pos_idx_sorted_idx[..., 0] == 1)
            e_t_pos_idx = torch.zeros((e_pos_idx.shape[0], 2), dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device)  # [E, 2]
            e_t_pos_idx[:, 0].scatter_(dim=0, index=_t_pos_e_idx[t_pos_e_type_1], src=t_idx[t_pos_e_type_1])
            e_t_pos_idx[:, 1].scatter_(dim=0, index=_t_pos_e_idx[t_pos_e_type_2], src=t_idx[t_pos_e_type_2])
        else:
            e_t_pos_idx = None
        
        self._e_pos_idx = e_pos_idx
        self._t_pos_e_idx = t_pos_e_idx
        self._e_t_pos_idx = e_t_pos_idx


    def normal_consistency(self):
        edge_nrm = self.v_nrm[self.e_pos_idx]  # [E, 2, 3]
        nc = (1.0 - torch.cosine_similarity(edge_nrm[:, 0, :], edge_nrm[:, 1, :], dim=-1)).mean()
        return nc

    def _laplacian_v1(self, reciprocal=False):
        verts, faces = self.v_pos, self.t_pos_idx
        V = verts.shape[0]
        F = faces.shape[0]

        # neighbor
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj_idx = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
        adj_val = -torch.ones(adj_idx.shape[1]).to(verts)

        # diagonal
        diag_idx = torch.stack((adj_idx[0], adj_idx[0]), dim=0)
        diag_val = torch.ones(adj_idx.shape[1]).to(verts)

        # sparse matrix
        idx = torch.cat([adj_idx, diag_idx], dim=1)
        val = torch.cat([adj_val, diag_val], dim=0)
        L = torch.sparse_coo_tensor(idx, val, (V, V))

        # coalesce operation sums the duplicate indices
        L = L.coalesce()
        return L

    def _laplacian_v2(self, reciprocal=False):
        V = self.v_pos.shape[0]
        e0, e1 = self.e_pos_idx.unbind(1)

        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        A = torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32, device=self.device)
        deg = torch.sparse.sum(A, dim=1).to_dense()
        if reciprocal:
            deg = torch.nan_to_num(torch.reciprocal(deg), nan=0.0, posinf=0.0, neginf=0.0)
        val = torch.cat([deg[e0], deg[e1]])
        L = torch.sparse_coo_tensor(idx, val, (V, V), dtype=torch.float32, device=self.device)

        # idx = torch.arange(V, device=self.device)
        # idx = torch.stack([idx, idx], dim=0)
        # ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        # L -= torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32, device=self.device)
        return L
    
    @torch.no_grad()
    def laplacian(self, reciprocal=False):
        '''
        L: [V, V], edge laplacian
            L[i, j] = sum_{k\in V} 1_{k \in N_V(i, 1)}, (i, j) is a edge
            L[i, j] = 0, i == j
            L[i, j] = 0, otherwise
        if reciprocal is True, return 1 / L
        '''
        return self._laplacian_v2(reciprocal=reciprocal)

    def laplacian_func(self, v_attr:torch.Tensor, depth=1):
        if depth == 1:
            return v_attr
        L = self.laplacian(reciprocal=True)
        v_attr = torch.matmul(L, v_attr)
        return self.laplacian_func(v_attr, depth=depth-1)

    def laplacian_loss(self, v_attr:torch.Tensor, depth=1):
        return self.laplacian_func(v_attr, depth=depth).norm(dim=-1).mean()

    def compute_uv_mask(self, texture_size:Union[int, Tuple[int]]):
        nvdiffrast_renderer = NVDiffRendererBase(device='cuda')
        mesh = Mesh(
            v_pos=self.v_pos.to(dtype=torch.float32, device='cuda'),
            t_pos_idx=self.t_pos_idx.to(dtype=torch.int64, device='cuda'),
            v_tex=self.v_tex.to(dtype=torch.float32, device='cuda'),
            t_tex_idx=self.t_tex_idx.to(dtype=torch.int64, device='cuda'),
        )
        mask = nvdiffrast_renderer.simple_inverse_rendering(
            mesh, None, None, None,
            None, None, texture_size,
            enable_antialis=False,
        )['mask'][0].to(device=self.device)
        return mask

    def get_visible_faces(self, c2ws:torch.Tensor, perspective=True, backend="open3d"):
        batch_size = c2ws.shape[0]
        c2ws = c2ws.to(device=self.device)

        ray_hit = self.v_pos[self.t_pos_idx].mean(dim=1).unsqueeze(0)
        if perspective:
            ray_directions = ray_hit - c2ws[:, :3, 3].unsqueeze(-2)
            ray_origins = c2ws[:, :3, 3].unsqueeze(-2).expand_as(ray_directions)
        else:
            ray_origins = ray_hit + (2.0 * math.sqrt(3.0)) * c2ws[:, :3, 2].unsqueeze(-2)
            ray_directions = - c2ws[:, :3, 2].unsqueeze(-2).expand_as(ray_origins)

        print(f"Ray casting, V={self.v_pos.shape[0]}, F={self.t_pos_idx.shape[0]}, may take a while ...")
        t = perf_counter()
        if backend == 'trimesh':
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.to_trimesh())
            triangle_index = intersector.intersects_first(
                ray_origins=ray_origins.reshape(-1, 3).detach().cpu().numpy(), 
                ray_directions=ray_directions.reshape(-1, 3).detach().cpu().numpy(),
            )
            triangle_index_invalid_idx = -1
        elif backend == 'open3d':
            intersector = o3d.t.geometry.RaycastingScene()  # This class supports only the CPU device.
            intersector.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.to_open3d(), vertex_dtype=o3d.core.float32, triangle_dtype=o3d.core.int64, device=o3d.core.Device('CPU:0')))
            rays_o3d = o3d.core.Tensor(np.concatenate([
                ray_origins.reshape(-1, 3).detach().cpu().numpy(),
                ray_directions.reshape(-1, 3).detach().cpu().numpy(),
            ], axis=-1), dtype=o3d.core.float32, device=o3d.core.Device('CPU:0'))
            # NOTE: t_hit, geometry_ids, primitive_ids, primitive_uvs, primitive_normals
            triangle_index = intersector.cast_rays(rays_o3d)['primitive_ids']
            triangle_index = np.asarray(triangle_index.numpy(), dtype=np.int64)
            triangle_index_invalid_idx = intersector.INVALID_ID
        else:
            raise NotImplementedError
        print(f"Ray casting wastes {perf_counter() - t} sec")

        t_idx_visible = torch.as_tensor(triangle_index, dtype=torch.int64, device=self.device)
        if triangle_index_invalid_idx != -1:
            t_idx_visible = torch.where(t_idx_visible == triangle_index_invalid_idx, -1, t_idx_visible)
        t_idx_visible_dummy = t_idx_visible.add(1).reshape(batch_size, -1)
        t_mask_visible_dummy = torch.zeros((batch_size, self.t_pos_idx.shape[0] + 1), dtype=torch.bool, device=self.device)
        t_mask_visible_dummy = torch.scatter(t_mask_visible_dummy, -1, t_idx_visible_dummy, 1)
        t_mask_visible = t_mask_visible_dummy[:, 1:]
        return t_mask_visible
    
    def get_visible_vertices(self, c2ws:torch.Tensor, perspective=True):
        batch_size = c2ws.shape[0]
        device = c2ws.device

        t_pos_idx = self.t_pos_idx
        t_mask_visible = self.get_visible_faces(c2ws, perspective=perspective)
        v_pos_mask_visible = torch.zeros((batch_size, self.v_pos.shape[0]), dtype=torch.bool, device=device)
        for b in range(batch_size):
            t_pos_idx_visible = torch.masked_select(t_pos_idx, t_mask_visible[b].unsqueeze(-1))
            t_pos_idx_visible_unique = torch.unique(t_pos_idx_visible, return_inverse=False, return_counts=False)
            v_pos_mask_visible[b] = torch.index_fill(v_pos_mask_visible[b], -1, t_pos_idx_visible_unique, 1)
        return v_pos_mask_visible
    
    def get_seams_submesh(self, reverse=True):
        v_pos = self.v_pos
        v_tex = self.v_tex
        f_v_idx_pos = self.t_pos_idx
        f_v_idx_tex = self.t_tex_idx
        v_idx_sel_pos, v_idx_sel_tex, f_v_idx_sel_pos, f_v_idx_sel_tex = get_boundary_tex(f_v_idx_pos, f_v_idx_tex, paired=reverse)
        if reverse:
            f_v_sel_tex = v_tex[f_v_idx_sel_tex]
            f_v_sel_tex_reverse = reverse_triangle_group_2d(f_v_sel_tex)
            v_tex[f_v_idx_sel_tex] = f_v_sel_tex_reverse
        return Mesh(
            v_pos, 
            f_v_idx_sel_pos.reshape(-1, 3), 
            v_tex, 
            f_v_idx_sel_tex.reshape(-1, 3), 
        )

def parse_v_rgb(v_rgb: np.ndarray) -> torch.Tensor:
    v_rgb = torch.as_tensor(v_rgb, dtype=torch.float32).div(255.0)
    if v_rgb.shape[-1] == 1:
        v_rgb = v_rgb.tile(1, 3)
    elif v_rgb.shape[-1] == 2:
        v_rgb = torch.cat([
            v_rgb[..., [0]], 
            torch.zeros_like(v_rgb[..., [0]]), 
            v_rgb[..., [1]],
        ], dim=-1)
    elif v_rgb.shape[-1] == 3:
        v_rgb = v_rgb
    elif v_rgb.shape[-1] > 3:
        v_rgb = v_rgb[..., :3]
    else:
        raise NotImplementedError
    return v_rgb


class Texture(DeviceMixin, ExporterMixin):
    texture_key_suffix_dict = {
        'map_kd': ('map_Kd', 'diffuse'),
        'map_ks': ('map_Ks', 'metallic_roughness'),
        'map_pm': ('map_Pm', 'metallic'),
        'map_pr': ('map_Pr', 'roughness'),
        'map_bump': ('map_Bump', 'normal'),
    }
    def __init__(
        self, 
        mesh:Mesh, 
        v_rgb:Optional[torch.Tensor]=None, 
        map_Kd:Optional[torch.Tensor]=None,
        map_Ks:Optional[torch.Tensor]=None,
        map_normal:Optional[torch.Tensor]=None,
        **kwargs,
    ) -> None:
        '''
        1. RGB channel of map_Ks:
            * R: unused or specular, default R channel is 1.0, but default specular is 0.5
            * G: roughness, default roughness is 1.0
            * B: matellic, default matellic is 0.0
        2. if map_Ks is None, sometimes we set map_Ke to map_Kd and set map_Kd to 0.0 or 1.0
        '''
        self.mesh: Mesh = mesh
        self.v_rgb = v_rgb
        self.map_Kd = map_Kd
        self.map_Ks = map_Ks
        self.map_normal = map_normal

        assert v_rgb is None or mesh.device == v_rgb.device, \
            f'mesh is on {mesh.device} but v_rgb is on {v_rgb.device}'
        assert map_Kd is None or mesh.device == map_Kd.device, \
            f'mesh is on {mesh.device} but map_Kd is on {map_Kd.device}'
        assert map_Ks is None or mesh.device == map_Ks.device, \
            f'mesh is on {mesh.device} but map_Ks is on {map_Ks.device}'
        assert map_normal is None or mesh.device == map_normal.device, \
            f'mesh is on {mesh.device} but map_normal is on {map_normal.device}'
        self.attr_list.extend(['mesh', 'v_rgb', 'map_Kd', 'map_Ks', 'map_normal'])
    
    @property
    def device(self):
        return self.mesh.device

    def to_trimesh(self) -> trimesh.Trimesh:
        if self.v_rgb is not None:
            mesh:trimesh.Trimesh = self.mesh.to_trimesh()
            mesh.visual = ColorVisuals(
                vertex_colors=self.v_rgb.clamp(0.0, 1.0).detach().cpu().numpy(),
            )
        elif (self.mesh._v_tex is not None and self.mesh._t_tex_idx is not None) or self.map_Kd is not None:
            m = self.mesh.merge_faces()
            mesh:trimesh.Trimesh = m.to_trimesh()
            if m._v_tex is not None and m._t_tex_idx is not None:
                uv = m.v_tex.detach().cpu().numpy()
            else:
                uv = None
            if self.map_Kd is not None:
                map_Kd = Image.fromarray(self.map_Kd.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
            else:
                map_Kd = None
            if self.map_Ks is not None:
                map_Ks = Image.fromarray(self.map_Ks.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGB')
            else:
                map_Ks = None
            if self.map_normal is not None:
                map_normal = Image.fromarray(self.map_normal.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGB')
            else:
                map_normal = None
            mesh.visual = TextureVisuals(
                uv=uv,
                material=PBRMaterial(
                    baseColorTexture=map_Kd,
                    baseColorFactor=None,
                    metallicRoughnessTexture=map_Ks,
                    # NOTE: default value of metallic and roughness in trimesh is 1.0
                    metallicFactor=0.0 if map_Ks is None else None,
                    roughnessFactor=1.0 if map_Ks is None else None,
                    emissiveTexture=None,
                    emissiveFactor=None,
                    normalTexture=map_normal,
                ),
            )
        return mesh
    
    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        mesh:o3d.geometry.TriangleMesh = self.mesh.to_open3d()
        if self.v_rgb is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.v_rgb[..., :3].clamp(0.0, 1.0).detach().cpu().numpy())
        if self.mesh._v_tex is not None and self.mesh._t_tex_idx is not None:
            f_v_uv = self.mesh.v_tex[self.mesh.t_tex_idx]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(f_v_uv.reshape(-1, 2).detach().cpu().numpy())
        if self.map_Kd is not None:
            image = Image.fromarray(self.map_Kd.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
            mesh.textures = [o3d.geometry.Image(np.array(image, dtype=np.uint8))]
        return mesh
    
    def _to_pymeshlab(self) -> Dict:
        if self.v_rgb is not None:
            m, v_rgb, _ = self.mesh.merge_faces(self.v_rgb)
        else:
            m = self.mesh.merge_faces()
            v_rgb = None
        mesh: Dict = m._to_pymeshlab()
        if v_rgb is not None:
            mesh['v_color_matrix'] = v_rgb[..., :3].clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float64)
        if m._v_tex is not None and m._t_tex_idx is not None:
            mesh['v_tex_coords_matrix'] = m.v_tex.detach().cpu().numpy().astype(np.float64)
        return mesh
    
    def _to_custom(self) -> Dict:
        mesh: Dict = self.mesh._to_custom()
        map_Kd = Image.fromarray(self.map_Kd.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA') if self.map_Kd is not None else None
        map_Ks = Image.fromarray(self.map_Ks.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGB') if self.map_Ks is not None else None
        map_normal = Image.fromarray(self.map_normal.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGB') if self.map_normal is not None else None
        mesh.update(dict(
            map_Kd = map_Kd, 
            map_Ks = None,  # specular, default is 0.5
            map_Bump = map_normal, 
            map_Pm = map_Ks.getchannel('B') if map_Ks is not None else None,  # B channel of map_Ks
            map_Pr = map_Ks.getchannel('G') if map_Ks is not None else None,  # G channel of map_Ks
        ))
        return mesh

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        m = mesh.copy(include_cache=True)
        if isinstance(mesh.visual, TextureVisuals):
            m.visual.vertex_attributes.update(mesh.visual.vertex_attributes.data)
        m.merge_vertices(merge_tex=False, merge_norm=True)
        if isinstance(m.visual, ColorVisuals):
            if m.visual.vertex_colors is not None:
                v_rgb = parse_v_rgb(m.visual.vertex_colors)
            else:
                v_rgb = None
            map_Kd = None
            map_Ks = None
            map_normal = None
            v_tex = None
            t_tex_idx = None
        elif isinstance(m.visual, TextureVisuals):
            v_rgb = None
            map_Kd, map_Ks, map_normal = parse_texture_visuals(m.visual)
            if map_Kd is not None:
                map_Kd = torch.as_tensor(np.array(map_Kd.convert('RGBA'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            else:
                v_rgb = dict(m.visual.vertex_attributes).get('color', None)
                if v_rgb is not None:
                    v_rgb = parse_v_rgb(v_rgb)
            if map_Ks is not None:
                map_Ks = torch.as_tensor(np.array(map_Ks.convert('RGB'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            if map_normal is not None:
                map_normal = torch.as_tensor(np.array(map_normal.convert('RGB'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            if m.visual.uv is not None:
                v_tex = torch.as_tensor(m.visual.uv, dtype=torch.float32)
                t_tex_idx = torch.as_tensor(m.faces, dtype=torch.int64)
            else:
                v_tex = None
                t_tex_idx = None
        else:
            v_rgb = None
            map_Kd = None
            map_Ks = None
            map_normal = None
            v_tex = None
            t_tex_idx = None
        mesh = Mesh.from_trimesh(m)
        mesh._v_tex = v_tex
        mesh._t_tex_idx = t_tex_idx
        return Texture(mesh=mesh, v_rgb=v_rgb, map_Kd=map_Kd, map_Ks=map_Ks, map_normal=map_normal)

    @classmethod
    def from_open3d(cls, mesh: Union[o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh]):
        if isinstance(mesh, o3d.t.geometry.TriangleMesh):
            mesh = mesh.to_legacy()
        if mesh.vertex_colors is not None:
            v_rgb = np.asarray(mesh.vertex_colors, dtype=np.float32)
            # NOTE: std::vector<Eigen::Vector3d> with 0 elements.
            if v_rgb.shape[0] == 0:
                v_rgb = None
            else:
                v_rgb = torch.as_tensor(v_rgb, dtype=torch.float32)
        else:
            v_rgb = None
        if mesh.triangle_uvs is not None:
            t_tex = np.asarray(mesh.triangle_uvs, dtype=np.float32)
            # NOTE: std::vector<Eigen::Vector3d> with 0 elements.
            if t_tex.shape[0] == 0:
                v_tex = None
                t_tex_idx = None
            else:
                t_tex = torch.as_tensor(t_tex, dtype=torch.float32)
                v_tex, t_tex_idx = torch.unique(t_tex, dim=0, sorted=False, return_inverse=True, return_counts=False)
                t_tex_idx = t_tex_idx.reshape(-1, 3)
        else:
            v_tex = None
            t_tex_idx = None
        m = Mesh.from_open3d(mesh)
        m._v_tex = v_tex
        m._t_tex_idx = t_tex_idx
        return Texture(mesh=m, v_rgb=v_rgb, map_Kd=None, map_Ks=None, map_normal=None)  # TODO: material

    @classmethod
    def from_pymeshlab(cls, mesh: ml.Mesh):
        try:
            v_rgb = torch.as_tensor(mesh.vertex_color_matrix()[..., :3], dtype=torch.float32)
        except ml.MissingCompactnessException:
            v_rgb = None
        try:
            v_tex = torch.as_tensor(mesh.vertex_tex_coord_matrix(), dtype=torch.float32)
            t_tex_idx = torch.as_tensor(np.asarray(mesh.face_matrix(), dtype=np.int64), dtype=torch.int64)
        except ml.MissingCompactnessException or ml.MissingComponentException:
            try:
                t_tex = torch.as_tensor(mesh.wedge_tex_coord_matrix(), dtype=torch.float32)
                v_tex, t_tex_idx = torch.unique(t_tex, dim=0, sorted=False, return_inverse=True, return_counts=False)
                t_tex_idx = t_tex_idx.reshape(-1, 3)
            except ml.MissingCompactnessException or ml.MissingComponentException:
                v_tex = None
                t_tex_idx = None
        m = Mesh.from_pymeshlab(mesh)
        m._v_tex = v_tex
        m._t_tex_idx = t_tex_idx
        return Texture(mesh=m, v_rgb=v_rgb, map_Kd=None, map_Ks=None, map_normal=None)  # TODO: material

    def reset_map_Kd_mask(self):
        if self.map_Kd is not None:
            H, W, C = self.map_Kd.shape
            self.map_Kd[:, :, [-1]] = self.mesh.compute_uv_mask(texture_size=(H, W)).to(self.map_Kd)
        return self

    def outpaint_map_Kd(self):
        if self.map_Kd is not None:
            H, W, C = self.map_Kd.shape
            map_Kd = self.map_Kd.unsqueeze(0).permute(0, 3, 1, 2)
            map_Kd, map_Kd_mask = pull_push(map_Kd[:, :-1, :, :], map_Kd[:, [-1], :, :] > 0.0)
            map_Kd = torch.cat([map_Kd, map_Kd_mask.to(dtype=map_Kd.dtype)], dim=1)
            self.map_Kd = map_Kd.permute(0, 2, 3, 1).squeeze(0)
        return self

