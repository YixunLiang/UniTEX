'''
obj saver for whole mesh with pbr texture
support different face index of position, uv and normal
'''
import os
from typing import Optional
import numpy as np
from PIL import Image


class CacheMixin:
    _func_name_queue = []
    _kwargs_queue = []
    def _cache_func(self, func_name, *args, **kwargs):
        assert hasattr(self, func_name)
        assert len(args) == 0, 'use keyword parameters instead of position parameters'
        self._func_name_queue.append(func_name)
        self._kwargs_queue.append(kwargs)
        return self
    
    def _apply_func(self):
        for func_name, kwargs in zip(self._func_name_queue, self._kwargs_queue):
            getattr(self, func_name)(**kwargs)
        self._func_name_queue = []
        self._kwargs_queue = []
        return self


class ObjSaver(CacheMixin):
    def __init__(self) -> None:
        '''
        obj_dict = {
            'obj_str': '',
            'mtl_str': {
                '[mtllibname]': '',
                '[mtllibname]': '',
            },
            'mtl_map': {
                '[mtllibname]_[mtlname]_[map_type].[map_format]',
                '[mtllibname]_[mtlname]_[map_type].[map_format]',
            },
        }
        '''
        self.obj_dict = {
            'obj_str': '',
            'mtl_str': dict(),
            'mtl_map': dict(),
        }
    
    def write_mesh(self, obj_path):
        assert os.path.splitext(obj_path)[1] == '.obj', 'support obj only'
        obj_dir = os.path.dirname(obj_path)
        os.makedirs(obj_dir, exist_ok=True)

        obj_str = self.obj_dict['obj_str']
        with open(obj_path, 'w') as f:
            f.write(obj_str)
        for k, v in self.obj_dict['mtl_str'].items():
            mtl_path = os.path.join(obj_dir, k+'.mtl')
            with open(mtl_path, "w") as f:
                f.write(v)
        for k, v in self.obj_dict['mtl_map'].items():
            map_path = os.path.join(obj_dir, k)
            v.save(map_path)

    def add_mesh(
        self,
        mtllibname:str,
        mtlname:str,
        v_pos:np.ndarray,
        t_pos_idx:np.ndarray,
        v_nrm:Optional[np.ndarray]=None,
        t_nrm_idx:Optional[np.ndarray]=None,
        v_tex:Optional[np.ndarray]=None,
        t_tex_idx:Optional[np.ndarray]=None,
        Ka=(1.0,1.0,1.0),
        Kd=(0.8,0.8,0.8),
        Ks=(0.0,0.0,0.0),
        map_Kd:Optional[Image.Image]=None,
        map_Ks:Optional[Image.Image]=None,
        map_Bump:Optional[Image.Image]=None,
        map_Pm:Optional[Image.Image]=None,
        map_Pr:Optional[Image.Image]=None,
        map_format="png",
    ) -> str:
        obj_str = ""
        obj_str += f"mtllib {mtllibname}\n"
        obj_str += f"usemtl {mtlname}\n"
        mtl_str = ""
        mtl_str += f"newmtl {mtlname}\n"
        mtl_str += f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
        mtl_str += f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"
        mtl_str += f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"
        mtl_dict = dict()
        
        for v in v_pos:
            obj_str += f"v {v[0]} {v[1]} {v[2]}\n"
        if v_nrm is not None:
            for v in v_nrm:
                obj_str += f"vn {v[0]} {v[1]} {v[2]}\n"
        if v_tex is not None:
            include_materials = True
            # NOTE: flip v for obj uv
            v_tex[:, 1] = 1 - v_tex[:, 1]
            for v in v_tex:
                obj_str += f"vt {v[0]} {v[1]}\n"
        else:
            # NOTE: if uv is missing, ignore materials
            include_materials = False
        
        # NOTE: obj index starts from 1
        if t_tex_idx is not None and t_nrm_idx is not None:
            t_idx = np.stack([t_pos_idx, t_tex_idx, t_nrm_idx], axis=-1) + 1
            for t in t_idx:
                obj_str += f"f {t[0, 0]}/{t[0, 1]}/{t[0, 2]} {t[1, 0]}/{t[1, 1]}/{t[1, 2]} {t[2, 0]}/{t[2, 1]}/{t[2, 2]}\n"
        elif t_tex_idx is not None and t_nrm_idx is None:
            t_idx = np.stack([t_pos_idx, t_tex_idx], axis=-1) + 1
            for t in t_idx:
                obj_str += f"f {t[0, 0]}/{t[0, 1]}/ {t[1, 0]}/{t[1, 1]}/ {t[2, 0]}/{t[2, 1]}/\n"
        elif t_tex_idx is None and t_nrm_idx is not None:
            t_idx = np.stack([t_pos_idx, t_nrm_idx], axis=-1) + 1
            for t in t_idx:
                obj_str += f"f {t[0, 0]}//{t[0, 1]} {t[1, 0]}//{t[1, 1]} {t[2, 0]}//{t[2, 1]}\n"
        elif t_tex_idx is None and t_nrm_idx is None:
            t_idx = t_pos_idx + 1
            for t in t_idx:
                obj_str += f"f {t[0]}// {t[1]}// {t[2]}//\n"
        obj_str += '\n'
        self.obj_dict['obj_str'] += obj_str

        if include_materials:
            if map_Kd is not None:
                map_type = 'diffuse'
                imagename = f"{mtllibname}_{mtlname}_{map_type}.{map_format}"
                mtl_str += f"map_Kd {imagename}\n"
                mtl_dict[imagename] = map_Kd
            if map_Ks is not None:
                map_type = 'specular'
                imagename = f"{mtllibname}_{mtlname}_{map_type}.{map_format}"
                mtl_str += f"map_Ks {imagename}\n"
                mtl_dict[imagename] = map_Ks
            if map_Bump is not None:
                map_type = 'normal'
                imagename = f"{mtllibname}_{mtlname}_{map_type}.{map_format}"
                mtl_str += f"map_Bump {imagename}\n"
                mtl_dict[imagename] = map_Bump
            if map_Pm is not None:
                map_type = 'metallic'
                imagename = f"{mtllibname}_{mtlname}_{map_type}.{map_format}"
                mtl_str += f"map_Pm {imagename}\n"
                mtl_dict[imagename] = map_Pm
            if map_Pr is not None:
                map_type = 'roughness'
                imagename = f"{mtllibname}_{mtlname}_{map_type}.{map_format}"
                mtl_str += f"map_Pr {imagename}\n"
                mtl_dict[imagename] = map_Pm
            mtl_str += '\n'
            
            if mtllibname not in self.obj_dict['mtl_str'].keys():
                self.obj_dict['mtl_str'][mtllibname] = ''
            self.obj_dict['mtl_str'][mtllibname] += mtl_str
            self.obj_dict['mtl_map'].update(mtl_dict)
