'''
Geometry Renderer for Unified PBR
'''
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from ...texture.pbr.pbr import PBRModel
from .renderer_base import NVDiffRendererBase


class NVDiffRendererPBR(nn.Module):
    def __init__(self):
        super().__init__()
        self.renderer_base = NVDiffRendererBase()
        self.pbr_model = PBRModel()
        self.cache = []

    def render_base(
        self, mesh, map_Kd:Tensor, map_Ks:Optional[Tensor], 
        c2ws, intrinsics, render_size:Union[int, Tuple[int]],
        **kwargs,
    ):
        if map_Ks is None:
            # NOTE: [ones, roughness, metallic]
            map_Ks = torch.tensor([1.0, 1.0, 0.0], dtype=map_Kd.dtype, device=map_Kd.device).expand_as(map_Kd)
        render_result = self.renderer_base.simple_rendering(
            mesh, None, (map_Kd, map_Ks), None,
            c2ws, intrinsics, render_size,
            render_world_normal=True,
            render_world_position=True,
            render_v_attr=False,
            render_uv=True,
            render_map_attr=True,
            **kwargs,
        )
        alpha = render_result['alpha']
        world_position = render_result['world_position']
        world_normal = render_result['world_normal']
        image_Kd, image_Ks = render_result['map_attr'].split([map_Kd.shape[-1], map_Ks.shape[-1]], dim=-1)
        view_position = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
        render_result = {
            'alpha': alpha,
            'world_position': world_position,
            'world_normal': world_normal,
            'map_Kd': image_Kd,
            'map_Ks': image_Ks,
            'view_position': view_position,
        }
        self.cache.append(render_result)
        return render_result
    
    @property
    def index_list(self):
        length = len(self.cache)
        return list(reversed(range(-length, length)))
    
    def render_pbr(
        self, index=-1, 
        lambda_albedo_r=1.0,
        lambda_albedo_g=1.0,
        lambda_albedo_b=1.0,
        lambda_matellic=1.0,
        lambda_roughness=1.0,
        lambda_diffuse=1.0, 
        lambda_specular=1.0,
    ):
        render_result = self.cache[index]
        map_Kd = render_result['map_Kd']
        map_Ks = render_result['map_Ks']
        map_Kd_scale = torch.tensor([lambda_albedo_r, lambda_albedo_g, lambda_albedo_b, 1.0], dtype=map_Kd.dtype, device=map_Kd.device)
        map_Ks_scale = torch.tensor([1.0, lambda_roughness, lambda_matellic], dtype=map_Ks.dtype, device=map_Ks.device)
        diffuse, specular = self.pbr_model(
            view_position=render_result['view_position'],
            world_position=render_result['world_position'],
            world_normal=render_result['world_normal'],
            map_Kd=map_Kd_scale * map_Kd,
            map_Ks=map_Ks_scale * map_Ks,
        )
        background = render_result['background']
        gb_map_attr = lambda_diffuse * diffuse + lambda_specular * specular
        if background is not None:
            if isinstance(background, float):
                gb_map_attr = torch.lerp(torch.full_like(gb_map_attr, fill_value=background), gb_map_attr, render_result['alpha'])
            elif isinstance(background, torch.Tensor):
                gb_map_attr = torch.lerp(background.to(gb_map_attr).expand_as(gb_map_attr), gb_map_attr, render_result['alpha'])
            else:
                raise NotImplementedError
        render_result = {
            'diffuse': diffuse,
            'specular': specular,
            'rgb': gb_map_attr,
        }
        return render_result

    def clear_cache(self):
        self.cache = []

