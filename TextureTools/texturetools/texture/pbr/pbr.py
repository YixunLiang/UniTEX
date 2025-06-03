import math
import os
from typing import Optional, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
# NOTE: ~/.cache/torch_extensions/pyxxx_cuxxx/renderutils_plugin
from . import renderutils as ru

def dot(x:torch.Tensor, y:torch.Tensor, dim=-1):
    return (x * y).sum(dim=dim, keepdim=True)

def reflect(x:torch.Tensor, n:torch.Tensor, dim=-1):
    return 2 * dot(x, n, dim=dim) * n - x

def cube_to_dir(s, x, y):
    # x: origin to front, y: origin to top, z: origin to right
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x  # front
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x  # back
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y    # top
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y  # down
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)   # right
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x) # left
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    '''
    latlong_map: [Hi, Wi, 3]
    res: Ho, Wo
    cubemap: [6, Ho, Wo, 3]

    http://www.paulbourke.net/panorama/cubemaps/
    '''
    device = latlong_map.device
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                                indexing='ij')
        v = torch.nn.functional.normalize(cube_to_dir(s, gx, gy), dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5  # phi in range(-pi, pi), scale to (0, 1)
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi  # theta in range(0, pi), scale to (0, 1)
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    '''
    cubemap: [6, Hi, Wi, 3]
    res: Ho, Wo
    latlong_map: [Ho, Wo, 3]

    NOTE: cubemap_to_latlong and latlong_to_cubemap are not reciprocal
    '''
    device = cubemap.device
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)  # theta in range(0, pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)  # phi in range(-pi, pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = torch.nn.functional.normalize(cube_to_dir(s, gx, gy), dim=-1)
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

class PBRModel(nn.Module):
    def __init__(self, env_hdr_path:Optional[str]=None, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.FG_LUT = torch.as_tensor(np.fromfile(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envmaps', 'bsdf_256_256.bin'), 
            dtype=np.float32,
        ).reshape(1, 256, 256, 2), dtype=torch.float32, device=self.device)
        if env_hdr_path is not None:
            latlong = cv2.imread(env_hdr_path, -1)
            if latlong.dtype == np.uint8:
                latlong = latlong / 255
            latlong = torch.as_tensor(latlong, dtype=torch.float32, device=self.device)
        else:
            latlong = torch.ones((4, 4, 3), dtype=torch.float32, device=self.device)
        cubemap = latlong_to_cubemap(latlong, [512, 512])
        self.light_diffuse = ru.diffuse_cubemap(cubemap)
        self.light_specular = ru.specular_cubemap(cubemap, roughness=0.08, cutoff=0.99)

    def forward(self, view_position, world_position, world_normal, map_Kd, map_Ks):
        nrm = torch.nn.functional.normalize(world_normal, dim=-1)
        wo = torch.nn.functional.normalize(view_position - world_position, dim=-1)
        wi = torch.nn.functional.normalize(reflect(wo, nrm), dim=-1)
        cos_wo_nrm = torch.clamp(dot(wo, nrm), min=0.0, max=1.0)
        
        # https://thinkmoult.com/radiance-specularity-and-roughness-value-examples.html
        map_albedo = map_Kd[..., :3]
        map_specular = torch.full_like(map_Ks[..., [0]], fill_value=0.5)
        map_roughness = map_Ks[..., [1]]
        map_metallic  = map_Ks[..., [2]]
        coefficient_diffuse  = map_albedo * (1.0 - map_metallic) + 0.0 * map_metallic
        coefficient_specular  = (0.04 * (1.0 - map_metallic) + map_albedo * map_metallic) * (1.0 - map_specular) + 0.0 * map_specular

        diffuse_lookup = dr.texture(self.light_diffuse.unsqueeze(0), nrm.contiguous(), filter_mode='linear', boundary_mode='cube')
        specular_lookup = dr.texture(self.light_specular.unsqueeze(0), wi.contiguous(), filter_mode='linear', boundary_mode='cube')
        fg_lookup = dr.texture(self.FG_LUT, torch.cat([cos_wo_nrm, map_roughness], dim=-1), filter_mode='linear', boundary_mode='clamp')

        diffuse = coefficient_diffuse * diffuse_lookup
        specular = (coefficient_specular * fg_lookup[...,0:1] + fg_lookup[...,1:2]) * specular_lookup
        return diffuse, specular

