'''
Convert mesh to sdf field with ray tracing and 
extract iso field to mesh with sparse grid marching cubes
'''
import math
from typing import Optional, Tuple
import torch

from ...raytracing import RayTracing
from ..triangle_topology.clean import remove_unreferenced_vertices


def mesh_to_gi(
    vertices:torch.Tensor,
    faces:torch.Tensor,
    normals:Optional[torch.Tensor]=None,
    uvs_2d:Optional[torch.Tensor]=None,
    faces_2d:Optional[torch.Tensor]=None,
    H=1024,
    W=1024,
    mode='sphere',
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    convert mesh to geometry image

    vertices: [V, 3], float32, normalized, range(-1.0, 1.0)
    faces: [F, 3], int64
    normals: [F, 3], float32, normalized face normal
    uvs_2d: [V_2d, 2], float32, normalized, range(-1.0, 1.0)
    faces_2d: [F_2d, 3], int64
    mode: sphere, shell, cylinder, uv

    uv_mask: [H, W, 1], bool
    uv_ccm: [H, W, 3], float32, range(-1.0, 1.0)
    uv_normal: [H, W, 3], float32, range(-1.0, 1.0)
    '''
    device = vertices.device

    if normals is None:
        areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
        normals = torch.nn.functional.normalize(areas, dim=-1)

    us = torch.linspace(-1.0, 1.0, W+1, dtype=torch.float32, device=device)[:W].add(1.0 / W)
    vs = torch.linspace(-1.0, 1.0, H+1, dtype=torch.float32, device=device)[:H].add(1.0 / H).unsqueeze(-1)
    if mode == 'sphere':
        bvh = RayTracing(vertices=vertices, faces=faces)
        rays_d = torch.stack([
            torch.sin(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.cos(us * torch.pi + torch.pi),
            torch.sin(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.sin(us * torch.pi + torch.pi),
            torch.cos(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.ones_like(us),
        ], dim=-1)
        rays_o = torch.zeros_like(rays_d)
        rays_mask, _, rays_tid, rays_h, rays_uv = bvh.intersects_closest(rays_o, rays_d)
        uv_mask = rays_mask.unsqueeze(-1)
        uv_ccm = rays_h
        uv_normal = normals[rays_tid, :]
    elif mode == 'shell':
        bvh = RayTracing(vertices=vertices, faces=faces)
        rays_d = torch.stack([
            torch.sin(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.cos(us * torch.pi + torch.pi),
            torch.sin(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.sin(us * torch.pi + torch.pi),
            torch.cos(vs * (torch.pi / 2.0) + (torch.pi / 2.0)) * torch.ones_like(us),
        ], dim=-1)
        rays_o = (- math.sqrt(3)) * rays_d
        rays_mask, _, rays_tid, rays_h, rays_uv = bvh.intersects_closest(rays_o, rays_d)
        uv_mask = rays_mask.unsqueeze(-1)
        uv_ccm = rays_h
        uv_normal = normals[rays_tid, :]
    elif mode == 'cylinder':
        bvh = RayTracing(vertices=vertices, faces=faces)
        rays_d = torch.stack([
            torch.ones_like(vs) * torch.cos(us * torch.pi + torch.pi),
            torch.ones_like(vs) * torch.sin(us * torch.pi + torch.pi),
            torch.zeros_like(vs) * torch.zeros_like(us),
        ], dim=-1)
        rays_o = torch.stack([
            torch.zeros_like(vs) * torch.zeros_like(us),
            torch.zeros_like(vs) * torch.zeros_like(us),
            vs * torch.ones_like(us),
        ], dim=-1)
        rays_mask, _, rays_tid, rays_h, rays_uv = bvh.intersects_closest(rays_o, rays_d)
        uv_mask = rays_mask.unsqueeze(-1)
        uv_ccm = rays_h
        uv_normal = normals[rays_tid, :]
    elif mode == 'uv':
        uvs_3d = torch.cat([uvs_2d, torch.zeros_like(uvs_2d[..., [0]])], dim=-1)
        bvh = RayTracing(vertices=uvs_3d, faces=faces_2d)
        rays_o = torch.stack([
            us * torch.ones_like(vs),
            torch.ones_like(us) * vs,
            torch.full_like(us, fill_value=-1.0) * torch.ones_like(vs),
        ], dim=-1)
        rays_d = torch.zeros_like(rays_o)
        rays_d[..., -1].fill_(1.0)
        rays_mask, _, rays_tid, rays_h, rays_uv = bvh.intersects_closest(rays_o, rays_d)
        rays_uv = torch.cat([rays_uv, 1.0 - rays_uv.sum(-1, keepdim=True)], dim=-1)
        rays_h_3d = torch.sum(vertices[faces[rays_tid, :], :] * rays_uv.unsqueeze(-1), dim=-2)
        uv_mask = rays_mask.unsqueeze(-1)
        uv_ccm = rays_h_3d
        uv_normal = normals[rays_tid, :]
    else:
        raise NotImplementedError(f'mode {mode} is not supported')
    return uv_mask, uv_ccm, uv_normal


def gi_to_mesh(
    uv_mask:torch.Tensor,
    uv_ccm:torch.Tensor,
    uv_attr:Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    '''
    convert geometry image to mesh

    vertices: [V, 3], float32, normalized, range(-1.0, 1.0)
    uvs: [V, 2], float32, normalized, range(-1.0, 1.0)
    faces: [F, 3], int64
    uv_ccm: [H, W, 3], float32, range(-1.0, 1.0)
    uv_mask: [H, W, 1], bool
    '''
    H, W, _ = uv_ccm.shape
    device = uv_ccm.device

    vertices = uv_ccm.reshape(-1, 3)
    if uv_attr is not None:
        attrs = uv_attr.reshape(-1, uv_attr.shape[-1])
    else:
        attrs = None
    ii = torch.arange(W, dtype=torch.int64, device=device).unsqueeze(-2)
    jj = torch.arange(H, dtype=torch.int64, device=device).unsqueeze(-1)
    indices_spatial = jj * W + ii
    uvs = torch.stack(torch.broadcast_tensors(ii / (W + 1), jj / (H + 1)), dim=-1).mul(2.0).sub(1.0).reshape(-1, 2)
    faces_offset = torch.as_tensor([
        [0, W, 1],
        [0, W+1, 1],
        [0, W, W+1],
        [1, W, W+1],
    ], dtype=torch.int64, device=device)
    uv_mask_all = uv_mask.all().item()
    if not uv_mask_all:
        kernel = torch.as_tensor([
            [1, 2], 
            [4, 8], 
        ], dtype=torch.int64, device=device)
        table = {
            7: [0],
            11: [1],
            13: [2],
            14: [3],
            15: [1, 2],
        }
        adjacent = torch.nn.functional.pad(uv_mask.permute(2, 0, 1).unsqueeze(0), (0, 1, 0, 1), mode='circular')
        adjacent = torch.nn.functional.conv2d(
            adjacent.to(dtype=torch.float32), 
            kernel.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
        ).to(dtype=torch.int64).squeeze(0).squeeze(0)
        adjacent_unique = torch.unique(adjacent.reshape(-1), dim=0).tolist()
        faces = []
        for adj_cur in adjacent_unique:
            if adj_cur in table.keys():
                idx_cur = indices_spatial.masked_select(adjacent == adj_cur)
                faces_cur = idx_cur.unsqueeze(-1).unsqueeze(-1) + faces_offset[table[adj_cur], :]
                faces += [faces_cur.reshape(-1, 3)]
        if len(faces) == 0:
            faces = None
        else:
            faces = torch.cat(faces, dim=0)
    else:
        faces = indices_spatial.unsqueeze(-1).unsqueeze(-1) + faces_offset[[1, 2], :]
        faces = faces.reshape(-1, 3)
    if faces is not None:
        faces = faces // W % H * W + faces % W % W
        if uv_mask_all:
            if attrs is not None:
                vertices = torch.cat([vertices, uvs, attrs], dim=-1)
            else:
                vertices = torch.cat([vertices, uvs], dim=-1)
            vertices, faces = remove_unreferenced_vertices(vertices, faces)
            if attrs is not None:
                vertices, uvs, attrs = torch.split(vertices, [3, 2, uv_attr.shape[-1]], dim=-1)
            else:
                vertices, uvs = torch.split(vertices, [3, 2], dim=-1)
    return vertices, uvs, faces, attrs


