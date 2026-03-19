from typing import Optional, Tuple
import torch


def sample_surface(
    vertices:torch.Tensor,
    faces:torch.Tensor,
    areas:Optional[torch.Tensor]=None,
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    uniform sampling on faces by probability

    vertices: [V, 3], float32
    faces: [F, 3], int64
    areas: [F,], float32, directed
    N: num of samples on mesh

    samples: [N, 3], float32
    face_index: [N,], int64
    face_uvw: [N, 3], float32
    '''
    generator = None if seed is None else torch.Generator(device='cuda').manual_seed(seed)
    if areas is None:
        areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
    weight_cum = torch.cumsum(torch.norm(areas, p=2, dim=-1), dim=0)
    face_pick = torch.rand((N,), dtype=torch.float32, device='cuda', generator=generator) * weight_cum[-1]
    face_index = torch.searchsorted(weight_cum, face_pick)
    face_uv = torch.rand((N, 2), dtype=torch.float32, device='cuda', generator=generator)
    face_uv[face_uv.sum(dim=-1) > 1.0, :] -= 1.0
    face_uv = torch.abs(face_uv)
    face_uvw = torch.cat([face_uv, 1.0 - face_uv.sum(-1, keepdim=True)], dim=-1)
    samples = (vertices[faces[face_index, :], :] * face_uvw.unsqueeze(-1)).sum(dim=1)
    return samples, face_index, face_uvw

