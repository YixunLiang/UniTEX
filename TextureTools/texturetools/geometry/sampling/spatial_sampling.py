from typing import Optional, Tuple
import torch

# https://github.com/ashawkey/cubvh.git
# https://github.com/opencv/opencv/issues/14868
from cubvh import cuBVH

from .surface_sampling import sample_surface


def sample_spatial(
    vertices:torch.Tensor,
    faces:torch.Tensor,
    bvh:Optional[cuBVH]=None,
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    uniform sampling in spatial bbox by probability 
    and return projection of samples on surface faces

    vertices: [V, 3], float32
    faces: [F, 3], int64
    bvh: cuBVH
    N: num of samples on mesh

    samples: [N, 3], float32
    face_index: [N,], int64
    face_uvw: [N, 3], float32
    '''
    generator = None if seed is None else torch.Generator(device='cuda').manual_seed(seed)
    if bvh is None:
        # TODO: cubvh build bvh tree on cpu with one thread, too slow
        bvh = cuBVH(vertices=vertices, triangles=faces)
    samples = torch.rand((N, 3), dtype=torch.float32, device='cuda', generator=generator)
    udfs, face_index, face_uvw = bvh.unsigned_distance(samples, return_uvw=True)
    return samples, face_index, face_uvw


def sample_near_surface(
    vertices:torch.Tensor,
    faces:torch.Tensor,
    areas:Optional[torch.Tensor]=None,
    normals:Optional[torch.Tensor]=None,
    bvh:Optional[cuBVH]=None,
    N=10_000_000,
    N_radio:float=1.1,
    seed:Optional[int]=666,
    distance_threhold:float=1.0,
    depth:int=8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    uniform sampling near suface in spatial bbox by probability 
    and return projection of samples on surface faces

    vertices: [V, 3], float32
    faces: [F, 3], int64
    bvh: cuBVH
    N: num of samples on mesh

    samples: [N, 3], float32
    face_index: [N,], int64
    face_uvw: [N, 3], float32
    '''
    generator = None if seed is None else torch.Generator(device='cuda').manual_seed(seed)
    distance_threhold = distance_threhold * (2.0 / (2 ** depth))
    if areas is None:
        areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
    if normals is None:
        normals = torch.nn.functional.normalize(areas, dim=-1)
    vertex_normal = torch.zeros((vertices.shape[0], 3, 3),dtype=vertices.dtype, device=vertices.device)
    vertex_normal.scatter_add_(0, faces.unsqueeze(-1).expand(faces.shape[0], 3, 3), normals.unsqueeze(1).expand(faces.shape[0], 3, 3))
    vertex_normal = torch.nn.functional.normalize(vertex_normal.sum(dim=1), dim=-1)
    if bvh is None:
        # TODO: cubvh build bvh tree on cpu with one thread, too slow
        bvh = cuBVH(vertices=vertices, triangles=faces)
    samples, face_index, face_uvw = sample_surface(
        vertices=vertices,
        faces=faces,
        areas=areas,
        N=int(N * N_radio),
        seed=seed,
    )
    # NOTE: maybe less than N
    samples = samples[:N, :]
    face_index = face_index[:N]
    face_uvw = face_uvw[:N, :]
    samples_normal = (vertex_normal[faces[face_index, :], :] * face_uvw.unsqueeze(-1)).sum(dim=1)
    deltas = torch.rand((N, 3), dtype=torch.float32, device='cuda', generator=generator).mul(2.0).sub(1.0)
    samples = samples + distance_threhold * deltas * samples_normal
    udfs, face_index, face_uvw = bvh.unsigned_distance(samples, return_uvw=True)
    return samples, face_index, face_uvw

