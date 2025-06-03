from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import trimesh
from ...mesh.structure_v2 import PBRMesh, PBRScene, trimesh_to_pbr_mesh
from ..sampling.surface_sampling import sample_surface


def sample_pbr_mesh(
    pbr_mesh:Union[PBRMesh, List[PBRMesh], PBRScene, List[PBRScene]],
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    '''
    pbr_mesh: PBRMesh, list[PBRMesh], PBRScene, list[PBRScene]
    N: num of samples on mesh
    samples: [N, 3], float32
    face_index: [N,], int64
    face_attr: Dict[str, [N, C]], float32
    '''
    if not (isinstance(pbr_mesh, PBRMesh) or isinstance(pbr_mesh, PBRScene)):
        pbr_mesh = PBRScene(pbr_mesh)
    samples, face_index, face_uvw = sample_surface(pbr_mesh.vertices, pbr_mesh.faces, pbr_mesh.areas, N=N, seed=seed)
    face_attr = pbr_mesh(face_index, face_uvw)
    return samples, face_index, face_attr


def sample_trimesh(
    mesh:Union[trimesh.Trimesh, trimesh.Scene],
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    '''
    mesh: trimesh.Trimesh, trimesh.Scene
    N: num of samples on mesh
    samples: [N, 3], float32
    face_index: [N,], int64
    face_attr: Dict[str, [N, C]], float32
    '''
    pbr_mesh = trimesh_to_pbr_mesh(mesh)
    samples, face_index, face_attr = sample_pbr_mesh(pbr_mesh, N=N, seed=seed)
    samples, face_index, face_attr = samples.detach().cpu().numpy(), face_index.cpu().numpy(), {k: v.detach().cpu().numpy() for k, v in face_attr.items()}
    return samples, face_index, face_attr

