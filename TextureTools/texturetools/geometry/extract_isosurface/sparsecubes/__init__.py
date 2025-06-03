from typing import Optional, Tuple
import torch
from .sparsecubes import SparseCubes

def extrac_isosurface(
    cube_vertices: torch.Tensor,
    cube_vertices_sdf: torch.Tensor,
    cube_faces: torch.Tensor,
    sdf_offset:Optional[float]=None, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    sc = SparseCubes()
    if sdf_offset is not None and sdf_offset != 0.0:
        cube_vertices_sdf = cube_vertices_sdf - sdf_offset
    vertices, faces, _ = sc(cube_vertices, cube_vertices_sdf, cube_faces)
    return vertices, faces

