from typing import Optional, Union
import trimesh
from .mesh_header_loader import parse_mesh_info


def convert_to_whole_mesh(scene:Union[trimesh.Trimesh, trimesh.Scene]):
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        # NOTE: bake scene.graph to scene.geometry
        geometry = scene.dump()
        if len(geometry) == 1:
            mesh = geometry[0]
        else:  # NOTE: missing some attributes
            mesh = trimesh.util.concatenate(geometry)
    else:
        raise ValueError(f"Unknown mesh type.")
    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    return mesh


def load_whole_mesh(mesh_path, limited_faces:Optional[int]=10_000_000) -> trimesh.Trimesh:
    # NOTE: skip large file by header
    if limited_faces is not None:
        num_faces = parse_mesh_info(mesh_path)['F']
        assert num_faces <= limited_faces, \
            f'num faces {num_faces} is larger than limited_faces {limited_faces}'
    scene = trimesh.load(mesh_path, process=False)
    mesh = convert_to_whole_mesh(scene)
    return mesh

