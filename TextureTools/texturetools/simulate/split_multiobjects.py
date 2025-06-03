import os
from typing import List
from tqdm import tqdm
import numpy as np
import trimesh

from ..mesh.trimesh_utils import align_to_bbox


# NOTE: pip install python-fcl
def merge_collision(scene:trimesh.Scene, distance=0.0) -> List[trimesh.Scene]:
    '''
    scene: trimesh.Scene
    distance: distance threhold for collision detection
    scene_list: scenes separated from each other
    '''
    if len(scene.graph.nodes_geometry) == 0:
        return []
    elif len(scene.graph.nodes_geometry) == 1:
        return [scene]
    mesh_dict = dict()
    node_name_list_dict = dict()
    cm = trimesh.collision.CollisionManager()
    if distance > 0.0:
        _is_collision = lambda: cm.min_distance_internal() < distance
    else:
        _is_collision = lambda: cm.in_collision_internal()
    for node_idx, node_name in enumerate(tqdm(scene.graph.nodes_geometry, desc='detect collision')):
        transform, geom_name = scene.graph[node_name]
        mesh = scene.geometry[geom_name].copy()
        if hasattr(mesh, "to_3D"):
            check = trimesh.util.isclose(transform, trimesh.util._IDENTITY, atol=1e-8)
            check[:2, :3] = True
            if not check.all():
                mesh = mesh.to_3D()
            else:
                transform = transform[:3, :3]
        mesh.apply_transform(transform)
        node_idx_list = [node_idx]
        node_name_list = [node_name]
        if len(mesh_dict) > 0:
            cm.add_object(str(node_idx), mesh)
            for _node_idx, _mesh in mesh_dict.items():
                cm.add_object(str(_node_idx), _mesh)
                if _is_collision():
                    node_idx_list.append(_node_idx)
                    node_name_list.extend(node_name_list_dict[_node_idx])
                    mesh = trimesh.util.concatenate(mesh, _mesh)
                cm.remove_object(str(_node_idx))
            cm.remove_object(str(node_idx))
            for _node_idx in node_idx_list[1:]:
                mesh_dict.pop(_node_idx)
                node_name_list_dict.pop(_node_idx)
        mesh_dict[node_idx] = mesh
        node_name_list_dict[node_idx] = node_name_list
    del cm, mesh_dict
    scene_list = []
    for node_idx, node_name_list in tqdm(node_name_list_dict.items(), desc='reconstruct collision'):
        graph = scene.graph.copy()
        for node_name in set(scene.graph.nodes_geometry) - set(node_name_list):
            graph.transforms.remove_node(node_name)
        geometry = {geom_name: scene.geometry[geom_name].copy() for geom_name in graph.geometry_nodes.keys()}
        metadata = scene.metadata.copy()
        if scene.has_camera:
            camera = scene.camera.copy()
        else:
            camera = None
        scene_list.append(trimesh.Scene(geometry=geometry, graph=graph, metadata=metadata, camera=camera))
    return scene_list


def process(mesh_path_src, mesh_path_dst):
    scene = trimesh.load(mesh_path_src, force='scene')
    scene = align_to_bbox(scene, scale=1.0)
    scene_list = merge_collision(scene, distance=0.001)
    os.makedirs(os.path.dirname(mesh_path_dst), exist_ok=True)
    for _idx, _scene in enumerate(scene_list):
        _scene.export(mesh_path_dst + '.' + str(_idx) + '.glb')
    scene.export(mesh_path_dst)


if __name__ == '__main__':
    path_src = '/home/chenxiao/下载/0206/examples/45693886.glb'
    path_dst = '/home/chenxiao/下载/0206/examples/45693886_a.glb'
    process(path_src, path_dst)


