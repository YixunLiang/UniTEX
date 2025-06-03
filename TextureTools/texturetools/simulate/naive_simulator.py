from glob import glob
import json
import os
from time import perf_counter
from typing import Dict, List
import numpy as np
import trimesh


def move_bboxes_to_floor_v1(bboxes:np.ndarray, height=-10.0, dim=1):
    '''
    bboxes: [N, 2, 3], aaabbb
    trans: [N, 3], xyz
    '''
    N, _, _ = bboxes.shape
    assert dim in [0, 1, 2]
    trans = np.zeros((N, 3), dtype=np.float32)
    trans[:, dim] = - bboxes[:, 0, dim] + height
    return trans


def move_bboxes_to_ceil_v1(bboxes:np.ndarray, ceil_height=10.0, dim=1):
    '''
    bboxes: [N, 2, 3], aaabbb
    trans: [N, 3], xyz
    '''
    N, _, _ = bboxes.shape
    assert dim in [0, 1, 2]
    trans = np.zeros((N, 3), dtype=np.float32)
    trans[:, dim] = - bboxes[:, 1, dim] + ceil_height
    return trans


def intersect(bbox_1, bbox_2, dim=1):
    '''
    bbox: [2, 3]
    '''
    dims = [[1, 2], [2, 0], [0, 1]][dim]
    top = np.array([bbox_1[1, dims[0]], bbox_1[1, dims[1]], bbox_2[1, dims[0]], bbox_2[1, dims[1]]], dtype=np.float32).reshape(2, 2)
    down = np.array([bbox_1[0, dims[0]], bbox_1[0, dims[1]], bbox_2[0, dims[0]], bbox_2[0, dims[1]]], dtype=np.float32).reshape(2, 2)
    return np.prod(np.clip(np.min(top, axis=0) - np.max(down, axis=0), a_min=0, a_max=None)) > 0.0


def move_bboxes_to_floor_v2(bboxes:np.ndarray, floor_height=-10.0, dim=1):
    '''
    bboxes: [N, 2, 3], aaabbb
    trans: [N, 3], xyz
    '''
    N, _, _ = bboxes.shape
    assert dim in [0, 1, 2]
    indices = np.argsort(bboxes[:, 0, dim])
    trans = np.zeros((N, 3), dtype=np.float32)
    for i, j in enumerate(indices):
        bbox = bboxes[j, :, :]
        if i == 0:
            trans[j, dim] = - bbox[0, dim] + floor_height
            continue
        height = floor_height
        for bbox_extra, trans_extra in zip(bboxes[indices[:i], :, :], trans[indices[:i], :]):
            if intersect(bbox, bbox_extra, dim=dim):
                if bbox[0, dim] > bbox_extra[1, dim] + trans_extra[dim]:
                    height = max(height, bbox_extra[1, dim] + trans_extra[dim])
        trans[j, dim] = - bbox[0, dim] + height
    return trans


def move_bboxes_to_ceil_v2(bboxes:np.ndarray, ceil_height=10.0, dim=1):
    '''
    bboxes: [N, 2, 3], aaabbb
    trans: [N, 3], xyz
    '''
    N, _, _ = bboxes.shape
    assert dim in [0, 1, 2]
    indices = np.argsort(bboxes[:, 0, dim])
    trans = np.zeros((N, 3), dtype=np.float32)
    for i, j in enumerate(indices):
        bbox = bboxes[j, :, :]
        if i == 0:
            trans[j, dim] = - bbox[1, dim] + ceil_height
            continue
        height = ceil_height
        for bbox_extra, trans_extra in zip(bboxes[indices[:i], :, :], trans[indices[:i], :]):
            if intersect(bbox, bbox_extra, dim=dim):
                if bbox[1, dim] < bbox_extra[0, dim] + trans_extra[dim]:
                    height = min(height, bbox_extra[0, dim] + trans_extra[dim])
        trans[j, dim] = - bbox[1, dim] + height
    return trans


def load_quad_mesh(mesh_path):
    # NOTE: this function is only used for wall saved by scene generation pipeline
    vertices = []
    faces_quad = []
    faces_others = []
    with open(mesh_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        splits = line.split(' ')
        name = splits[0]
        others = splits[1:]
        if name == 'v':
            vertices.append([float(v) for v in others])
        elif name == 'f':
            if len(others) == 4:
                faces_quad.append([int(f) for f in others])
            else:
                faces_others.append(np.asarray([int(f) for f in others], dtype=np.int64))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces_quad = np.asarray(faces_quad, dtype=np.int64)
    return {
        'vertices': vertices, 
        'faces_quad': faces_quad, 
        'faces_others': faces_others, 
    }


def points_to_lines(points:np.ndarray, lines:np.ndarray):
    '''
    points: [N, 2], (x, y)
    lines: [M, 2, 2], [[x1, y1], [x2, y2]], line from (x1, y1) to (x2, y2)
    distances: [N, M], directed
    '''
    d = lines[:, 1, :] - lines[:, 0, :]
    d_norm = np.linalg.norm(d, axis=-1, keepdims=False)
    v = points[:, None, :] - lines[:, 0]
    areas = np.cross(d, v)
    distances = np.nan_to_num(np.divide(areas, d_norm), nan=-1.0, posinf=-1.0, neginf=-1.0) 
    return distances


def bbox_simulate(
    scene:trimesh.Scene, 
    labels:np.ndarray, 
    heights:np.ndarray, 
) -> trimesh.Scene:
    '''
    labels: [N, 3], up, down, auto-simulation on forward/backward/right/left
    heights: [6,], (x1, y1, z1, x2, y2, z2) of wall
    walls: [2, M, 3], (x, y, z) of ceil/floor, num, x/y/z
    NOTE: prior is up/down > forward/backward/right/left
    '''
    N = len(scene.geometry)
    # NOTE: bboxes: [N, 2, 3], num, aaabbb
    if N == 0:
        return trimesh.Scene()
    else:
        bboxes = np.stack([mesh.bounding_box.bounds for mesh in scene.geometry.values()], axis=0)
    trans = np.zeros((N, 3), dtype=np.float32)

    # up, down
    ceil, floor = np.where(labels[:, 0])[0], np.where(labels[:, 1])[0]
    if len(ceil) > 0:
        trans[ceil, :] += move_bboxes_to_ceil_v2(bboxes[ceil, :, :], ceil_height=heights[0], dim=1)
    if len(floor) > 0:
        trans[floor, :] += move_bboxes_to_floor_v2(bboxes[floor, :, :], floor_height=heights[1], dim=1)
    bboxes[:, :, 1] += trans[:, [1]]

    for (mesh, translation) in zip(scene.geometry.values(), trans):
        mesh.apply_translation(translation)
    return scene


def compute_wall_scale(scene:trimesh.Scene, wall_mesh:trimesh.Trimesh):
    bboxes = np.stack([mesh.bounding_box.bounds for mesh in scene.geometry.values()], axis=0)  # [N, 2, 3]
    cubes = bboxes[:, [0], :] + np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
    ], dtype=np.float32) * (bboxes[:, [1], :] - bboxes[:, [0], :])  # [N, 8, 3]
    for scale_reciprocal in np.linspace(2.0, 0.5, 20):
        isin = wall_mesh.ray.contains_points(scale_reciprocal * cubes.reshape(-1, 3)).reshape(-1, 8)
        if isin.all():
            break
    scale = np.reciprocal(scale_reciprocal)
    return scale


# up, down, auto-simulation on forward/backward/right/left
DEFAULT_LABEL = [False, False, False]
LABEL_DICT_PATH = os.environ.get('LABEL_DICT_PATH', os.path.join(os.path.dirname(__file__), 'label_dict.json'))
MISSING_LABEL_PATH = os.environ.get('MISSING_LABEL_PATH', os.path.join(os.path.dirname(__file__), 'missing_label.json'))
with open(LABEL_DICT_PATH, 'r') as f:
    LABEL_DICT = json.load(f)
os.makedirs(os.path.dirname(MISSING_LABEL_PATH), exist_ok=True)


if __name__ == '__main__':
    # test_root = '/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/'
    test_root = '/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/batch_testing/final_select_examples/living_rooms/easy/0005_4/'
    scene_path = os.path.join(test_root, 'scene_pos_scale_refine.glb')
    wall_geometry_path = os.path.join(test_root, 'pano_estimation/layout_mesh_low_poly.obj')
    wall_texture_path = os.path.join(test_root, 'pano_estimation/layout_mesh_low_poly_with_texture.obj')
    final_scene_path = os.path.join(test_root, 'final_scene.glb')

    # load meshes, labels, wall
    scene:trimesh.Scene = trimesh.load(scene_path, force='scene', process=False)
    labels = []
    for name in scene.geometry.keys():
        _type = name.split('(')[0].split('-')[0]
        if _type in LABEL_DICT.keys():
            labels.append(LABEL_DICT[_type])
        else:
            with open(MISSING_LABEL_PATH, 'a') as f:
                f.write(_type + '\n')
            labels.append(DEFAULT_LABEL)
            # raise ValueError(f'type {_type} is not in LABEL_DICT')
    labels = np.array(labels, dtype=bool)
    # wall_geometry = load_quad_mesh(wall_geometry_path)
    # walls:np.ndarray = wall_geometry['vertices'].reshape(2, -1, 3)  # [2, N, 3]
    wall_texture = trimesh.load(wall_texture_path, process=False, force='mesh')

    # compute scale
    # scale = compute_wall_scale(scene, wall_texture)
    # wall_texture = wall_texture.apply_scale(scale)
    # walls = walls * scale

    # simulate
    t = perf_counter()
    # NOTE: left, down, backward, right, up, forward ==> up, down, forward, backward, right, left
    heights = wall_texture.bounding_box.bounds.reshape(-1)[[4, 1, 5, 2, 3, 0]]
    scene = bbox_simulate(scene, labels=labels, heights=heights)
    print('bbox_simulate', perf_counter() - t)

    # export scene
    scene.add_geometry(wall_texture, geom_name='wall')
    scene.export(final_scene_path)

