'''
rotation augment
'''
import os
from typing import Tuple
from PIL import Image
import numpy as np
import cv2
import trimesh


################ begin: global variables ################
c2w_0 = np.array([
    [ 1, 0, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, -1, 0, 0],
    [ 0, 0, 0, 1],
], dtype=np.float32)
c2w_0_inv = np.array([
    [ 1, 0, 0, 0],
    [ 0, 0, -1, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, 0, 1],
], dtype=np.float32)
radius = 2.8
c2ws = np.array([
    [[ 1.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  1.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  1.0000,  radius],
    [ 0.0000,  0.0000,  0.0000,  1.0000]],

    [[ 0.0000,  0.0000,  1.0000,  radius],
    [ 0.0000,  1.0000,  0.0000,  0.0000],
    [-1.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  1.0000]],

    [[-1.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  1.0000,  0.0000,  0.0000],
    [-0.0000, -0.0000, -1.0000, -radius],
    [ 0.0000,  0.0000,  0.0000,  1.0000]],

    [[ 0.0000,  0.0000, -1.0000, -radius],
    [ 0.0000,  1.0000,  0.0000, -0.0000],
    [ 1.0000, -0.0000,  0.0000, -0.0000],
    [ 0.0000,  0.0000,  0.0000,  1.0000]],

    [[ 1.0000,  0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  1.0000,  radius],
    [ 0.0000, -1.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  1.0000]],

    [[-1.0000,  0.0000, -0.0000, -0.0000],
    [-0.0000, -0.0000, -1.0000, -radius],
    [-0.0000, -1.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  0.0000,  1.0000]]
], dtype=np.float32)
axes_index = {
    'X': 0,
    'Y': 1,
    'Z': 2,
}
eulers_degree = np.array([
    [ 0.0000,  0.0000,  0.0000],
    [ 0.0000,  270.0000,  0.0000],
    [ 0.0000,  180.0000,  0.0000],
    [ 0.0000,  90.0000,  0.0000],
    [ 90.0000,  0.0000,  0.0000],
    [ 270.0000,  0.0000,  0.0000]
], dtype=np.float32)
eulers_degree_2d = np.array([
    0.0000,
    90.0000,
    180.0000,
    270.0000
], dtype=np.float32)
eulers_degree_2d_to_3d = np.array([
    [ 0.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  90.0000],
    [ 0.0000,  0.0000,  180.0000],
    [ 0.0000,  0.0000,  270.0000]
], dtype=np.float32)
xys = np.array([
    [ 1.0000,  1.0000],
    [ -1.0000,  1.0000],
    [ -1.0000,  -1.0000],
    [ 1.0000,  -1.0000]
], dtype=np.float32)
xyzs = np.array([
    [
        [ 1.0000,  1.0000, 1.0000],
        [ -1.0000,  1.0000, 1.0000],
        [ -1.0000,  -1.0000, 1.0000],
        [ 1.0000,  -1.0000, 1.0000]
    ],
    [
        [ 1.0000,  1.0000,  -1.0000],
        [ 1.0000,  1.0000,  1.0000],
        [ 1.0000,  -1.0000,  1.0000],
        [ 1.0000,  -1.0000,  -1.0000]
    ],
    [
        [ -1.0000,  1.0000,  -1.0000],
        [ 1.0000,  1.0000,  -1.0000],
        [ 1.0000,  -1.0000,  -1.0000],
        [ -1.0000,  -1.0000,  -1.0000]
    ],
    [
        [ -1.0000,  1.0000,  1.0000],
        [ -1.0000,  1.0000,  -1.0000],
        [ -1.0000,  -1.0000,  -1.0000],
        [ -1.0000,  -1.0000,  1.0000]
    ],
    [
        [ 1.0000,  1.0000,  -1.0000],
        [ -1.0000,  1.0000,  -1.0000],
        [ -1.0000,  1.0000,  1.0000],
        [ 1.0000,  1.0000,  1.0000]
    ],
    [
        [ 1.0000,  -1.0000,  1.0000],
        [ -1.0000,  -1.0000,  1.0000],
        [ -1.0000,  -1.0000,  -1.0000],
        [ 1.0000,  -1.0000,  -1.0000]
    ],
], dtype=np.float32)
# xyzs_center = xyzs.mean(axis=-2)
xyzs_center = np.array([
    [ 0.0000,  0.0000,  1.0000],
    [ 1.0000,  0.0000,  0.0000],
    [ 0.0000,  0.0000,  -1.0000],
    [ -1.0000,  0.0000,  0.0000],
    [ 0.0000,  1.0000,  0.0000],
    [ 0.0000,  -1.0000,  0.0000]
], dtype=np.float32)
################ end: global variables ################


def apply_c2w(vertices:np.ndarray, c2ws:np.ndarray):
    '''
    vertices: [..., 3]
    c2ws: [..., 4, 4]
    '''
    vertices = np.concatenate([vertices, np.ones_like(vertices[..., [0]])], axis=-1)
    vertices = np.matmul(vertices[..., None, :], c2ws.swapaxes(-1, -2)).squeeze(-2)
    vertices = vertices[..., :3]
    return vertices

def apply_rotation(vertices:np.ndarray, rotations:np.ndarray):
    '''
    vertices: [..., 3]
    rotations: [..., 3, 3]
    '''
    vertices = np.matmul(vertices[..., None, :], rotations.swapaxes(-1, -2)).squeeze(-2)
    return vertices

def euler_to_rotations(euler_degree:np.ndarray, axes='XYZ') -> np.ndarray:
    euler = np.radians(euler_degree)
    cos = np.cos(euler)
    sin = np.sin(euler)
    ones = np.ones_like(euler)
    zeros = np.zeros_like(euler)
    axis_rotation = np.stack([
        ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos,
        cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos,
        cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones,
    ], axis=-1).reshape(*euler.shape, 3, 3, 3)
    return np.linalg.multi_dot([axis_rotation[..., idx, axes_index[axis], :, :] for idx, axis in enumerate(axes.upper())])

def euler_to_index(euler_degree:np.ndarray, axes='XYZ') -> Tuple[np.ndarray, np.ndarray]:
    vertices = apply_euler_3d(xyzs, -euler_degree, axes=axes)
    vertices_center = vertices.mean(axis=-2)
    vertices_center_chamfer_distance = np.sum(np.square(vertices_center[..., :, None, :] - xyzs_center[..., None, :, :]), axis=-1)
    index_0 = np.argmin(vertices_center_chamfer_distance, axis=-1)
    xyzs_remapped = xyzs[..., index_0, :, :]
    vertices_delta_chamfer_distance_0 = np.sum(np.square(vertices[..., :, :] - xyzs_remapped[..., [0], :]), axis=-1)
    index_1 = np.argmin(vertices_delta_chamfer_distance_0, axis=-1)
    return index_0, index_1

def apply_euler_2d(images:np.ndarray, euler_degree:np.ndarray, axes='XYZ') -> np.ndarray:
    N, H, W, C = images.shape
    index_0, index_1 = euler_to_index(euler_degree=euler_degree, axes=axes)
    images = images[index_0, :, :, :]
    for idx in range(N):
        image = images[idx, :, :, :]
        euler_degree_2d = eulers_degree_2d[index_1[idx]]
        c2w_2d = cv2.getRotationMatrix2D((W // 2, H // 2), euler_degree_2d, 1.0)
        image = cv2.warpAffine(image, c2w_2d, (W, H))
        images[idx, :, :, :] = image
    return images

def apply_euler_3d(vertices:np.ndarray, euler_degree:np.ndarray, axes='XYZ'):
    return apply_rotation(vertices=vertices, rotations=euler_to_rotations(euler_degree, axes=axes))

def ccms_to_vertices(ccms:np.ndarray) -> np.ndarray:
    rgbs = (ccms[..., :3] / 255.0) * 2.0 - 1.0
    alphas = ccms[..., [3]] / 255.0
    vertices = rgbs[(alphas > 0).repeat(3, axis=-1)].reshape(-1, 3)
    vertices = apply_c2w(vertices, c2w_0)
    return vertices

def vertices_to_ccms(ccms:np.ndarray, vertices:np.ndarray) -> np.ndarray:
    rgbs = (ccms[..., :3] / 255.0) * 2.0 - 1.0
    alphas = ccms[..., [3]] / 255.0
    vertices = apply_c2w(vertices, np.linalg.inv(c2w_0))
    rgbs[(alphas > 0).repeat(3, axis=-1)] = vertices.reshape(-1)
    ccms[..., :3] = (np.clip(rgbs * 0.5 + 0.5, 0.0, 1.0) * 255).astype(np.uint8)
    return ccms

def augment_images(input_path, output_path, euler_degree:np.ndarray, axes='XYZ'):
    ccms = np.stack([np.array(Image.open(os.path.join(input_path, f'{i:04d}_nocs.png')).convert('RGBA')) for i in range(6)], axis=0)
    albedos = np.stack([np.array(Image.open(os.path.join(input_path, f'{i:04d}_albedo.png')).convert('RGBA')) for i in range(6)], axis=0)

    vertices = ccms_to_vertices(ccms)
    vertices = apply_euler_3d(vertices, euler_degree, axes=axes)
    ccms = vertices_to_ccms(ccms, vertices)
    ccms = apply_euler_2d(ccms, euler_degree, axes=axes)
    albedos = apply_euler_2d(albedos, euler_degree, axes=axes)

    os.makedirs(output_path, exist_ok=True)
    trimesh.Trimesh(vertices, process=False).export(os.path.join(output_path, 'pcd.ply'))
    for idx, (ccm, albedo) in enumerate(zip(ccms, albedos)):
        cv2.imwrite(os.path.join(output_path, f'{idx:04d}_nocs.png'), ccm[..., [2,1,0,3]])
        cv2.imwrite(os.path.join(output_path, f'{idx:04d}_albedo.png'), albedo[..., [2,1,0,3]])


if __name__ == '__main__':
    input_path = 'test_data/sketchfab_free_geometry_diffusion/glb_render/000-002/4377f80a646a4d12aa58f2f29fabf405/'
    output_dir = 'test_result/test_augment'

    euler_degree_list = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([90.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 90.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 90.0], dtype=np.float32),
        np.array([180.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 180.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 180.0], dtype=np.float32),
        np.array([270.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 270.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 270.0], dtype=np.float32),
    ]
    for euler_degree in euler_degree_list:
        output_path = os.path.join(output_dir, f'xyz-' + '-'.join([f'{int(angle)}' for angle in euler_degree]))
        augment_images(input_path, output_path, euler_degree)

