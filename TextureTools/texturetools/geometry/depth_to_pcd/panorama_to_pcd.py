'''
Simple tools for converting panorama image to point cloud
'''

import os
from typing import List, Tuple, Union
import torch
import trimesh
import numpy as np
from PIL import Image


mode_to_dtype = {
    "I": np.int32, 
    "I;16": np.int16, 
    "I;16B": np.int16, 
    "F": np.float32,
}
dtype_to_fmax = {
    np.uint8: 255.0,
    np.int16: 65535.0,
    np.int32: 4294967295.0,
    np.float32: 1.0,
}
def parse_metric_depth(image: Union[torch.Tensor, np.ndarray, Image.Image, str]) -> np.ndarray:
    '''
    image: Any to [H, W, 1], float32, metric depth
    '''
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            ...
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        return image.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[:, :, None]
        elif image.ndim == 3:
            ...
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        return image.astype(np.float32) / dtype_to_fmax.get(image.dtype, 1.0)
    elif isinstance(image, Image.Image):
        image = np.array(image, dtype=np.float32) / dtype_to_fmax.get(mode_to_dtype.get(image.mode, np.uint8))
        if image.ndim == 2:
            return image[:, :, None]
        elif image.ndim == 3:
            return image
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
    elif isinstance(image, str):
        return parse_metric_depth(Image.open(image))
    else:
        raise NotImplementedError(f'type of image {type(image)} is not supported')

def parse_rgb(image: Union[torch.Tensor, np.ndarray, Image.Image, str]) -> np.ndarray:
    '''
    image: Any to [H, W, 3], float32, range(0.0, 1.0)
    '''
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.ndim == 3:
            ...
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        return image.permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        elif image.ndim == 3:
            ...
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        return np.clip(image, 0, 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        return parse_rgb(parse_metric_depth(image) * 255)
    elif isinstance(image, str):
        return parse_rgb(Image.open(image))
    else:
        raise NotImplementedError(f'type of image {type(image)} is not supported')

def convert_xyz_to_kjl(xyz: np.ndarray) -> np.ndarray:
    return np.matmul(np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=xyz.dtype), xyz[..., None])[..., 0]

def latlong_depth_to_pcd_plus_np(depth:np.ndarray, xyz_kjl=False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    depth: [H, W, 1]
    xyz: [H, W, 3]
    polars: [2, 3]
    '''
    H, W, _ = depth.shape
    dtype = depth.dtype
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=dtype), 
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=dtype),
        indexing='ij',
    )
    psi = gx * np.pi  # phi in range(-pi, pi)
    theta = gy * (np.pi / 2) + np.pi / 2  # theta in range(0, pi)
    rd = np.stack([
        np.sin(theta) * np.sin(psi), 
        np.cos(theta), 
        -np.sin(theta) * np.cos(psi)
    ], axis=-1)
    xyz = depth * rd
    polars = depth[[0, -1], :, :].mean(axis=1, keepdims=False) * np.array([
        [0.0, 1.0, 0.0], 
        [0.0, -1.0, 0.0], 
    ], dtype=dtype)
    if xyz_kjl:
        xyz = convert_xyz_to_kjl(xyz)
        polars = convert_xyz_to_kjl(polars)
    return xyz, polars

def latlong_to_faces_np(H:int, W:int) -> np.ndarray:
    '''
    faces: [H, W, 3]
    '''
    index = np.arange(H, dtype=np.int64)[:, None] * W + np.arange(W, dtype=np.int64)[None, :]
    offset = np.array([[0, W+1, 1], [0, W, W+1]], dtype=np.int64)
    faces = index[:, :, None, None] + offset
    faces = faces // W % H * W + faces % W % W
    return faces

def latlong_to_faces_plus_np(H:int, W:int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    faces: [H, W, 3]
    polars: [2, W, 3]
    '''
    ix = np.arange(H - 1, dtype=np.int64)
    iy = np.arange(W, dtype=np.int64)
    ox = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    oy = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
    faces = ((ix[:, None, None, None] + ox) % H) * W + ((iy[None, :, None, None] + oy) % W)
    polars = np.stack([
        np.stack([
            np.full_like(iy, fill_value=H * W), 
            iy, 
            np.roll(iy, shift=1, axis=0),
        ], axis=-1),
        np.stack([
            (H - 1) * W + iy, 
            np.full_like(iy, fill_value=H * W + 1), 
            np.roll((H - 1) * W + iy, shift=1, axis=0),
        ], axis=-1),
    ], axis=0)
    return faces, polars

def wall_to_faces(N:int) -> List[List[int]]:
    iy = np.arange(N, dtype=np.int64)
    ox = np.array([[0, 1, 1, 0]], dtype=np.int64)
    oy = np.array([[0, 0, 1, 1]], dtype=np.int64)
    faces = (ox % 2) * N + ((iy[:, None] + oy) % N)
    faces = faces.tolist() + [list(range(N)), list(range(N, 2 * N))[::-1]]
    return faces

def visualize_colored_point_cloud(
    rgb_image: Union[torch.Tensor, np.ndarray, Image.Image, str],
    depth_image: Union[torch.Tensor, np.ndarray, Image.Image, str],
    with_faces: bool = False,
    xyz_kjl: bool = False,
) -> trimesh.Trimesh:
    '''
    rgb_image: 
        * Tensor: [3, H, W], float32, range(0.0, 1.0)
        * ndarray: [H, W, 3], uint8
        * Image
        * str: path to image
    depth_image
        * Tensor: [1, H, W], float32, metric depth
        * ndarray: [H, W, 1], uint8/int16/int32/float32, metric depth
        * Image
        * str: path to image
    pcd: Trimesh
    '''
    depth = parse_metric_depth(depth_image)
    rgb = parse_rgb(rgb_image)
    H, W, _ = depth.shape
    xyz, polars = latlong_depth_to_pcd_plus_np(depth.mean(axis=-1, keepdims=True), xyz_kjl=xyz_kjl)
    xyz = np.concatenate([
        xyz.reshape(-1, 3),
        polars.reshape(-1, 3),
    ], axis=0)
    if with_faces:
        faces, polars = latlong_to_faces_plus_np(H, W)
        faces = np.concatenate([
            faces.reshape(-1, 3),
            polars.reshape(-1, 3),
        ], axis=0)
        faces = np.flip(faces, axis=-1)
    else:
        faces = None
    rgb = rgb[..., :3]
    polars = (rgb[[0, -1], :, :] / 255.0).mean(axis=1, keepdims=False)
    polars = (np.clip(polars, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.concatenate([
        rgb.reshape(-1, 3),
        polars.reshape(-1, 3),
    ], axis=0)
    pcd = trimesh.Trimesh(vertices=xyz, faces=faces, vertex_colors=rgb, process=False)
    return pcd

class FakeMesh:
    def __init__(self, vertices:List[List[float]], faces:List[List[int]]):
        self.vertices = vertices
        self.faces = faces

    def export(self, path):
        assert os.path.splitext(path)[1] == '.obj'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            for v in self.vertices:
                fp.write('v ' + ' '.join([f'{_v:.8f}' for _v in v]) + '\n')
            for f in self.faces:
                fp.write('f ' + ' '.join([f'{_f+1:d}' for _f in f]) + '\n')

def visualize_poly_np(
    poly: np.ndarray,
    with_faces: bool = True,
    xyz_kjl: bool = False,
) -> FakeMesh:
    '''
    poly: [2, N, 3]
    pcd: Trimesh
    '''
    _, N, _ = poly.shape
    xyz = poly
    if xyz_kjl:
        xyz = convert_xyz_to_kjl(xyz)
    xyz = xyz.reshape(-1, 3).tolist()
    if with_faces:
        faces = wall_to_faces(N)
    else:
        faces = None
    pcd = FakeMesh(vertices=xyz, faces=faces)
    return pcd


if __name__ == "__main__":
    # generated depth
    rgb_image = '/home/chenxiao/下载/0302/individualImage-p-rgbd-1500-0-rgb.png'
    depth_image = '/home/chenxiao/下载/0302/individualImage-p-rgbd-1500-0-depth.png'
    pcd_path = 'test_result/test_panorama_pcd/individualImage-p-rgbd-1500-0-pcd.ply'

    # ground-truth depth
    rgb_image = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_rgb/83.png'
    depth_image = '/media/chenxiao/data/roomverse/3FO4K5G1L00B/perspective/room_610/pano_depth/83.png'
    pcd_path = 'test_result/test_panorama_pcd/3FO4K5G1L00B--room_610--83-pcd.ply'

    os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
    pcd = visualize_colored_point_cloud(rgb_image, depth_image)
    pcd.export(pcd_path)

