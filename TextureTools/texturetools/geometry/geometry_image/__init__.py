import os
from time import perf_counter
from typing import Optional
import numpy as np
import torch
import trimesh
from timeout_decorator import timeout
from PIL import Image

from ..uv.uv_atlas import preprocess_blank_mesh_trimesh as preprocess_blank_mesh
from ..utils import to_tensor_f, to_tensor_i
from .convert_mesh_and_gi import mesh_to_gi, gi_to_mesh


@timeout(60)
def mesh_to_geometry_image(
    input_mesh_path, 
    output_image_path, 
    output_mesh_path, 
    scale:Optional[float]=0.95, 
    mode='sphere',
):
    '''
    input_mesh_path: normalized mesh path, range(-1.0, 1.0)
    output_image_path: geometry image path
    output_mesh_path: reconstructed mesh path
    scale: apply scale on input mesh
    mode: sphere, shell, cylinder, uv
    '''
    time_log = dict()

    ### load whole mesh
    t0 = perf_counter()
    mesh_trimesh:trimesh.Trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')
    # mesh_trimesh = trimesh.creation.uv_sphere(1.0)  # NOTE: debug with sphere
    if mode == 'uv':
        vertices, faces, uvs_2d, faces_2d = preprocess_blank_mesh(mesh_trimesh)
    else:
        vertices, faces = to_tensor_f(mesh_trimesh.vertices), to_tensor_i(mesh_trimesh.faces)
        uvs_2d, faces_2d = None, None
    if scale is not None and scale != 1.0:
        vertices = vertices * scale
    areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
    normals = torch.nn.functional.normalize(areas, dim=-1)
    time_log['load whole mesh'] = perf_counter() - t0

    ### mesh to gi
    t0 = perf_counter()
    uv_mask, uv_ccm, uv_normal = mesh_to_gi(vertices, faces, normals=normals, uvs_2d=uvs_2d, faces_2d=faces_2d, mode=mode)
    time_log['mesh to gi'] = perf_counter() - t0

    ### gi to mesh
    t0 = perf_counter()
    vertices, uvs, faces, normals = gi_to_mesh(uv_mask, uv_ccm, uv_normal)
    time_log['gi to mesh'] = perf_counter() - t0

    ### export image
    t0 = perf_counter()
    uv_mask_im = Image.fromarray(np.array(uv_mask[..., 0].mul(255.0).detach().cpu().numpy().astype(np.uint8)))
    uv_ccm_im = Image.fromarray(np.array(uv_ccm.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)))
    uv_normal_im = Image.fromarray(np.array(uv_normal.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)))
    uv_ccm_im.putalpha(uv_mask_im)
    uv_normal_im.putalpha(uv_mask_im)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    uv_ccm_im.save(os.path.splitext(output_image_path)[0] + '_ccm.png')
    uv_normal_im.save(os.path.splitext(output_image_path)[0] + '_normal.png')
    time_log['export image'] = perf_counter() - t0

    ### export mesh
    t0 = perf_counter()
    if scale is not None and scale != 1.0:
        vertices = vertices / scale
    # vertices = torch.cat([uvs, torch.zeros_like(uvs[..., [0]])], dim=-1)  # NOTE: debug with 2d mesh
    vertices = vertices.detach().cpu().numpy()
    uvs = uvs.detach().cpu().numpy()
    faces = faces.cpu().numpy()
    if normals is not None:
        normals = normals.detach().cpu().numpy()
    mesh = trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_normals=normals, 
        visual=trimesh.visual.TextureVisuals(
            uv=uvs,
            material=trimesh.visual.material.PBRMaterial(),
        ), 
        process=False,
    )
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    mesh.export(output_mesh_path)
    time_log['export mesh'] = perf_counter() - t0

    print('time log:\n\t' + '\n\t'.join(f'{k:36s}:{v:.6f}' for k, v in time_log.items()))
    return time_log


def small_test():
    test_examples = [
        'cute_wolf/textured_mesh.glb',
        'car/867ceb1e7dc245539e0301ef0ece74f4.glb',
        'car2/88a596cf876c4175a59a3323510d49f0.glb',
        'car3/368873bdc2634959b020d58a88797158.glb',
        'watch/9057e49289d742eb9663edea9aadf3e8.glb',
    ]
    for test_example in test_examples:
        input_mesh_path = os.path.join('gradio_examples_mesh', test_example)
        output_image_path = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_gi_uv.png')
        output_mesh_path = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_gi_uv_rec.glb')

        t = perf_counter()
        convert_mesh_and_gi(input_mesh_path, output_image_path, output_mesh_path)
        print('>> uv_baking', perf_counter() - t)


if __name__ == '__main__':
    small_test()

