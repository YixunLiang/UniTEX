'''
Convert mesh to sdf field with ray tracing and 
extract iso field to mesh with sparse grid marching cubes
'''

from glob import glob
import json
import os
from time import perf_counter
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from timeout_decorator import timeout

# https://github.com/nv-tlabs/FlexiCubes.git
# from flexicubes import FlexiCubes

# https://github.com/libigl/libigl.git
# import igl

# https://github.com/ashawkey/cubvh.git
# https://github.com/opencv/opencv/issues/14868
from cubvh import cuBVH

# https://github.com/lcp29/trimesh-ray-optix
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# export OptiX_INSTALL_DIR=${HOME}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
from triro.ray.ray_optix import RayMeshIntersector

# https://pytorch-geometric.readthedocs.io/en/1.3.1/index.html
# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
import torch_cluster

# https://github.com/YuliangXiu/bvh-distance-queries.git
# https://github.com/NVIDIA/cuda-samples.git
# export CUDA_SAMPLES_INC=${HOME}/cuda-samples-12.4.1/Common
# https://github.com/LCH1238/bevdet-tensorrt-cpp/issues/18
# remove nvcc compile args '-DNUM_THREADS=256' in setup.py
# from bvh_distance_queries import BVH

# https://kaolin.readthedocs.io/en/stable/index.html
# pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
# import kaolin


from ..utils import to_tensor_f, to_tensor_i
from ..sampling.surface_sampling import sample_surface
from ..triangle_topology.clean import remove_unreferenced_vertices
from ...io.dump_glb import dump_glb
from ..extract_isosurface.sparsecubes.build_cubes import build_cubes
from ..extract_isosurface.sparsecubes import extrac_isosurface
from ..connected_components.large_connected_components import large_connnected_components
from ..connected_components.visiable_connected_components import visiable_connected_components
from ...raytracing.check_visibility import self_rt, cross_rt


def assign_candidates(points:torch.Tensor, candidates:torch.Tensor, depth=8) -> torch.Tensor:
    '''
    points: [N, 3]
    candidates: [M, 3]
    assigned_candidates: [N, 3]
    '''
    b = (2 ** depth)
    bx, by, bz = b ** 2, b, 1
    indices_candidates = torch.floor((torch.clamp(candidates, -1.0, 1.0) * 0.5 + 0.5) * b).to(dtype=torch.int64)
    indices_candidates_unique, indices_candidates_inverse, indices_candidates_counts = torch.unique(indices_candidates, dim=0, return_inverse=True, return_counts=True)
    indices_candidates_index = indices_candidates_inverse.argsort(stable=True)[torch.cat([indices_candidates_counts.new_zeros(1), indices_candidates_counts.cumsum(dim=0)])[:-1]]
    candidates = candidates[indices_candidates_index, :]
    assigned_index = torch_cluster.knn(candidates, points, k=1)
    batch_index, index = assigned_index.unbind(0)
    assigned_candidates = torch.gather(candidates, 0, index.unsqueeze(-1).repeat(1, 3))
    return assigned_candidates


@timeout(60)
def mesh_to_sdf_octree(
    input_mesh_path, 
    output_mesh_path, 
    scale:Optional[float]=0.95, 
    N=10_000_000, 
    depth=8, 
    depth_refine:Optional[int]=10, 
    expand:Optional[int]=1, 
    expand_refine:Optional[int]=2, 
    sample_offset:Optional[float]=1.0, 
    sdf_offset:Optional[float]=1.0, 
    correct_sdf=False, 
    enable_double_samples=False, 
    enable_self_rt=False, 
    enable_cross_rt=True, 
    enable_lcc=False, 
    enable_vcc=True, 
    vcc_method='optix', 
    N_vcc=1_000, 
):
    '''
    input_mesh_path: normalized mesh path, range(-1.0, 1.0)
    output_mesh_path: reconstructed mesh path
    scale: apply scale on input mesh
    N: num of samples on mesh
    depth, depth_refine: depth of octree
    expand, expand_refine: expand of octree
    sample_offset: random perturbation for sample points, scale length is size of leaf cubes
    sdf_offset: dilation for sdf field, scale length is size of leaf cubes
    enable_self_rt: enable self ray tracing
    enable_cross_rt: if enable self ray tracing, enable cross ray tracing
    enable_lcc: enable largest connnected components(lcc), prior: lcc > vcc
    enable_vcc: enable visible connnected components, prior: lcc > vcc
    vcc_method: optix(recommended), slangrt(recommended), nvdiffrt
    N_vcc: num rays (default=1_000) or num cameras (default=6) of visible connnected components
    '''
    time_log = dict()
    sample_offset = sample_offset * (2.0 / (2 ** (depth_refine or depth))) if sample_offset is not None else None
    sdf_offset = sdf_offset * (2.0 / (2 ** (depth_refine or depth))) if sdf_offset is not None else None

    ### load whole mesh
    t0 = perf_counter()
    mesh_trimesh:trimesh.Trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')
    # mesh_trimesh = trimesh.creation.uv_sphere(1.0)  # NOTE: debug with sphere
    vertices = to_tensor_f(mesh_trimesh.vertices)
    if scale is not None and scale != 1.0:
        vertices = vertices * scale
    faces = to_tensor_i(mesh_trimesh.faces)
    time_log['load whole mesh'] = perf_counter() - t0

    ### build trees
    t0 = perf_counter()
    areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
    normals = torch.nn.functional.normalize(areas, dim=-1)
    # TODO: cubvh build bvh tree on cpu with one thread, too slow
    bvh_cubvh = cuBVH(vertices=vertices, triangles=faces)
    if enable_self_rt or enable_cross_rt:
        bvh_optix = RayMeshIntersector(vertices=vertices, faces=faces)
    else:
        bvh_optix = None
    time_log['build trees'] = perf_counter() - t0

    ### sample surface points
    t0 = perf_counter()
    surface_points, surface_points_faces, _ = sample_surface(vertices, faces, areas=areas, N=N, seed=666)
    if enable_double_samples and sdf_offset is not None and sdf_offset != 0.0:
        surface_points = surface_points.unsqueeze(1) + sdf_offset * to_tensor_f([1.0, -1.0]).unsqueeze(-1) * normals[surface_points_faces, :].unsqueeze(1)
        surface_points = surface_points.reshape(-1, 3)
        surface_points_faces = surface_points_faces.repeat_interleave(2, dim=0)
    time_log['sample surface points'] = perf_counter() - t0

    ### build sparse cubes
    t0 = perf_counter()
    cube_vertices, cube_faces = build_cubes(
        surface_points, 
        depth=depth, 
        depth_refine=depth_refine, 
        expand=expand, 
        expand_refine=expand_refine,
    )
    del surface_points, surface_points_faces
    time_log['build sparse cubes'] = perf_counter() - t0

    ### compute sdf
    t0 = perf_counter()
    cube_vertices_sdf, cube_vertices_faces, cube_vertices_uv = bvh_cubvh.unsigned_distance(cube_vertices, return_uvw=True)
    if correct_sdf:
        cube_vertices_hits = torch.sum(vertices[faces[cube_vertices_faces, :], :] * cube_vertices_uv.unsqueeze(-1), dim=1)
        cube_vertices_sdf = torch.abs(torch.sum((cube_vertices - cube_vertices_hits) * normals[cube_vertices_faces, :], dim=-1))
    if enable_self_rt:
        cube_vertices_mask = self_rt(bvh_optix, cube_vertices)
        if enable_cross_rt:
            cube_vertices_index = torch.arange(cube_vertices.shape[0], dtype=torch.int64, device='cuda')
            cube_vertices_inner = cube_vertices[cube_vertices_mask, :]
            cube_vertices_inner_hits = torch.sum(vertices[faces[cube_vertices_faces[cube_vertices_mask], :], :] * cube_vertices_uv[cube_vertices_mask, :].unsqueeze(-1), dim=1)
            cube_vertices_inner_normals = normals[cube_vertices_faces[cube_vertices_mask], :]
            cube_vertices_near = cube_vertices_inner_hits.unsqueeze(1) + sample_offset * to_tensor_f([1.0, -1.0]).unsqueeze(-1) * cube_vertices_inner_normals.unsqueeze(1)
            cube_vertices_near_mask = self_rt(bvh_optix, cube_vertices_near)
            cube_vertices_outer = assign_candidates(cube_vertices_inner, torch.masked_select(cube_vertices_near, cube_vertices_near_mask.logical_not().unsqueeze(-1)).reshape(-1, 3))
            cube_vertices_mask_mask = cross_rt(bvh_optix, cube_vertices_inner.unsqueeze(1), cube_vertices_outer.unsqueeze(1), exhaustive_mode=False)
            cube_vertices_mask.scatter_(0, cube_vertices_index[cube_vertices_mask][~cube_vertices_mask_mask], False)
        cube_vertices_sdf.scatter_(0, torch.where(cube_vertices_mask)[0], -1.0, reduce='multiply')
        del cube_vertices_mask
    time_log['compute sdf'] = perf_counter() - t0

    ### run flexicubes plus
    t0 = perf_counter()
    vertices, faces = extrac_isosurface(
        cube_vertices, cube_vertices_sdf, cube_faces, 
        sdf_offset=sdf_offset,
    )
    del cube_vertices, cube_vertices_sdf, cube_faces
    time_log['run flexicubes plus'] = perf_counter() - t0

    ## largest connected components
    t0 = perf_counter()
    if enable_vcc:
        faces = visiable_connected_components(
            vertices, faces, 
            n_rays_or_cameras=N_vcc, 
            sample_offset=sample_offset,
            method=vcc_method,
        )
    elif enable_lcc:
        faces = large_connnected_components(faces)
    time_log['largest connected components'] = perf_counter() - t0

    ## export reconstructed mesh
    t0 = perf_counter()
    if enable_lcc or enable_vcc:
        vertices, faces = remove_unreferenced_vertices(vertices, faces)
    if scale is not None and scale != 1.0:
        vertices = vertices / scale
    dump_glb(vertices, faces, output_path=output_mesh_path)
    time_log['export reconstructed mesh'] = perf_counter() - t0

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
        output_mesh_path_dense = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_dense.glb')
        output_mesh_path_octree = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_octree.glb')

        # t = perf_counter()
        # mesh_to_sdf_dense(input_mesh_path, output_mesh_path_dense)
        # print('>> mesh_to_sdf_dense', perf_counter() - t)

        t = perf_counter()
        mesh_to_sdf_octree(input_mesh_path, output_mesh_path_octree)
        print('>> mesh_to_sdf_octree', perf_counter() - t)


def large_test():
    src = '/mnt/nas-algo/chenxiao/dataset/vae_example_data/objaverse_data/*/*.glb'
    dst = '/mnt/nas-algo/chenxiao/dataset/reconstruct_results'
    config_list = {
        'rt_plus_sdf': {
            'enable_self_rt': True, 
            'enable_cross_rt': True,
            'enable_lcc': True,
            'enable_vcc': False,
        },
        'rt_udf': {
            'enable_self_rt': False, 
            'enable_cross_rt': True,
            'enable_lcc': False,
            'enable_vcc': True,
        }
    }
    for key, config in config_list.items():
        for input_path in tqdm(glob(src), desc=key):
            uid = os.path.join(os.path.basename(os.path.dirname(input_path)), os.path.splitext(os.path.basename(input_path))[0])
            output_path = os.path.join(dst, key, uid + '.glb')
            log_path = os.path.join(dst, key, uid + '.json')
            if os.path.isfile(output_path):
                continue
            try:
                time_log = mesh_to_sdf_octree(input_path, output_path, **config)
            except Exception as e:
                time_log = 'failed'
                print(e)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({uid: time_log}, f, indent=4)


if __name__ == '__main__':
    small_test()
    # large_test()

