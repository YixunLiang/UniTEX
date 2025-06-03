from glob import glob
import json
import os
from time import perf_counter
import pandas as pd
import torch
from tqdm import tqdm
import trimesh
from timeout_decorator import timeout

from ..utils import to_tensor_f, to_tensor_i
from .edge_sampling import select_and_sample_on_edges
from .surface_sampling import sample_surface
from ...mesh.structure_v2 import trimesh_to_pbr_mesh
from .uv_sampling import sample_pbr_mesh


@timeout(60)
def geomerty_sampling(
    input_mesh_path,
    output_mesh_sharp_path,
    output_mesh_coarse_path,
    scale=1.0,
    N=10_000_000,
    angle_threhold_deg=15.0,
    method='equal_steps',
    merge_close_vertices=False,
):
    '''
    input_mesh_path: mesh path
    output_mesh_path: sampled point cloud path
    method: probability, equal_steps
    '''
    time_log = dict()

    ### load whole mesh
    t0 = perf_counter()
    mesh:trimesh.Trimesh = trimesh.load(input_mesh_path, process=False, force='mesh')
    if merge_close_vertices:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    # trimesh.util.concatenate(loaded.dump())
    vertices = to_tensor_f(mesh.vertices)
    if scale is not None and scale != 1.0:
        vertices = vertices * scale
    faces = to_tensor_i(mesh.faces)
    areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
    normals = torch.nn.functional.normalize(areas, dim=-1)
    time_log['load whole mesh'] = perf_counter() - t0

    ### sample on surface
    t0 = perf_counter()
    surface_points, surface_points_faces, surface_points_weights = sample_surface(vertices, faces, areas=areas, N=N, seed=666)
    time_log['sample on surface'] = perf_counter() - t0

    ### select and sample on sharp edges
    t0 = perf_counter()
    edge_points, edge_points_edges, edge_points_weights = select_and_sample_on_edges(vertices, faces, normals=normals, method=method, angle_threhold_deg=angle_threhold_deg, N=N, seed=666)
    time_log['select and sample on sharp edges'] = perf_counter() - t0

    ### export point cloud
    t0 = perf_counter()
    vertices_sharp_np = edge_points.detach().cpu().numpy() if edge_points is not None else None
    vertices_coarse_np = surface_points.detach().cpu().numpy()
    mesh_sharp = trimesh.Trimesh(vertices=vertices_sharp_np, process=False)
    mesh_coarse = trimesh.Trimesh(vertices=vertices_coarse_np, process=False)
    os.makedirs(os.path.dirname(output_mesh_sharp_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mesh_coarse_path), exist_ok=True)
    mesh_sharp.export(output_mesh_sharp_path)
    mesh_coarse.export(output_mesh_coarse_path)
    time_log['export point cloud'] = perf_counter() - t0

    print('time log:\n\t' + '\n\t'.join(f'{k:36s}:{v:.6f}' for k, v in time_log.items()))
    return time_log


@timeout(60)
def texture_sampling(
    input_mesh_path,
    output_mesh_path,
    N=10_000_000,
):
    '''
    input_mesh_path: mesh path
    output_mesh_path: sampled point cloud path
    '''
    time_log = dict()

    ### load scene
    t0 = perf_counter()
    mesh:trimesh.Scene = trimesh.load(input_mesh_path, force='scene')
    time_log['load scene'] = perf_counter() - t0

    ### move mesh to gpu
    t0 = perf_counter()
    pbr_mesh = trimesh_to_pbr_mesh(mesh)
    time_log['move mesh to gpu'] = perf_counter() - t0

    ### sample on mesh
    t0 = perf_counter()
    samples, face_index, face_attr = sample_pbr_mesh(pbr_mesh, N=N, seed=666)
    time_log['sample on mesh'] = perf_counter() - t0

    ### move samples to cpu
    t0 = perf_counter()
    samples, face_index, face_attr = samples.detach().cpu().numpy(), face_index.cpu().numpy(), {k: v.detach().cpu().numpy() for k, v in face_attr.items()}
    time_log['move samples to cpu'] = perf_counter() - t0

    ### export point cloud
    t0 = perf_counter()
    pcd = trimesh.Trimesh(
        vertices=samples, 
        vertex_normals=None, 
        vertex_colors=face_attr['albedo'],
        process=False,
    )
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    pcd.export(output_mesh_path)
    time_log['export point cloud'] = perf_counter() - t0

    print('time log:\n\t' + '\n\t'.join(f'{k:36s}:{v:.6f}' for k, v in time_log.items()))
    return time_log


################################################################


def small_test_gs():
    mesh_path = '/home/chenxiao/下载/0214/000-015/8a286c83be89423aa69b3ff606a67d40.obj'
    sample_sharp_path = '/home/chenxiao/下载/0214/000-015/8a286c83be89423aa69b3ff606a67d40_sharp.ply'
    sample_coarse_path = '/home/chenxiao/下载/0214/000-015/8a286c83be89423aa69b3ff606a67d40_coarse.ply'
    geomerty_sampling(mesh_path, sample_sharp_path, sample_coarse_path)

def large_test_gs():
    src = '/mnt/nas-algo/chenxiao/dataset/vae_example_data/objaverse_data/*/*.glb'
    dst = '/mnt/nas-algo/chenxiao/dataset/edge_sampling_results'
    path_json = '/mnt/nas-algo/chenxiao/dataset/edge_sampling_results.json'
    path_csv = '/mnt/nas-algo/chenxiao/dataset/edge_sampling_results.csv'

    for input_path in tqdm(glob(src)):
        uid = os.path.join(os.path.basename(os.path.dirname(input_path)), os.path.splitext(os.path.basename(input_path))[0])
        output_sharp_path = os.path.join(dst, uid + '_sharp.ply')
        output_coarse_path = os.path.join(dst, uid + '_coarse.ply')
        log_path = os.path.join(dst, uid + '.json')
        if os.path.isfile(output_sharp_path) and os.path.isfile(output_coarse_path):
            continue
        try:
            time_log = geomerty_sampling(input_path, output_sharp_path, output_coarse_path)
        except Exception as e:
            time_log = 'failed'
            print(e)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({uid: time_log}, f, indent=4)

    log_dict = dict()
    KEYS = [
        'load whole mesh',
        'sample on surface',
        'select and sample on sharp edges',
        'export point cloud',
    ]
    for log_path in glob(os.path.join(dst, '*/*.json')):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_dict.update(json.load(f))
    for uid, v in log_dict.items():
        if isinstance(v, str):
            log_dict[uid] = {k: v for k in KEYS}
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=4)
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.DataFrame(log_dict).T
    df.to_csv(path_csv)
    df[df['load whole mesh'] != 'failed'].to_csv(os.path.splitext(path_csv)[0] + '_success.csv')
    df[df['load whole mesh'] == 'failed'].to_csv(os.path.splitext(path_csv)[0] + '_fail.csv')

def small_test_ts():
    test_examples = [
        'cute_wolf/textured_mesh.glb',
        'car/867ceb1e7dc245539e0301ef0ece74f4.glb',
        'car2/88a596cf876c4175a59a3323510d49f0.glb',
        'car3/368873bdc2634959b020d58a88797158.glb',
        'watch/9057e49289d742eb9663edea9aadf3e8.glb',
    ]
    for test_example in test_examples:
        input_mesh_path = os.path.join('gradio_examples_mesh', test_example)
        output_mesh_path = os.path.join('gradio_examples_mesh_results', os.path.splitext(test_example)[0] + '_sampling.ply')

        t = perf_counter()
        texture_sampling(input_mesh_path, output_mesh_path)
        print('>> texture_sampling_v2', perf_counter() - t)

def large_test_ts():
    path_src = '/home/chenxiao/下载/0206/examples/*.glb'
    path_dst = '/home/chenxiao/下载/0206/samples/'
    path_log = '/home/chenxiao/下载/0206/logs/'
    path_json = '/home/chenxiao/下载/0206/logs.json'
    path_csv = '/home/chenxiao/下载/0206/logs.csv'

    for input_mesh_path in glob(path_src):
        uid = os.path.splitext(os.path.basename(input_mesh_path))[0]
        output_mesh_path = os.path.join(path_dst, uid + '.ply')
        log_path = os.path.join(path_log, uid + '.json')

        try:
            time_log = texture_sampling(input_mesh_path, output_mesh_path)
        except Exception as e:
            time_log = 'failed'
            print(e)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({uid: time_log}, f, indent=4)

    log_dict = dict()
    KEYS = [
        'load scene',
        'move mesh to gpu',
        'sample on mesh',
        'move samples to cpu',
        'export point cloud',
    ]
    for log_path in glob(os.path.join(path_log, '*.json')):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_dict.update(json.load(f))
    for uid, v in log_dict.items():
        if isinstance(v, str):
            log_dict[uid] = {k: v for k in KEYS}
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=4)
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.DataFrame(log_dict).T
    df.to_csv(path_csv)
    df[df['load whole mesh'] != 'failed'].to_csv(os.path.splitext(path_csv)[0] + '_success.csv')
    df[df['load whole mesh'] == 'failed'].to_csv(os.path.splitext(path_csv)[0] + '_fail.csv')


if __name__ == '__main__':
    small_test_gs()
    # small_test_ts()
    # large_test_gs()
    # large_test_ts()

