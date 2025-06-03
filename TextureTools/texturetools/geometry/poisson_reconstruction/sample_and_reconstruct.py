from glob import glob
import os
import numpy as np
from tqdm import tqdm
import pymeshlab as ml
import trimesh
from ...video.export_nvdiffrast_video import VideoExporter

def get_trimesh(ms: ml.MeshSet) -> trimesh.Trimesh:
    m = ms.current_mesh()
    v = np.asarray(m.vertex_matrix(), dtype=np.float32)
    vn = np.asarray(m.vertex_normal_matrix(), dtype=np.float32)
    f = np.asarray(m.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=v, vertex_normals=vn, faces=f, process=False)

def save_ply(ms: ml.MeshSet, output_file_path: str):
    output_file_path = os.path.splitext(output_file_path)[0] + '.ply'
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    ms.save_current_mesh(
        output_file_path,
        binary=True,
        save_vertex_quality=False,
        save_vertex_flag=False,
        save_vertex_color=False,
        save_vertex_coord=True,
        save_vertex_normal=True,
        save_vertex_radius=False,
        save_face_quality=False,
        save_face_flag=False,
        save_face_color=False, 
        save_wedge_color=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
    )

def save_obj(ms: ml.MeshSet, output_file_path: str):
    output_file_path = os.path.splitext(output_file_path)[0] + '.obj'
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    ms.save_current_mesh(
        output_file_path,
        save_vertex_color=False,
        save_vertex_coord=True,
        save_vertex_normal=True,
        save_face_color=False, 
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False,
    )

def poisson_sample(input_file_path, output_file_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file_path)
    ms.generate_sampling_poisson_disk(samplenum=1_000_000)
    save_ply(ms, output_file_path)

def poisson_reconstruct(input_file_path, output_file_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file_path)
    ms.generate_surface_reconstruction_screened_poisson(depth=12, preclean=True, threads=20)
    save_obj(ms, output_file_path)

def sample_and_reconstruct(input_file_path, output_file_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file_path)
    ms.generate_sampling_poisson_disk(samplenum=1_000_000)
    ms.generate_surface_reconstruction_screened_poisson(depth=12, preclean=True, threads=20)
    save_obj(ms, output_file_path)

def render_geometry(video_exporter:VideoExporter, mesh_path):
    return video_exporter.export_condition(
        mesh_path, 
        geometry_scale=1.0, 
        n_views=64,
        n_rows=8,
        n_cols=8,
        H=512,
        W=512,
        fov_deg=49.1,
        scale=0.85,
        perspective=False,
        orbit=False,
        background='white',
        return_info=False,
        return_image=False,
        return_mesh=False,
        return_camera=False,
    )

def render_and_sample(video_exporter, input_file_path, output_file_path):
    geometry = render_geometry(video_exporter, input_file_path)
    mask = (geometry['alpha'] > 0).repeat(3, axis=-1)
    v = geometry['ccm'][mask].reshape(-1, 3) * 2.0 - 1.0
    vn = geometry['normal'][mask].reshape(-1, 3) * 2.0 - 1.0
    ms = ml.MeshSet()
    m = ml.Mesh(vertex_matrix=v.astype(np.float64), v_normals_matrix=vn.astype(np.float64))
    ms.add_mesh(m)
    save_ply(ms, output_file_path)

def render_and_reconstruct(video_exporter, input_file_path, output_file_path):
    geometry = render_geometry(video_exporter, input_file_path)
    mask = (geometry['alpha'] > 0).repeat(3, axis=-1)
    v = geometry['ccm'][mask].reshape(-1, 3) * 2.0 - 1.0
    vn = geometry['normal'][mask].reshape(-1, 3) * 2.0 - 1.0
    ms = ml.MeshSet()
    m = ml.Mesh(vertex_matrix=v.astype(np.float64), v_normals_matrix=vn.astype(np.float64))
    ms.add_mesh(m)
    ms.generate_surface_reconstruction_screened_poisson(depth=12, preclean=True, threads=20)
    save_obj(ms, output_file_path)

def test():
    src = "/mnt/nas-algo/chenxiao/dataset/GeomDist/shapes/*.obj"
    dst = "/mnt/nas-algo/chenxiao/dataset/GeomDist/reconstructs"
    video_exporter = VideoExporter()
    for input_file_path in tqdm(glob(src)):
        output_file_path = os.path.join(dst, os.path.basename(input_file_path))
        # sample_and_reconstruct(input_file_path, output_file_path)
        render_and_reconstruct(video_exporter, input_file_path, output_file_path)
