
from glob import glob
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw

from texturetools.io.mesh_header_loader import parse_mesh_info
from texturetools.video.export_nvdiffrast_video import VideoExporter


NAME = 'rt_udf'
if NAME in ['wn_sdf', 'rtv_sdf', 'bvh_sdf']:
    KEYS = [
        "load whole mesh",
        "sample surface points",
        "build cubes",
        "compute sdf",
        "run flexicubes plus",
        "export reconstructed mesh",
    ]
elif NAME in ['rt_plus_sdf', 'rt_udf', 'rt_udf_v1', 'rt_udf_v2']:
    KEYS = [
        "load whole mesh",
        "build trees",
        "sample surface points",
        "build sparse cubes",
        "compute sdf",
        "run flexicubes plus",
        "largest connected components",
        "export reconstructed mesh",
    ]
else:
    raise NotImplementedError(f'NAME {NAME} is not supported')


def render_videos():
    src_raw = '/mnt/nas-algo/chenxiao/dataset/vae_example_data/objaverse_data/'
    src_rec = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}/*/*.glb'
    dst_vis = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}_vis'
    video_exporter = VideoExporter()

    for p in tqdm(glob(src_rec)):
        uid = os.path.join(os.path.basename(os.path.dirname(p)), os.path.splitext(os.path.basename(p))[0])

        p_dst = os.path.join(dst_vis, uid+'_grid.png')
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)

        try:
            res_raw = video_exporter.export_condition(os.path.join(src_raw, uid+'.glb'), geometry_scale=1.0, H=1024, W=1024, n_rows=1, n_cols=4, background='white')
            res_rec = video_exporter.export_condition(p, geometry_scale=1.0, H=1024, W=1024, n_rows=1, n_cols=4, background='white')
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            continue

        im = Image.fromarray(np.concatenate([np.array(res_raw['normal']), np.array(res_rec['normal'])], axis=0))
        ImageDraw.Draw(im).text([0, 0], uid, fill='red')
        im.save(p_dst)


def merge_jsons():
    src_raw = '/mnt/nas-algo/chenxiao/dataset/vae_example_data/objaverse_data/'
    src_json = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}/*/*.json'
    dst_json = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}.json'
    dst_csv = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}.csv'
    dst_csv_success = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}_success.csv'
    dst_csv_fail = f'/mnt/nas-algo/chenxiao/dataset/reconstruct_results/{NAME}_fail.csv'

    data = dict()
    for json_path in tqdm(glob(src_json)):
        with open(json_path, 'r', encoding='utf-8') as f:
            data.update(json.load(f))

    for uid, v in data.items():
        if isinstance(v, str):
            data[uid] = {k: v for k in KEYS}
        info = parse_mesh_info(os.path.join(src_raw, uid+'.glb'))
        data[uid].update({
            'vertices': info['V'], 
            'faces': info['F'],
        })

    # export json
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    # export csv
    df = pd.DataFrame(data).T
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    df.to_csv(dst_csv)
    os.makedirs(os.path.dirname(dst_csv_success), exist_ok=True)
    df[df['load whole mesh'] != 'failed'].to_csv(dst_csv_success)
    os.makedirs(os.path.dirname(dst_csv_fail), exist_ok=True)
    df[df['load whole mesh'] == 'failed'].to_csv(dst_csv_fail)


if __name__ == '__main__':
    merge_jsons()
    # render_videos()

