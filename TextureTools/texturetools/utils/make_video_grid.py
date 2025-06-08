from glob import glob
import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm

def make_one_video():
    #### parameters begin ####
    video_path = "//home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/mv_4_views_xl--albedo_ft_arbit_video/*.mp4"
    dump_path = "/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/mv_4_views_xl--albedo_ft_arbit_video.mp4"
    n_frames, H, W = 180, 1024, 1024  # meta data of videos
    n_rows, n_cols = 3, 3  # layout of grid
    enable_text = True
    text_template = lambda _path: os.path.basename(_path)  # text wrt file path
    #### parameters end ####

    video_paths = sorted(glob(video_path))
    n_item_per_page = n_rows * n_cols
    n_pages = len(range(0, len(video_paths), n_item_per_page))

    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with imageio.v3.imopen(dump_path, "w", plugin="pyav") as video_writer:
        video_writer.init_video_stream("mpeg4", fps=30)
        for i, idx in tqdm(enumerate(range(0, len(video_paths), n_item_per_page)), total=n_pages):
            video_paths_group = video_paths[idx:idx+n_item_per_page]
            video_grid = np.zeros((n_frames, n_rows * H, n_cols * W, 3), dtype=np.uint8)
            for j, p in enumerate(tqdm(video_paths_group, desc=f'reading page {i:04d}', total=len(video_paths_group))):
                data = np.stack(imageio.mimread(p, memtest=False), axis=0)
                if enable_text:
                    text = np.zeros((H, W, 3), dtype=np.uint8)
                    text = cv2.putText(text, text_template(p), [0, 10 * max(H // 255, 1)], 0, 0.5 * max(H // 255, 1), [255, 0, 255], max(H // 255, 1))
                    data = np.where(text.any(-1, keepdims=True), text, data)
                video_grid[:, (j // n_cols) * H: (j // n_cols + 1) * H, (j % n_cols) * W: (j % n_cols + 1) * W, :] = data
            for video_frame in tqdm(video_grid, desc=f'writing page {i:04d}', total=len(video_grid)):
                video_writer.write_frame(video_frame)


def make_two_video():
    #### parameters begin ####
    video_1_path = "/home/chenxiao/下载/shared_results/outputs/sketchfab_sel/mv_4_views_xl_video/*.mp4"
    video_2_path = "/home/chenxiao/下载/shared_results/outputs/sketchfab_sel/mv_4_views_xl_video_gt/*.mp4"
    dump_path = "/home/chenxiao/下载/shared_results/outputs/sketchfab_sel/mv_4_views_xl_video_vs_grid.mp4"
    n_frames, H, W = 180, 1024, 2 * 1024  # meta data of videos
    n_rows, n_cols = 3, 1  # layout of grid
    enable_text = True
    text_template = lambda _path: os.path.basename(_path)  # text wrt file path
    #### parameters end ####


    uid_list = list(set(map(lambda p: os.path.basename(p), glob(video_1_path))).intersection(
        set(map(lambda p: os.path.basename(p), glob(video_2_path)))
    ))
    n_item_per_page = n_rows * n_cols
    n_pages = len(range(0, len(uid_list), n_item_per_page))

    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with imageio.v3.imopen(dump_path, "w", plugin="pyav") as video_writer:
        video_writer.init_video_stream("mpeg4", fps=30)
        for i, idx in tqdm(enumerate(range(0, len(uid_list), n_item_per_page)), total=n_pages):
            uid_list_group = uid_list[idx:idx+n_item_per_page]
            video_grid = np.zeros((n_frames, n_rows * H, n_cols * W, 3), dtype=np.uint8)
            for j, p in enumerate(tqdm(uid_list_group, desc=f'reading page {i:04d}', total=len(uid_list_group))):
                data_1 = np.stack(imageio.mimread(os.path.join(os.path.dirname(video_1_path), p), memtest=False), axis=0)
                data_2 = np.stack(imageio.mimread(os.path.join(os.path.dirname(video_2_path), p), memtest=False), axis=0)
                data = np.concatenate([data_1, data_2], axis=2)
                if enable_text:
                    text = np.zeros((H, W, 3), dtype=np.uint8)
                    text = cv2.putText(text, text_template(p), [0, 10 * max(H // 255, 1)], 0, 0.5 * max(H // 255, 1), [255, 0, 255], max(H // 255, 1))
                    data = np.where(text.any(-1, keepdims=True), text, data)
                video_grid[:, (j // n_cols) * H: (j // n_cols + 1) * H, (j % n_cols) * W: (j % n_cols + 1) * W, :] = data
            for video_frame in tqdm(video_grid, desc=f'writing page {i:04d}', total=len(video_grid)):
                video_writer.write_frame(video_frame)


