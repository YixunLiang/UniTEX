'''
for mv texture generation
'''
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2


prefix = 'mv_box_views_xl_v2.0'
src_1 = f'/home/chenxiao/下载/shared_results/inputs/clay_native3d_v2/*/reference_image.png'
src_2 = f'/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/{prefix}/*/mv_rgb.png'
dst_1 = f'/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/{prefix}_grid/*.png'
dst_2 = f'/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2'

root_src_1 = f'/home/chenxiao/下载/shared_results/inputs/clay_native3d_v2'
root_src_2 = f'/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/{prefix}'
root_dst_1 = f'/home/chenxiao/下载/shared_results/outputs/clay_native3d_v2/{prefix}_grid'


src_1 = glob(src_1)
src_2 = glob(src_2)
uid_list = set(map(lambda p: os.path.basename(os.path.dirname(p)), src_1)).intersection(
    set(map(lambda p: os.path.basename(os.path.dirname(p)), src_2))
)


for uid in tqdm(uid_list):
    im_1 = cv2.resize(cv2.imread(os.path.join(root_src_1, uid, 'reference_image.png'), -1), (512, 512))
    im_2 = cv2.resize(cv2.imread(os.path.join(root_src_2, uid, 'mv_rgb.png'), -1), (6 * 512, 512))
    if im_1.shape[-1] == 3:
        im_1 = np.concatenate([im_1, np.full_like(im_1[..., [0]], 255)], axis=-1)
    if im_2.shape[-1] == 3:
        im_2 = np.concatenate([im_2, np.full_like(im_2[..., [0]], 255)], axis=-1)
    im = np.concatenate([im_1, im_2], axis=1)
    p = os.path.join(root_dst_1, uid+'.png')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, im)

grid = []
for p in tqdm(glob(dst_1)):
    grid.append(cv2.imread(p, -1))

for idx in tqdm(range(0, len(grid), 10)):
    subgrid = grid[idx:idx+10]
    subgrid = np.concatenate(subgrid, axis=0)
    subgrid[:, 512-2:512+2, :] = [0, 0, 0, 255]
    p = os.path.join(dst_2, f'{prefix}_grid_{idx:04d}.png')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, subgrid)

