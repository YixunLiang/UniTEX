'''
https://github.com/dcoeurjo/OTColorTransfer
https://dcoeurjo.github.io/OTColorTransfer/python
'''

import numpy as np
import cv2
from tqdm import tqdm

def CTSOT(src, dst, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer, ported by @iperov

    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value - clip it manually
    """
    dtype = src.dtype
    H, W, C = src.shape
    new_src = src.copy().reshape(-1, C)
    dst = dst.reshape(-1, C)
    for step in tqdm(range(steps)):
        advect = np.zeros_like(new_src)
        for batch in range (batch_size):
            dir = np.random.normal(size=(C,)).astype(dtype)
            dir = dir / np.linalg.norm(dir, axis=-1)
            projsource = np.sum(new_src * dir, axis=-1)
            projtarget = np.sum(dst * dir, axis=-1)
            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)
            a = projtarget[idTarget] - projsource[idSource]
            advect[idSource] += a[:, None] * dir
        new_src += advect / batch_size
    new_src = new_src.reshape(H, W, C)
    if reg_sigmaXY != 0.0:
        new_src = src + cv2.bilateralFilter(new_src - src, 0, reg_sigmaV, reg_sigmaXY)
    return new_src



if __name__ == '__main__':
    src_path = '/home/chenxiao/code/MVDiffusion/test_result/test_geometry_renderer/pred/uv_pcd_reproject_middle.png'
    dst_path = '/home/chenxiao/code/MVDiffusion/test_result/test_geometry_renderer/pred/uv_pcd_reproject_origin.png'
    out_path = '/home/chenxiao/code/MVDiffusion/test_result/test_geometry_renderer/pred/uv_pcd_reproject_debug.png'

    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    out = CTSOT((src / 255.0).astype(np.float32), (dst / 255.0).astype(np.float32), reg_sigmaXY=16.0) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)

