import os
from time import sleep
import imageio
from tqdm import tqdm
from typing import List
import numpy as np
from PIL.ImageColor import colormap
import cv2
import torch


def image_fusion(src:torch.Tensor, dst:torch.Tensor, mask:torch.Tensor, n_erode_pix=0, color_transfer=False) -> torch.Tensor:
    '''
    src: [B, H, W, 3]
    dst: [B, H, W, 3]
    mask: [B, H, W, 1]
    res: [B, H, W, 3]
    '''
    shape = torch.broadcast_shapes(src.shape, dst.shape, mask.shape)
    batch_size, H, W, _ = shape
    src = src.expand(batch_size, H, W, -1)
    dst = dst.expand(batch_size, H, W, -1)
    mask = mask.expand(batch_size, H, W, -1)

    src_np = src.clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    dst_np = dst.clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    mask_np = mask[:, :, :, -1].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    res = torch.empty_like(src)
    if color_transfer:
        for i in range(batch_size):
            res_np = transfer_skin_color_np(src_np[i], dst_np[i], (mask_np[i] > 0))
            res_np = image_fusion_np(src_np[i], res_np, mask_np[i], n_erode_pix=n_erode_pix)
            res[i] = torch.as_tensor(res_np, dtype=src.dtype, device=src.device).div(255.0)
    else:
        for i in range(batch_size):
            res_np = image_fusion_np(src_np[i], dst_np[i], mask_np[i], n_erode_pix=n_erode_pix)
            res[i] = torch.as_tensor(res_np, dtype=src.dtype, device=src.device).div(255.0)
    return res

def image_fusion_np(src: np.ndarray, dst: np.ndarray, mask:np.ndarray, n_erode_pix=0) -> np.ndarray:
    '''
    scr, dst: uint8
    mask: mask wrt src, uint8, range(0, 255)
    n_erode_pix: num of erode pixels for mask
    res: uint8, change pixels in mask, keep pixels out of mask
    '''
    if n_erode_pix > 0:
        mask = cv2.erode(mask, np.ones((n_erode_pix, n_erode_pix), dtype=np.uint8))
    elif n_erode_pix < 0:
        mask = cv2.dilate(mask, np.ones((-n_erode_pix, -n_erode_pix), dtype=np.uint8))
    x1, y1, dx, dy = cv2.boundingRect(mask)
    res = cv2.seamlessClone(
        src, dst, mask,
        p=(x1+dx//2, y1+dy//2),
        flags=cv2.NORMAL_CLONE,
    )
    # res = cv2.rectangle(res, (x1, y1), (x1+dx, y1+dy), [255, 0, 255], 1)
    return res

def image_soft_fusion_np(src: np.ndarray, dst: np.ndarray, mask:np.ndarray, n_erode_pix=0) -> np.ndarray:
    '''
    scr, dst: uint8
    mask: mask wrt src, uint8, range(0, 255)
    n_erode_pix: num of erode pixels for mask
    res: uint8, change pixels in mask, keep pixels out of mask
    '''
    if n_erode_pix > 0:
        mask = cv2.erode(mask, np.ones((n_erode_pix, n_erode_pix), dtype=np.uint8))
    elif n_erode_pix < 0:
        mask = cv2.dilate(mask, np.ones((-n_erode_pix, -n_erode_pix), dtype=np.uint8))
    alpha = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), dtype=np.uint8)), (175, 175), 1.0, 1.0)
    res = np.clip(src * (alpha[..., None] / 255.0) + dst * (1.0 - alpha[..., None] / 255.0), 0, 255).astype(np.uint8)
    return res

def test_image_fusion():
    src = cv2.imread('test_result/test_blending/debug6/debug_-7_src.png')
    dst = cv2.imread('test_result/test_blending/debug6/debug_-7_dst.png')
    mask = cv2.imread('test_result/test_blending/debug6/debug_-7_mask_v2.png')
    mask = mask[..., -1]
    n_erode_pix = -1

    # mask = np.zeros_like(mask)
    # mask[1024-512:1024+512, 1024-512:1024+512] = 255
    # mask[:5, :] = 0
    # mask[-5:, :] = 0
    # mask[:, -5:] = 0
    # mask[:, -5:] = 0
    # mask = cv2.erode(mask, np.ones((15, 15), dtype=np.uint8))
    # mask = cv2.dilate(mask, np.ones((15, 15), dtype=np.uint8))

    if n_erode_pix > 0:
        mask = cv2.erode(mask, np.ones((n_erode_pix, n_erode_pix), dtype=np.uint8))
    elif n_erode_pix < 0:
        mask = cv2.dilate(mask, np.ones((-n_erode_pix, -n_erode_pix), dtype=np.uint8))
    x1, y1, dx, dy = cv2.boundingRect(mask)
    for _ in tqdm(range(10)):
        res = cv2.seamlessClone(
            src, dst, mask,
            p=(x1+dx//2, y1+dy//2),
            flags=cv2.NORMAL_CLONE,
        )
        dst = res
    res = cv2.rectangle(res, (x1, y1), (x1+dx, y1+dy), [255, 0, 255], 1)
    # mask_inv = (255 - mask.astype(np.int64)).astype(np.uint8)
    # x1, y1, dx, dy = cv2.boundingRect(mask_inv)
    # res2 = cv2.seamlessClone(
    #     dst, res, mask,
    #     p=(x1+dx//2, y1+dy//2),
    #     flags=cv2.MONOCHROME_TRANSFER,
    # )
    # res2 = cv2.rectangle(res2, (x1, y1), (x1+dx, y1+dy), [255, 0, 255], 1)

    alpha = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), dtype=np.uint8)), (175, 175), 1.0, 1.0)
    res3 = np.clip(src * (alpha[..., None] / 255.0) + dst * (1.0 - alpha[..., None] / 255.0), 0, 255).astype(np.uint8)

    cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_mask.png', mask)
    # cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_mask_inv.png', mask_inv)
    cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_res.png', res)
    # cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_res2.png', res2)
    cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_alpha.png', alpha)
    cv2.imwrite('test_result/test_blending/debug6/debug_5_5_again_res3.png', res3)
    

def get_major_mean_std_np(input_img_hsv, mask):
    masked_img = input_img_hsv[mask]
    valids_x_values = masked_img[:, 0]
    valids_y_values = masked_img[:, 1]
    histr, x_edge, y_edge = np.histogram2d(valids_x_values, valids_y_values, bins=[30, 30])
    max_idx_x, max_idx_y = np.unravel_index(np.argmax(histr, axis=None), histr.shape)

    x_bin_value_left, x_bin_value_right = x_edge[max_idx_x], x_edge[max_idx_x + 1]
    y_bin_value_left, y_bin_value_right = y_edge[max_idx_y], y_edge[max_idx_y + 1]

    a = valids_x_values >= x_bin_value_left
    b = valids_x_values <= x_bin_value_right
    max_x_bin_mask = np.logical_and(valids_x_values >= x_bin_value_left, valids_x_values <= x_bin_value_right)
    max_y_bin_mask = np.logical_and(valids_y_values >= y_bin_value_left, valids_y_values <= y_bin_value_right)
    max_bin_mask = np.logical_and(max_x_bin_mask, max_y_bin_mask)
    max_bin_values = masked_img[max_bin_mask, :]
    bin_values_mean = np.mean(max_bin_values, axis=0)
    bin_values_std = np.std(max_bin_values, axis=0)
    return bin_values_mean, bin_values_std

def transfer_skin_color_np(src_img, tgt_img, mask):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    source_major_mean, _ = get_major_mean_std_np(src_img, mask)
    target_major_mean, _ = get_major_mean_std_np(tgt_img, mask)
    target_new = (tgt_img - target_major_mean) + source_major_mean

    blender_weight_norm = np.linalg.norm(tgt_img - target_major_mean, axis=-1)
    blender_weight_norm_max = blender_weight_norm.max()
    blender_weight_norm_min = blender_weight_norm.min()
    blender_weight = (blender_weight_norm_max - blender_weight_norm) / (blender_weight_norm_max - blender_weight_norm_min)
    blender_weight = np.tile(blender_weight[:, :, None], [1, 1, 3])

    target_blender = blender_weight * target_new + (1 - blender_weight) * tgt_img
    target_final = cv2.cvtColor(np.clip(target_blender, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return target_final

def smooth_fusion_np(src:np.ndarray, masks:List[np.ndarray], n_erode_pix=0, n_iters=20, return_history=False) -> np.ndarray:
    '''
    scr: uint8
    masks: masks wrt src, uint8, range(0, 255)
    n_erode_pix: num of erode pixels for masks
    res: uint8
    '''
    if return_history:
        history = []
    for _ in tqdm(range(n_iters)):
        for mask in masks:
            dst = cv2.inpaint(src, mask, -1, flags=cv2.INPAINT_NS)
            src = image_fusion_np(src, dst, mask[:, :, -1], n_erode_pix=n_erode_pix)
            if return_history:
                history.append(np.concatenate([src, dst, mask[..., [0,0,0]]], axis=1))
    if return_history:
        return src, history
    return src

def test_smooth_fusion():
    H, W = 2048, 2048
    Hp, Wp = 256, 256
    color_dict = {k: np.array([int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)], dtype=np.uint8) for k, v in colormap.items()}
    color_values = np.stack(list(color_dict.values()), axis=0)
    generator = np.random.default_rng(seed=666)
    random_index = generator.choice(len(color_values), size=(H // Hp) * (W // Wp), replace=True)
    chess_board = color_values[random_index].reshape(H // Hp, W // Wp, 3)
    chess_board = np.tile(chess_board[:, None, :, None, :], (1, Hp, 1, Wp, 1)).reshape(H, W, 3)
    chess_board_masks = np.eye((H // Hp) * (W // Wp), dtype=bool).reshape((H // Hp) * (W // Wp), H // Hp, W // Wp, 1)
    chess_board_masks = np.tile(chess_board_masks[:, :, None, :, None, :], (1, 1, Hp, 1, Wp, 1)).reshape((H // Hp) * (W // Wp), H, W, 1)
    chess_board_smooth, history = smooth_fusion_np(
        chess_board,
        chess_board_masks.astype(np.uint8) * 255, 
        n_erode_pix=0,
        n_iters=20,
        return_history=True,
    )
    cv2.imwrite('chess_board.png', chess_board)
    cv2.imwrite('chess_board_masks.png', chess_board_masks[0].astype(np.uint8) * 255)
    cv2.imwrite('chess_board_smooth.png', chess_board_smooth)
    imageio.mimsave('chess_board_history.mp4', history, fps=5)  # FIXME: cpu may stuck at here

def boundary_color_shift_np(src:np.ndarray, mask_boundary:np.ndarray):
    mask = np.zeros_like(mask_boundary)
    contours, _ = cv2.findContours(mask_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # areas = [cv2.contourArea(contour) for contour in contours]
    for idx, countour in enumerate(contours):
        mask = cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
    src = cv2.dilate(src, np.ones((9, 9), dtype=np.uint8))
    dst = cv2.inpaint(src, mask, 3, flags=cv2.INPAINT_TELEA)
    return dst

def boundary_color_shift_v2_np(src:np.ndarray, mask_boundary:np.ndarray):
    from scipy.interpolate import LinearNDInterpolator
    H, W, _ = src.shape
    src = src / 255.0
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy = np.stack([gx, gy], axis=-1)
    mask_boundary_bool = (mask_boundary > 0)
    xy_valid = xy[mask_boundary_bool]
    src_valid = src[mask_boundary_bool]
    dst = LinearNDInterpolator(xy_valid, src_valid)(gx, gy)
    dst = np.nan_to_num(dst, nan=0.0)
    dst = np.clip(dst * 255.0, 0, 255).astype(np.uint8)
    return dst

def boundary_color_shift_v3_np(src:np.ndarray, mask_boundary:np.ndarray):
    from scipy.spatial import KDTree
    H, W, _ = src.shape
    src = src / 255.0
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy = np.stack([gx, gy], axis=-1)
    mask_boundary_bool = (mask_boundary > 0)
    xy_valid = xy[mask_boundary_bool]
    src_valid = src[mask_boundary_bool]
    score, idx = KDTree(xy_valid).query(xy, k=64)
    weight = np.nan_to_num(1.0 / score, nan=0.0)
    weight = weight[..., None]
    dst = np.sum(weight * src_valid[idx, :], axis=-2, keepdims=False) / np.sum(weight, axis=-2, keepdims=False)
    dst = np.nan_to_num(dst, nan=0.0)
    dst = np.clip(dst * 255.0, 0, 255).astype(np.uint8)
    return dst

def boundary_color_shift_v4_np(src:np.ndarray, mask_boundary:np.ndarray):
    H, W, _ = src.shape
    src = src / 255.0
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy = np.stack([gx, gy], axis=-1)
    mask_boundary_bool = (mask_boundary > 0)
    xy_valid = xy[mask_boundary_bool]
    src_valid = src[mask_boundary_bool]

    H, W = 128, 128
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy_ds = np.stack([gx, gy], axis=-1)
    batch_size = 2 ** 10
    xy_ds = xy_ds.reshape(-1, 2)
    dst = np.zeros((xy_ds.shape[0], 3), dtype=np.float32)
    for idx in tqdm(range(0, xy_ds.shape[0], batch_size)):
        xy_ds_batch = xy_ds[idx:idx+batch_size, :]
        score = np.square(xy_ds_batch[:, None, :] - xy_valid).sum(axis=-1, keepdims=False)
        weight = np.nan_to_num(1.0 / score, nan=0.0)
        weight = weight[..., None]
        dst_batch = np.sum(weight * src_valid, axis=-2, keepdims=False) / np.sum(weight, axis=-2, keepdims=False)
        dst[idx:idx+batch_size, :] = dst_batch
    dst = dst.reshape(H, W, 3)
    dst = np.clip(dst * 255.0, 0, 255).astype(np.uint8)
    return dst

def boundary_color_shift_v5_np(src:np.ndarray, mask_boundary:np.ndarray):
    from scipy.spatial import KDTree
    H, W, _ = src.shape
    src = src / 255.0
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy = np.stack([gx, gy], axis=-1)
    contours, _ = cv2.findContours(mask_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # areas = [cv2.contourArea(contour) for contour in contours]
    dst = np.zeros_like(src)
    for idx, contour in tqdm(enumerate(contours), total=len(contours)):
        mask = np.zeros_like(mask_boundary)
        mask = cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
        mask_select = mask_boundary.copy()
        mask_select[~mask] = 0
        xy_select = xy[mask > 0]
        xy_valid_select = xy[mask_select > 0]
        src_valid_select = src[mask_select > 0]
        score, idx = KDTree(xy_valid_select).query(xy_select, k=64)
        weight = np.nan_to_num(1.0 / score, nan=0.0)
        weight = weight[..., None]
        dst_select = np.sum(weight * src_valid_select[idx, :], axis=-2, keepdims=False) / np.sum(weight, axis=-2, keepdims=False)
        dst_select = np.nan_to_num(dst_select, nan=0.0)
        dst_select = np.clip(dst_select * 255.0, 0, 255).astype(np.uint8)
        dst[mask > 0] = dst_select
    return dst

def boundary_color_shift_v6_np(src:np.ndarray, mask_boundary:np.ndarray):
    H, W, _ = src.shape
    src = src / 255.0
    gy, gx = np.meshgrid(
        np.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=np.float32),
        np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=np.float32),
        indexing='ij',
    )
    xy = np.stack([gx, gy], axis=-1)
    contours, _ = cv2.findContours(mask_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # areas = [cv2.contourArea(contour) for contour in contours]
    dst = np.zeros_like(src)
    for idx, contour in tqdm(enumerate(contours), total=len(contours)):
        mask = np.zeros_like(mask_boundary)
        mask = cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
        mask_select = mask_boundary.copy()
        mask_select[~mask] = 0
        xy_select = xy[mask > 0]
        xy_valid_select = xy[mask_select > 0]
        src_valid_select = src[mask_select > 0]
        score = np.square(xy_valid_select[:, None, :] - xy_select).sum(axis=-1, keepdims=False)
        weight = np.nan_to_num(1.0 / score, nan=0.0)
        weight = weight[..., None]
        dst_select = np.sum(weight * src_valid_select, axis=-2, keepdims=False) / np.sum(weight, axis=-2, keepdims=False)
        dst_select = np.nan_to_num(dst_select, nan=0.0)
        dst_select = np.clip(dst_select * 255.0, 0, 255).astype(np.uint8)
        dst[mask > 0] = dst_select
    return dst

def test_boundary_color_shift():
    test_root = '/home/chenxiao/code/MVDiffusion/test_data/online_cases/d2b57ca9-f299-44d8-b265-b28d4db5f8d7_outputs/cache/'

    src = cv2.imread(os.path.join(test_root, 'blended_uv_rgb_boundary_err.png'), -1)
    mask_boundary = cv2.imread(os.path.join(test_root, 'blended_uv_rgb_boundary.png'), -1)
    mask_boundary = ((mask_boundary > 0).any(axis=-1) * 255).astype(np.uint8)
    dst = boundary_color_shift_np(src, mask_boundary)
    cv2.imwrite(os.path.join(test_root, 'blended_uv_rgb_boundary_shift.png'), dst)

def image_fusion_hybrid(src:torch.Tensor, dst:torch.Tensor, mask:torch.Tensor, mask_boundary:torch.Tensor, n_erode_pix=0) -> torch.Tensor:
    '''
    src: [B, H, W, 3]
    dst: [B, H, W, 3]
    mask: [B, H, W, 1]
    mask_boundary: [B, H, W, 1]
    res: [B, H, W, 3]
    '''
    shape = torch.broadcast_shapes(src.shape, dst.shape, mask.shape)
    batch_size, H, W, _ = shape
    src = src.expand(batch_size, H, W, -1)
    dst = dst.expand(batch_size, H, W, -1)
    mask = mask.expand(batch_size, H, W, -1)
    mask_boundary = mask_boundary.expand(batch_size, H, W, -1)

    src_np = src.clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    dst_np = dst.clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    mask_np = mask[:, :, :, -1].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    mask_boundary = mask_boundary[:, :, :, -1].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    res = torch.empty_like(src)
    for i in range(batch_size):
        res_np = image_fusion_hybrid_np(src_np[i], dst_np[i], mask_np[i], mask_boundary[i], n_erode_pix=n_erode_pix)
        res[i] = torch.as_tensor(res_np, dtype=src.dtype, device=src.device).div(255.0)
    return res

def image_fusion_hybrid_np(src: np.ndarray, dst: np.ndarray, mask:np.ndarray, mask_boundary:np.ndarray, n_erode_pix=0) -> np.ndarray:
    '''
    scr, dst: uint8
    mask: mask wrt src, uint8, range(0, 255)
    mask_boundary: boundary in mask region for hard blending, uint8, range(0, 255)
    n_erode_pix: num of erode pixels for mask
    res: uint8, change pixels in mask, keep pixels out of mask
    '''
    if n_erode_pix > 0:
        mask = cv2.erode(mask, np.ones((n_erode_pix, n_erode_pix), dtype=np.uint8))
    elif n_erode_pix < 0:
        mask = cv2.dilate(mask, np.ones((-n_erode_pix, -n_erode_pix), dtype=np.uint8))
    mask_boundary = mask_boundary * (mask > 127)
    dst = np.clip((src * (mask[..., None] / 255.0) + dst * (1.0 - mask[..., None] / 255.0)) * (mask_boundary[..., None] / 255.0) + dst * (1.0 - mask_boundary[..., None] / 255.0), 0.0, 255.0).astype(np.uint8)
    x1, y1, dx, dy = cv2.boundingRect(mask)
    res = cv2.seamlessClone(
        src, dst, mask,
        p=(x1+dx//2, y1+dy//2),
        flags=cv2.NORMAL_CLONE,
    )
    # res = cv2.rectangle(res, (x1, y1), (x1+dx, y1+dy), [255, 0, 255], 1)
    return res

def test_image_fusion_hybrid():
    test_root = '/home/chenxiao/code/MVDiffusion/test_data/online_cases/d2b57ca9-f299-44d8-b265-b28d4db5f8d7_outputs/cache/'

    img = cv2.imread(os.path.join(test_root, 'rembg_image.png'), -1)
    fg, alpha = img[..., :3], img[..., [3]]
    alpha = ((alpha > 127) * 255).astype(np.uint8)
    bg = (np.ones_like(fg) * np.asarray([255, 0, 0])).astype(np.uint8)  # bgr

    cv2.imwrite(os.path.join(test_root, 'composed_fg.png'), fg)
    cv2.imwrite(os.path.join(test_root, 'composed_alpha.png'), alpha)
    cv2.imwrite(os.path.join(test_root, 'composed_bg.png'), bg)

    composed_hard = np.clip(fg * (alpha / 255.0) + bg * (1.0 - alpha / 255.0), 0.0, 255.0).astype(np.uint8)
    composed_poission = np.clip(image_fusion_np(fg, bg, alpha), 0.0, 255.0).astype(np.uint8)

    ## Method 1: composed_hard + bg
    hybrid_bg = bg.copy()
    hybrid_bg[:512, :, :] = composed_hard[:512, :, :]
    composed_hybrid = np.clip(image_fusion_np(fg, hybrid_bg, alpha), 0.0, 255.0).astype(np.uint8)

    ## Method 2: composed_hard(boundary) + bg
    alpha_boundary = (alpha[..., 0] - cv2.erode(alpha[..., 0], np.ones((9, 9), dtype=np.uint8)))[..., None]
    hybrid_bg = np.clip(composed_hard * (alpha_boundary / 255.0) + bg * (1.0 - alpha_boundary / 255.0), 0.0, 255.0).astype(np.uint8)
    hybrid_bg[512:, ...] = bg[512:, ...]
    composed_hybrid = np.clip(image_fusion_np(fg, hybrid_bg, alpha[..., 0]), 0.0, 255.0).astype(np.uint8)

    ## Ensembled Method
    alpha_boundary = (alpha[..., 0] - cv2.erode(alpha[..., 0], np.ones((9, 9), dtype=np.uint8)))[..., None]
    alpha_boundary[512:, ...] = 0
    composed_hybrid = np.clip(image_fusion_hybrid_np(fg, bg, alpha[..., 0], alpha_boundary[..., 0]), 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(os.path.join(test_root, 'composed_hard.png'), composed_hard)
    cv2.imwrite(os.path.join(test_root, 'composed_poission.png'), composed_poission)
    cv2.imwrite(os.path.join(test_root, 'composed_hybrid_background.png'), hybrid_bg)
    cv2.imwrite(os.path.join(test_root, 'composed_hybrid.png'), composed_hybrid)


if __name__ == '__main__':
    test_image_fusion()
    # test_smooth_fusion()
    # test_boundary_color_shift()
    # test_image_fusion_hybrid()



