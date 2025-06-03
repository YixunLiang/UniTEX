import cv2
import numpy as np
import torch


def image_outpainting(src:torch.Tensor, mask:torch.Tensor, n_erode_pix=0, n_radius=1) -> torch.Tensor:
    '''
    src: [B, H, W, 3]
    mask: [B, H, W, 1]
    '''
    shape = torch.broadcast_shapes(src.shape, mask.shape)
    batch_size, H, W, _ = shape
    src = src.expand(batch_size, H, W, -1)
    mask = mask.expand(batch_size, H, W, -1)

    src_np = src.clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    mask_np = mask[:, :, :, -1].clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8)
    res = torch.empty_like(src)
    for i in range(batch_size):
        res_np = image_outpainting_np(src_np[i], mask_np[i], n_erode_pix=n_erode_pix, n_radius=n_radius)
        res[i] = torch.as_tensor(res_np, dtype=src.dtype, device=src.device).div(255.0)
    return res


def image_outpainting_np(src: np.ndarray, mask:np.ndarray, n_erode_pix=0, n_radius=1) -> np.ndarray:
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
    res = cv2.inpaint(
        src, mask,
        inpaintRadius=n_radius,
        flags=cv2.INPAINT_NS,
    )
    return res


