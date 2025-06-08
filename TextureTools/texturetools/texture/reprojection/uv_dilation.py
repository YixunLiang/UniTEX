import numpy as np
import cv2
import torch


def dilate_erode(mask:torch.Tensor, kernel_size=3):
    mask = mask.to(dtype=torch.float32)
    kernel_size = 2 * (kernel_size // 2) + 1
    mask = torch.nn.functional.max_pool2d(mask, kernel_size, 1, kernel_size // 2)
    mask = 1 - torch.nn.functional.max_pool2d(1 - mask, kernel_size, 1, kernel_size // 2)
    mask = (mask > 0)
    return mask


def uv_dilation_cv(map_Kd:torch.Tensor, map_mask:torch.Tensor, max_iters=-1):
    '''
    uv map dilation (OpenCV version)

    map_Kd: [N, C, H, W], float32
    map_mask: [N, 1, H, W], bool
    '''
    _map_Kd = map_Kd.permute(0, 2, 3, 1).clamp(0, 1).mul(255).detach().cpu().numpy().astype(np.uint8)
    _map_mask = map_mask.to(dtype=torch.bool).permute(0, 2, 3, 1).mul(255).cpu().numpy().astype(np.uint8)
    for i in range(map_Kd.shape[0]):
        map_Kd[i] = torch.as_tensor(
            cv2.inpaint(_map_Kd[i], _map_mask[i], max_iters, cv2.INPAINT_TELEA),
            dtype=map_Kd.dtype, device=map_Kd.device,
        ).div(255).permute(2, 0, 1)
    return map_Kd


# NOTE: gradient accumulation will cause cuda out of memory
def uv_dilation(map_Kd:torch.Tensor, map_mask:torch.Tensor, max_iters=-1):
    '''
    uv map dilation (max pool version)

    map_Kd: [N, C, H, W], float32
    map_mask: [N, 1, H, W], bool
    '''
    _map_mask = ~map_mask.to(dtype=torch.bool)
    _map_Kd = map_Kd * _map_mask
    cnt = 0
    while True:
        if _map_mask.prod() > 0 or (max_iters > 0 and cnt >= max_iters):
            break
        _map_Kd, _map_mask = _uv_dilation_v2(_map_Kd, _map_mask)
        cnt += 1
    _map_mask = _map_mask.to(dtype=torch.float32)
    map_Kd = torch.clamp(_map_mask * _map_Kd + (1 - _map_mask) * map_Kd, 0.0, 1.0)
    return map_Kd


def get_gaussian_kernel1d(kernel_size, sigma, dtype, device):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
    kernel1d_x = get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = torch.matmul(kernel1d_y.unsqueeze(-1), kernel1d_x.unsqueeze(-2))
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d

def _uv_dilation_v1(map_Kd:torch.Tensor, map_mask:torch.Tensor, kernel_size=3):
    '''
    iterable uv map dilation

    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    kernel_size = 2 * (kernel_size // 2) + 1
    map_mask = map_mask.to(dtype=torch.float32)
    map_mask_dilation = torch.nn.functional.max_pool2d(map_mask, kernel_size, 1, kernel_size // 2)
    map_boundary_mask = (map_mask_dilation - map_mask).to(dtype=torch.bool)
    return torch.where(map_boundary_mask, torch.nn.functional.max_pool2d(map_Kd, kernel_size, 1, kernel_size // 2), map_Kd), map_mask_dilation.to(torch.bool)

def _uv_dilation_v2(map_Kd:torch.Tensor, map_mask:torch.Tensor, kernel_size=3):
    '''
    iterable uv map dilation

    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    kernel_size = 2 * (kernel_size // 2) + 1
    map_mask = map_mask.to(dtype=torch.float32)
    map_mask_dilation = torch.nn.functional.avg_pool2d(map_mask, kernel_size, 1, kernel_size // 2)
    map_dilation = torch.nn.functional.avg_pool2d(map_Kd, kernel_size, 1, kernel_size // 2)
    map_boundary_mask = torch.abs(map_mask_dilation - map_mask).to(dtype=torch.bool)
    return torch.where(map_boundary_mask, map_dilation / map_mask_dilation, map_Kd), map_mask_dilation.to(torch.bool)

def _uv_dilation_v3(map_Kd:torch.Tensor, map_mask:torch.Tensor, kernel_size=3):
    '''
    iterable uv map dilation

    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    kernel_size = 2 * (kernel_size // 2) + 1
    kernel = get_gaussian_kernel2d([kernel_size, kernel_size], [kernel_size * 0.15 + 0.35, kernel_size * 0.15 + 0.35], dtype=map_Kd.dtype, device=map_Kd.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    map_mask = map_mask.to(dtype=torch.float32)
    map_mask_dilation = torch.nn.functional.conv2d(map_mask, kernel, None, 1, kernel_size // 2)
    map_dilation = torch.nn.functional.conv2d(map_Kd, kernel.tile(map_Kd.shape[1], 1, 1, 1), None, 1, kernel_size // 2, groups=map_Kd.shape[1])
    map_boundary_mask = torch.abs(map_mask_dilation - map_mask).to(dtype=torch.bool)
    return torch.where(map_boundary_mask, map_dilation / map_mask_dilation, map_Kd), map_mask_dilation.to(torch.bool)

