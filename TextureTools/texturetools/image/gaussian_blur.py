# https://github.com/pytorch/vision/issues/7413
# https://github.com/pytorch/vision/issues/6437

from typing import List, Optional
import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur as gaussian_blur_tv


# copy from torchvision.transforms.functional.gaussian_blur
def _check_input(kernel_size: List[int], sigma: Optional[List[float]] = None):
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}")
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")
    return kernel_size, sigma


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None, backend='torchvision') -> Tensor:
    if backend == 'torchvision':
        return gaussian_blur_tv(img, kernel_size=kernel_size, sigma=sigma)
    elif backend == 'opencv':
        assert img.ndim == 4, f'shape of img should be [B, C, H, W], but ndim is {img.ndim}'
        kernel_size, sigma = _check_input(kernel_size=kernel_size, sigma=sigma)
        img_np = img.permute(0, 2, 3, 1).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        out_np = np.empty_like(img_np)
        if img_np.shape[-1] > 1:
            for i in range(img_np.shape[0]):
                out_np[i, :, :, :] = cv2.GaussianBlur(img_np[i, :, :, :], kernel_size, sigmaX=sigma[0], sigmaY=sigma[1])
        elif img_np.shape[-1] == 1:
            for i in range(img_np.shape[0]):
                out_np[i, :, :, 0] = cv2.GaussianBlur(img_np[i, :, :, 0], kernel_size, sigmaX=sigma[0], sigmaY=sigma[1])
        out = torch.as_tensor(out_np, dtype=img.dtype, device=img.device).div(255.0).permute(0, 3, 1, 2)
        return out
    else:
        raise NotImplementedError(backend)

