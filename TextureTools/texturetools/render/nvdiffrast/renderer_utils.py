'''
Common Utils for NVDiffrast Renderers
'''
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import imageio
import numpy as np
from PIL import Image
import torch


mode_to_dtype = {
    "I": np.int32,
    "I;16": np.int16,
    "I;16B": np.int16,
    "F": np.float32,
}
dtype_to_fmax = {
    np.uint8: 255.0,
    np.int16: 65535.0,
    np.int32: 4294967295.0,
    np.float32: 1.0,
}
def image_to_tensor(image:Union[Image.Image, List[Image.Image], Tuple[Image.Image]], device='cuda') -> torch.Tensor:
    if isinstance(image, Image.Image):
        dtype = mode_to_dtype.get(image.mode, np.uint8)
        # NOTE: division may cause overflow if do not convert to float32 here
        image = np.array(image, dtype=np.float32)
        image = image / dtype_to_fmax[dtype]
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        elif image.ndim == 3:
            if image.shape[-1] == 1:
                image = np.tile(image, (1, 1, 3))
            if image.shape[-1] == 2:
                image = np.concatenate([image, np.ones_like(image[..., [0]])], axis=-1)
            if image.shape[-1] > 3:
                image = image[:, :, :3]
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    elif isinstance(image, (List, Tuple)):
        dtype = mode_to_dtype.get(image[0].mode, np.uint8)
        # NOTE: division may cause overflow if do not convert to float32 here
        image = np.stack([np.array(im, dtype=np.float32) for im in image], axis=0)
        image = image / dtype_to_fmax[dtype]
        if image.ndim == 3:
            image = np.tile(image[:, :, :, None], (1, 1, 1, 3))
        elif image.ndim == 4:
            if image.shape[-1] == 1:
                image = np.tile(image, (1, 1, 1, 3))
            if image.shape[-1] == 2:
                image = np.concatenate([image, np.ones_like(image[..., [0]])], axis=-1)
            if image.shape[-1] > 3:
                image = image[:, :, :, :3]
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    return tensor


def tensor_to_image(tensor:torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
    if tensor.ndim == 3:
        image = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        if image.shape[-1] == 1:
            image = Image.fromarray(image[:, :, 0], mode='L')
        elif image.shape[-1] == 3:
            image = Image.fromarray(image, mode='RGB')
        elif image.shape[-1] == 4:
            image = Image.fromarray(image, mode='RGBA')
        else:
            raise NotImplementedError(f'num of channels error: {image.shape[-1]}')
    elif tensor.ndim == 4:
        image = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        if image.shape[-1] == 1:
            image = [Image.fromarray(im[:, :, 0], mode='L') for im in image]
        elif image.shape[-1] == 3:
            image = [Image.fromarray(im, mode='RGB') for im in image]
        elif image.shape[-1] == 4:
            image = [Image.fromarray(im, mode='RGBA') for im in image]
        else:
            raise NotImplementedError(f'num of channels error: {image.shape[-1]}')
    return image


def tensor_to_video(tensor:torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        video = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    return video


