import rembg.sessions
from tqdm import tqdm
from typing import Any, List, Optional, Tuple, Union
from PIL import Image, ImageOps
import numpy as np
import torch
import rembg


def get_bbox(mask:np.ndarray):
    assert mask.ndim == 2
    row = mask.sum(-1)
    col = mask.sum(-2)

    row_idx = np.where(row > 0)[0]
    col_idx = np.where(col > 0)[0]
    x1, y1, x2, y2 = col_idx.min(), row_idx.min(), col_idx.max(), row_idx.max()
    return np.array([x1, y1, x2, y2])


def remove_normal_background(normal:Image.Image, pixel_threshold=20) -> Image.Image:
    normal = np.array(normal)
    mask = (np.abs(
        normal.astype(np.int32) - \
        normal[0, 0].astype(np.int32)
    ).sum(axis=-1) > pixel_threshold)
    alpha = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    return alpha


def preprocess(
    image:Image.Image, 
    alpha:Optional[Image.Image]=None, 
    H=2048, 
    W=2048, 
    scale=0.8, 
    color='white',
    return_alpha=False,
    rembg_session=None,
) -> Image.Image:

    image = ImageOps.exif_transpose(image)
    rgb = image.convert('RGB')
    if alpha is None:
        if image.mode == 'RGBA' and np.sum(np.array(image.getchannel('A')) > 0) < image.size[0] * image.size[1] - 8:
            alpha = image.getchannel('A')
        else:
            if isinstance(rembg_session, rembg.sessions.BaseSession):
                # https://github.com/pymatting/pymatting/issues/19
                rgba = rembg.remove(image, alpha_matting=True, session=rembg_session)
            else:
                rgba = rembg_session(image)
            alpha = rgba.getchannel('A')

    bboxs = get_bbox(np.array(alpha))
    x1, y1, x2, y2 = bboxs
    dy, dx = y2 - y1, x2 - x1
    s = min(H * scale / dy, W * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W - Wt) / 2), int((H - Ht) / 2)
    bboxt = np.array([ox, oy, ox+Wt, oy+Ht])

    rgbc = rgb.crop(bboxs).resize((Wt, Ht))
    alphac = alpha.crop(bboxs).resize((Wt, Ht))
    alphat = Image.new('L', (W, H))
    alphat.paste(alphac, bboxt)

    inp_1 = Image.new('RGBA', (W, H), color)
    inp_1.paste(rgbc, bboxt, alphac)
    inp_1.putalpha(alphat)

    if not return_alpha:
        return inp_1
    return inp_1, alpha


def postprocess(
    image:Image.Image, 
    alpha:Image.Image, 
    H=2048, 
    W=2048, 
    scale=0.8,
    color='white',
) -> Image.Image:
    bboxs = get_bbox(np.array(alpha))
    x1, y1, x2, y2 = bboxs
    dy, dx = y2 - y1, x2 - x1
    s = min(H * scale / dy, W * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W - Wt) / 2), int((H - Ht) / 2)
    bboxt = np.array([ox, oy, ox+Wt, oy+Ht])

    rgbc = image.convert('RGB').crop(bboxt).resize((x2 - x1, y2 - y1))
    out_1 = Image.new('RGBA', alpha.size, color)
    out_1.paste(rgbc, bboxs)
    out_1.putalpha(alpha)
    return out_1


def refine_one(mv_refiner, image: Image.Image) -> Image.Image:
    inp = torch.as_tensor(np.array(image), dtype=torch.float32).div(255.0).clamp(0.0, 1.0).unsqueeze(0).permute(0, 3, 1, 2)
    out = mv_refiner(inp)
    out = Image.fromarray(out.permute(0, 2, 3, 1).squeeze(0).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8))
    return out


def refine_batch(mv_refiner, mv_image: List[Image.Image]) -> List[Image.Image]:
    for idx in range(len(mv_image)):
        image = mv_image[idx]
        image = refine_one(mv_refiner, image)
        mv_image[idx] = image
    mv_refiner.empty_cache()
    return mv_image


def make_grid(mv_image: List[Image.Image], n_rows=2, n_cols=2) -> Image.Image:
    mv_image = np.stack([np.array(im) for im in mv_image], axis=0)
    if mv_image.ndim == 3:
        mv_image = mv_image[:, :, :, None]
    N, H, W, C = mv_image.shape
    image = mv_image.reshape(n_rows, n_cols, H, W, C).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, C)
    if image.shape[-1] == 1:
        image = image[:, :, 0]
    image = Image.fromarray(image)
    return image


def split_grid(image: Image.Image, n_rows=2, n_cols=2) -> List[Image.Image]:
    image = np.array(image)
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape
    image = image.reshape(n_rows, H // n_rows, n_cols, W // n_cols, C).transpose(0, 2, 1, 3, 4).reshape(n_rows * n_cols, H // n_rows, W // n_cols, C)
    if image.shape[-1] == 1:
        image = image[:, :, :, 0]
    mv_image = [Image.fromarray(im) for im in image]
    return mv_image


