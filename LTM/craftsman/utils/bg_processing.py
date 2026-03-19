from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import os
import sys
import time
import cv2
import gradio as gr
import numpy as np
import torch
import PIL
from PIL import Image
import rembg
from rembg import remove
rembg_session = rembg.new_session()

def rmbg_sam(iamge, foreground_ratio):
    return iamge

def rmbg_rembg(iamge, foreground_ratio):
    return iamge

class RMBG(object):
    def __init__(self):
        pass

    def rmbg_rembg(self, input_image, background_color):
        def _rembg_remove(
            image: PIL.Image.Image,
            rembg_session = None,
            force: bool = False,
            **rembg_kwargs,
        ) -> PIL.Image.Image:
            do_remove = True
            if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
                # explain why current do not rm bg
                print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)

            # calculate the min bbox of the image
            alpha = image.split()[-1]
            image = image.crop(alpha.getbbox())
            return image
        return _rembg_remove(input_image, None, force_remove=True)

    def run(self, rm_type, image, foreground_ratio, background_choice, background_color=(255, 255, 255)):
        if "Original" in background_choice:
            return image
        else:
            if background_choice == "Alpha as mask":
                alpha = image.split()[-1]
                image = image.crop(alpha.getbbox())
            
            elif "Remove" in background_choice:
                if rm_type.upper() == "REMBG":
                    image = self.rmbg_rembg(image, background_color=background_color)
                else:
                    return -1
        
            # Calculate the new size after rescaling
            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            paste_position = ((image.width - resized_image.width) // 2, (image.height - resized_image.height) // 2)
            padded_image.paste(resized_image, paste_position)

            # expand image to 1:1
            width, height = padded_image.size
            if width == height:
                image = padded_image.resize((512, 512))
                return image.convert("RGB")
            new_size = (max(width, height), max(width, height))
            image = Image.new("RGBA", new_size, (*background_color, 255))
            paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
            image.paste(padded_image, paste_position)
            image = image.resize((512, 512))
            return image.convert("RGB")
        

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr

def prepare_data(single_image, crop_size):
    from apps.third_party.Wonder3D.mvdiffusion.data.single_image_dataset import SingleImageDataset
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[256, 256], bg_color='gray', crop_size=crop_size, single_image=single_image)
    return dataset[0]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def remove_background(
    image: PIL.Image.Image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)