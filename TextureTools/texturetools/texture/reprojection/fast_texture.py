'''
reimplementation for inverse rendering
'''
import json
import math
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import Image

from ...utils.timer import CPUTimer
from ...io.mesh_loader import load_whole_mesh
from ...mesh.structure import Mesh, Texture
from ...camera.conversion import intr_to_proj, c2w_to_w2c
from ...camera.generator import generate_intrinsics, generate_orbit_views_c2ws, generate_semisphere_views_c2ws
from ...render.nvdiffrast.renderer_base import NVDiffRendererBase
from ..stitching.mip import pull_push
from .uv_dilation import dilate_erode
from ...video.export_nvdiffrast_uv_video import export_video
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline

def tensor_to_ndarray(images:torch.Tensor) -> np.ndarray:
    if images.ndim == 4:
        images = images.permute(0, 2, 3, 1)
    elif images.ndim == 3:
        images = images.permute(1, 2, 0)
    else:
        raise NotImplementedError
    images = images.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    return images

def ndarray_to_tensor(images:np.ndarray) -> torch.Tensor:
    images = torch.as_tensor(images, dtype=torch.float32).div(255.0).clamp(0.0, 1.0)
    if images.ndim == 4:
        images = images.permute(0, 3, 1, 2)
    elif images.ndim == 3:
        images = images.permute(2, 0, 1)
    else:
        raise NotImplementedError
    return images

def make_grid(images:np.ndarray, n_rows:int, n_cols:int):
    B, H, W, C = images.shape
    grid = images.reshape(n_rows, n_cols, H, W, C).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, C)
    return grid


class ImageInpaintingModel:
    def __init__(self, base_model='flux-inpaint'):
        self.base_model = base_model
        if base_model == 'sdxl':
            unet = UNet2DConditionModel.from_pretrained(
                "pretrain_models/stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="unet",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.inpainting_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                "pretrain_models/stabilityai/stable-diffusion-xl-base-1.0",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.inpainting_pipeline.to("cuda")
        elif base_model == 'sdxl-inpainting':
            unet = UNet2DConditionModel.from_pretrained(
                "pretrain_models/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                subfolder="unet",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.inpainting_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                "pretrain_models/stabilityai/stable-diffusion-xl-base-1.0",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.inpainting_pipeline.to("cuda")
        elif base_model == 'flux':
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = FluxTransformer2DModel.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-dev",
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16
            )
            self.inpainting_pipeline = FluxInpaintPipeline.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-dev",
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
            )
            self.inpainting_pipeline.enable_model_cpu_offload()
        elif base_model == 'flux-inpainting':
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = FluxTransformer2DModel.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-Fill-dev",
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16
            )
            self.inpainting_pipeline = FluxFillPipeline.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-dev",
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
            )
            self.inpainting_pipeline.enable_model_cpu_offload()
        else:
            raise NotImplementedError(f'base_model {base_model} is not supported')
    
    def __call__(self, images, images_mask):
        prompt = ''
        images_rgb = images[:, :3, :, :] * images[:, [3], :, :] + 1.0 * (1.0 - images[:, [3], :, :])
        images_rgb = [Image.fromarray(im) for im in tensor_to_ndarray(images_rgb)]
        images_mask = [Image.fromarray(im[:, :, 0]) for im in tensor_to_ndarray(images_mask)]
        images_result = []
        for idx, (rgb, mask) in enumerate(zip(images_rgb, images_mask)):
            if self.base_model == 'sdxl':
                result = self.inpainting_pipeline(
                    prompt=prompt, 
                    image=rgb, 
                    mask_image=mask, 
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    strength=0.80,
                    height=1024,
                    width=1024,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            elif self.base_model == 'sdxl-inpainting':
                result = self.inpainting_pipeline(
                    prompt=prompt, 
                    image=rgb, 
                    mask_image=mask, 
                    num_inference_steps=20,  # steps between 15 and 30 work well for us
                    guidance_scale=8.0,
                    strength=0.99,  # make sure to use `strength` below 1.0
                    height=1024,
                    width=1024,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            elif self.base_model == 'flux':
                result = self.inpainting_pipeline(
                    prompt=prompt,
                    image=rgb,
                    mask_image=mask,
                    height=1024,
                    width=1024,
                    guidance_scale=7.0,
                    num_inference_steps=28,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            elif self.base_model == 'flux-inpainting':
                result = self.inpainting_pipeline(
                    prompt=prompt,
                    image=rgb,
                    mask_image=mask,
                    height=1024,  # 1632
                    width=1024,  # 1232
                    guidance_scale=30,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            else:
                raise NotImplementedError(f'base_model {self.base_model} is not supported')
            images_result.append(result)
        images_result = ndarray_to_tensor(np.stack([np.array(im) for im in images_result], axis=0))
        images_result = torch.cat([images_result.to(images), images[:, [3], :, :]], dim=1)
        return images_result


class ImageUpscalerModel:
    def __init__(self, base_model='flux-upscaler'):
        self.base_model = base_model
        if base_model == 'sdxl-upscaler':
            controlnet = ControlNetModel.from_pretrained(
                "pretrain_models/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.upscaler_pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                'pretrain_models/stabilityai/stable-diffusion-xl-base-1.0',
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safetensors=True,
            )
            self.upscaler_pipeline.to('cuda')
        elif base_model == 'flux-upscaler':
            controlnet = FluxControlNetModel.from_pretrained(
                "pretrain_models/jasperai/Flux.1-dev-Controlnet-Upscaler",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = FluxTransformer2DModel.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-dev",
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16
            )
            self.upscaler_pipeline = FluxControlNetPipeline.from_pretrained(
                "pretrain_models/black-forest-labs/FLUX.1-dev",
                controlnet=controlnet,
                transformer=model_nf4,
                torch_dtype=torch.bfloat16
            )
            self.upscaler_pipeline.enable_model_cpu_offload()
        else:
            raise NotImplementedError(f'base_model {base_model} is not supported')

    def __call__(self, images):
        prompt = ''
        images_rgb = images[:, :3, :, :]
        images_rgb = [Image.fromarray(im) for im in tensor_to_ndarray(images_rgb)]
        images_result = []
        for idx, rgb in enumerate(images_rgb):
            if self.base_model == 'sdxl-upscaler':
                result = self.upscaler_pipeline(
                    prompt=prompt,
                    negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                    image=rgb,
                    control_image=rgb,
                    width=1024,
                    height=1024,
                    strength=0.3,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=0.9,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            elif self.base_model == 'flux-upscaler':
                result = self.upscaler_pipeline(
                    prompt=prompt, 
                    control_image=rgb,
                    controlnet_conditioning_scale=0.6,
                    num_inference_steps=28, 
                    guidance_scale=3.5,
                    height=1024,
                    width=1024,
                    generator=torch.Generator("cpu").manual_seed(666),
                ).images[0]
            else:
                raise NotImplementedError(f'base_model {self.base_model} is not supported')
            images_result.append(result.resize((2048, 2048)))
        images_result = ndarray_to_tensor(np.stack([np.array(im) for im in images_result], axis=0))
        images_result = torch.cat([images_result.to(images), images[:, [3], :, :]], dim=1)
        return images_result


class FastTexture:
    def __init__(self, inpainting=False, upscaler=True) -> None:
        self.mesh_renderer = NVDiffRendererBase(device='cuda')
        self.c2ws = None
        self.w2cs = None
        self.intrinsics = None
        self.projections = None
        self.perspective = None
        self.images = None
        self.map_Kd = None
        self.map_Kd_mask = None
        self.v_pos = None
        self.t_v_pos = None
        self.v_tex = None
        self.t_v_tex = None
        if inpainting:
            self.image_inpainting_model = ImageInpaintingModel(base_model='sdxl')
        else:
            self.image_inpainting_model = None
        if upscaler:
            self.image_upscaler_model = ImageUpscalerModel(base_model='flux-upscaler')
        else:
            self.image_upscaler_model = None

    def set_c2ws(
        self, 
        n_views=8,
        radius=2.8,
        height=0.0,
        theta_0=0.0,
        degree=True,
        seed=22,
        method='orbit',
        transforms:Optional[torch.Tensor]=None,
    ):
        if method == 'orbit':
            c2ws = generate_orbit_views_c2ws(n_views + 1, radius=radius, height=height, theta_0=theta_0, degree=degree)[:n_views]
        elif method == 'sphere':
            c2ws = generate_semisphere_views_c2ws(n_views, radius=radius, seed=seed)
        else:
            raise NotImplementedError(f'method {method} is not supported')
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        if transforms is not None:
            c2ws = torch.matmul(transforms.to(c2ws), c2ws)
        w2cs = c2w_to_w2c(c2ws)
        self.c2ws = c2ws
        self.w2cs = w2cs

    def add_c2ws(
        self, 
        n_views=8,
        radius=2.8,
        height=0.0,
        theta_0=0.0,
        degree=True,
        seed=22,
        method='orbit',
        transforms:Optional[torch.Tensor]=None,
    ):
        if method == 'orbit':
            c2ws = generate_orbit_views_c2ws(n_views + 1, radius=radius, height=height, theta_0=theta_0, degree=degree)[:n_views]
        elif method == 'sphere':
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            c2ws = generate_semisphere_views_c2ws(n_views, radius=radius, generator=generator)
        else:
            raise NotImplementedError(f'method {method} is not supported')
        c2ws = c2ws.to(dtype=torch.float32, device='cuda')
        if transforms is not None:
            c2ws = torch.matmul(transforms.to(c2ws), c2ws)
        w2cs = c2w_to_w2c(c2ws)
        if self.c2ws is not None:
            self.c2ws = torch.cat([self.c2ws, c2ws])
            self.w2cs = torch.cat([self.w2cs, w2cs])
        else:
            self.c2ws = c2ws
            self.w2cs = w2cs

    def set_intrinsics(
        self, 
        f_x=49.1,
        f_y=49.1,
        fov=True,
        degree=True,
    ):
        intrinsics = generate_intrinsics(f_x=f_x, f_y=f_y, fov=fov, degree=degree)
        self.intrinsics = intrinsics.to(dtype=torch.float32, device='cuda')
        self.projections = intr_to_proj(self.intrinsics, perspective=fov)
        self.perspective = fov
        if fov:
            self.mesh_renderer.enable_perspective()
        else:
            self.mesh_renderer.enable_orthogonal()

    def set_images(self, images:Union[torch.Tensor, np.ndarray, Iterable[Image.Image]]):
        if isinstance(images, torch.Tensor):
            images = images
        elif isinstance(images, np.ndarray):
            images = ndarray_to_tensor(images)
        elif isinstance(images, Iterable):
            images = ndarray_to_tensor(np.stack([np.array(im) for im in images], axis=0))
        else:
            raise NotImplementedError
        B, C, H, W = images.shape
        self.images = images.to(dtype=torch.float32, device='cuda')

    def set_map_Kd(self, map_Kd:Union[torch.Tensor, np.ndarray, Image.Image]):
        if isinstance(map_Kd, torch.Tensor):
            map_Kd = map_Kd
        elif isinstance(map_Kd, np.ndarray):
            map_Kd = ndarray_to_tensor(map_Kd)
        elif isinstance(map_Kd, Image.Image):
            map_Kd = ndarray_to_tensor(np.array(map_Kd))
        else:
            raise NotImplementedError
        C, H, W = map_Kd.shape
        self.map_Kd = map_Kd.to(dtype=torch.float32, device='cuda')
    
    def set_map_Kd_mask(self, map_Kd_mask:Union[torch.Tensor, np.ndarray, Image.Image]):
        if isinstance(map_Kd_mask, torch.Tensor):
            map_Kd_mask = map_Kd_mask
        elif isinstance(map_Kd_mask, np.ndarray):
            map_Kd_mask = ndarray_to_tensor(map_Kd_mask)
        elif isinstance(map_Kd_mask, Image.Image):
            map_Kd_mask = ndarray_to_tensor(np.array(map_Kd_mask))
        else:
            raise NotImplementedError
        C, H, W = map_Kd_mask.shape
        self.map_Kd_mask = map_Kd_mask[[0], :, :].to(dtype=torch.bool, device='cuda')

    def _set_mesh(self, v_pos:torch.Tensor, t_v_pos:torch.Tensor, v_tex:torch.Tensor, t_v_tex:torch.Tensor):
        V_pos, _ = v_pos.shape
        F_pos, _ = t_v_pos.shape
        V_tex, _ = v_tex.shape
        F_tex, _ = t_v_tex.shape
        assert F_pos == F_tex
        self.v_pos = v_pos.to(dtype=torch.float32, device='cuda')
        self.t_v_pos = t_v_pos.to(dtype=torch.int64, device='cuda')
        self.v_tex = v_tex.to(dtype=torch.float32, device='cuda')
        self.t_v_tex = t_v_tex.to(dtype=torch.int64, device='cuda')

    def set_mesh(self, mesh:Mesh):
        self._set_mesh(
            mesh.v_pos,
            mesh.t_pos_idx,
            mesh.v_tex,
            mesh.t_tex_idx,
        )

    def export_mesh(self):
        return Mesh(
            self.v_pos, 
            self.t_v_pos, 
            self.v_tex, 
            self.t_v_tex,
        )

    def set_texture(self, texture:Texture):
        self.set_mesh(texture.mesh)
        self.set_map_Kd(texture.map_Kd.permute(2, 0, 1))

    def export_texture(self):
        return Texture(
            self.export_mesh(),
            None,
            self.map_Kd.permute(1, 2, 0),
            None,
            None,
        )

    @CPUTimer('image_to_uv_v1')
    def image_to_uv_v1(self, texture_size:Union[int, Tuple[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        map_Kd: [C, H, W]
        map_Kd_alpha: [1, H, W]
        
        NOTE: blending map_Kd layers is not correct, but create Delaunay mesh for 2048*2048 2d points 
            via q-hull or s-hull is extremely slow!
        '''
        mesh = Mesh(v_pos=self.v_pos, t_pos_idx=self.t_v_pos, v_tex=self.v_tex, t_tex_idx=self.t_v_tex)
        ret = self.mesh_renderer.simple_inverse_rendering(
            mesh, None, self.images.permute(0, 2, 3, 1), None,
            self.c2ws, self.intrinsics, texture_size, 
            render_uv=True, render_map_attr=True,
            grid_interpolate_mode='bilinear', enable_antialis=False,
        )
        map_Kd = ret['map_attr'].permute(0, 3, 1, 2)
        map_Kd_alpha = ret['alpha'].unsqueeze(0).repeat_interleave(map_Kd.shape[0], dim=0).permute(0, 3, 1, 2)
        map_Kd_alpha_visiable = ret['uv_alpha'].permute(0, 3, 1, 2)
        map_Kd = (map_Kd * map_Kd_alpha_visiable).sum(0)
        map_Kd_alpha_visiable = map_Kd_alpha_visiable.sum(0)
        map_Kd = torch.where(map_Kd_alpha_visiable > 0, map_Kd / map_Kd_alpha_visiable, 0.0)
        map_Kd_mask = (map_Kd_alpha_visiable > 0)
        map_Kd_mask = map_Kd_mask.logical_or((map_Kd_alpha > 0).logical_not())
        return map_Kd, map_Kd_alpha, map_Kd_mask

    @CPUTimer('image_to_uv_v2')
    def image_to_uv_v2(self, texture_size:Union[int, Tuple[int]]):
        '''
        map_Kd: [C, H, W]
        map_Kd_alpha: [1, H, W]
        '''
        B, C, H, W = self.images.shape
        render_size = (H, W)
        mesh = Mesh(v_pos=self.v_pos, t_pos_idx=self.t_v_pos, v_tex=self.v_tex, t_tex_idx=self.t_v_tex)
        if self.map_Kd is not None and self.map_Kd_mask is not None:
            ret = self.mesh_renderer.global_inverse_rendering(
                mesh, self.images.permute(0, 2, 3, 1),
                self.c2ws, self.intrinsics, render_size, texture_size,
                texture_attr=self.map_Kd.permute(1, 2, 0), texture_mask=self.map_Kd_mask.permute(1, 2, 0),
            )
        else:
            ret = self.mesh_renderer.global_inverse_rendering(
                mesh, self.images.permute(0, 2, 3, 1),
                self.c2ws, self.intrinsics, render_size, texture_size,
            )
        map_Kd = ret['map_attr'].permute(2, 0, 1)
        map_Kd_alpha = ret['map_alpha'].permute(2, 0, 1)
        map_Kd_mask = ret['map_mask'].permute(2, 0, 1)
        map_Kd_mask = map_Kd_mask.logical_or((map_Kd_alpha > 0).logical_not())
        return map_Kd, map_Kd_alpha, map_Kd_mask

    @CPUTimer('uv_to_image')
    def uv_to_image(self, image_size:Union[int, Tuple[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        images: [N, C, H, W]
        images_alpha: [N, 1, H, W]
        '''
        mesh = Mesh(v_pos=self.v_pos, t_pos_idx=self.t_v_pos, v_tex=self.v_tex, t_tex_idx=self.t_v_tex)
        ret = self.mesh_renderer.simple_rendering(
            mesh, None, self.map_Kd.permute(1, 2, 0), None,
            self.c2ws, self.intrinsics, image_size, 
            render_uv=True, render_map_attr=True,
            grid_interpolate_mode='bilinear', enable_antialis=False,
        )
        images = ret['map_attr'].permute(0, 3, 1, 2)
        images_alpha = ret['alpha'].permute(0, 3, 1, 2)
        if self.map_Kd_mask is None:
            images_mask = None
        else:
            ret = self.mesh_renderer.simple_rendering(
                mesh, None, self.map_Kd_mask.to(dtype=torch.float32).permute(1, 2, 0), None,
                self.c2ws, self.intrinsics, image_size, 
                render_uv=True, render_map_attr=True,
                grid_interpolate_mode='nearest', enable_antialis=False,
            )
            images_mask = ret['map_attr'].permute(0, 3, 1, 2)
            images_mask = images_mask.to(dtype=torch.bool)
        return images, images_alpha, images_mask

    def compute_map_Kd_mask(self):
        C, H, W = self.map_Kd.shape
        mesh = Mesh(v_pos=self.v_pos, t_pos_idx=self.t_v_pos, v_tex=self.v_tex, t_tex_idx=self.t_v_tex)
        ret = self.mesh_renderer.simple_inverse_rendering(
            mesh, None, None, None,
            None, None, (H, W),
            enable_antialis=False,
        )
        map_Kd_mask = ret['mask'][0].permute(2, 0, 1).to(dtype=torch.float32)
        return map_Kd_mask

    def outpaint_map_Kd(self):
        C, H, W = self.map_Kd.shape
        map_Kd = self.map_Kd.unsqueeze(0)
        map_Kd, map_Kd_mask = pull_push(map_Kd[:, :-1, :, :], map_Kd[:, [-1], :, :] > 0.0)
        self.map_Kd = torch.cat([map_Kd, map_Kd_mask.to(dtype=map_Kd.dtype)], dim=1).squeeze(0)

    def outpaint_map_Kd_mask(self):
        if self.map_Kd_mask is not None:
            C, H, W = self.map_Kd_mask.shape
            map_Kd_mask = self.compute_map_Kd_mask()
            map_Kd = self.map_Kd_mask.to(dtype=map_Kd.dtype).unsqueeze(0)
            map_Kd, _ = pull_push(map_Kd[:, :, :, :], map_Kd_mask)
            self.map_Kd_mask = (map_Kd > 0.0).squeeze(0)

    def outpaint_images(self):
        B, C, H, W = self.images.shape
        map_Kd = self.map_Kd.unsqueeze(0)
        images, images_mask = pull_push(self.images[:, :-1, :, :], self.images[:, [-1], :, :] > 0.0)
        self.images = torch.cat([images, images_mask.to(dtype=map_Kd.dtype)], dim=1)

    def inpaint_images(self, images, images_mask):
        return self.image_inpainting_model(images, images_mask)

    def upscale_map_Kd(self, map_Kd):
        return self.image_upscaler_model(map_Kd.unsqueeze(0)).squeeze(0)

def test():
    # Pear_basket_SF, chespin
    mesh_path = 'gradio_examples_mesh/0941f2938b3147c7bfe16ec49d3ac500/raw_mesh.glb'
    n_views = 4
    n_rows, n_cols = 1, n_views
    
    texture = Texture.from_trimesh(load_whole_mesh(mesh_path))
    texture.to(device='cuda')
    mesh = texture.mesh
    mesh.scale_to_bbox().apply_transform()

    fast_texture = FastTexture()
    fast_texture.set_mesh(mesh)
    fast_texture.set_intrinsics(49.1, 49.1, True, True)
    # fast_texture.set_intrinsics(0.85, 0.85, False, False)

    #### stage 1 ####
    fast_texture.set_c2ws(n_views=n_views, radius=2.8, height=0.0, method='orbit')
    # fast_texture.set_c2ws(n_views=n_views, radius=2.8, height=2.8 * math.cos(math.radians(60)), method='orbit')
    # fast_texture.set_c2ws(n_views=n_views, radius=2.8, seed=22, method='sphere')
    
    fast_texture.set_map_Kd(texture.map_Kd.permute(2, 0, 1))
    images, images_alpha, images_mask = fast_texture.uv_to_image(512)
    Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images.png')
    images_1 = images[:, :3, :, :].clone()  # NOTE: copy for validation
    images_1_mask = (images_alpha > 0).clone()  # NOTE: copy for validation

    fast_texture.set_images(images)
    map_Kd, map_Kd_alpha, map_Kd_mask = fast_texture.image_to_uv_v2(2048)
    Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))[..., :4]).save('__debug_map_Kd_re.png')
    Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd_mask[[0,0,0], :, :].to(dtype=torch.float32), map_Kd_alpha], dim=0))[..., :4]).save('__debug_map_Kd_re_mask.png')

    fast_texture.set_map_Kd(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))
    fast_texture.set_map_Kd_mask(map_Kd_mask)
    fast_texture.outpaint_map_Kd()
    images, images_alpha, images_mask = fast_texture.uv_to_image(512)
    Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images_re.png')
    Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images_mask[:, [0,0,0], :, :].to(dtype=torch.float32), images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images_re_mask.png')
    images_2 = images[:, :3, :, :].clone()  # NOTE: copy for validation

    texture = fast_texture.export_texture()
    export_video(texture, '__debug_mesh_video.mp4')
    texture.reset_map_Kd_mask()
    texture.outpaint_map_Kd()
    texture.export('__debug_mesh.glb')

    mse = torch.masked_select(images_1 - images_2, images_1_mask).square().mean()
    psnr = 10 * torch.log10(1 / mse)
    print('n_views', n_views, 'mse', mse.item(), 'psnr', psnr.item())

    #### stage 2 ####
    n_iters_inpainting = 0
    n_views_inpainting = 1
    n_rows_inpainting, n_cols_inpainting = 1, n_views_inpainting
    theta_0_list = np.linspace(0, 360, n_iters_inpainting+1, endpoint=True)[:n_iters_inpainting]
    for iter in range(n_iters_inpainting):
        theta_0 = theta_0_list[iter]
        fast_texture.set_c2ws(n_views=n_views_inpainting, radius=2.8, height=2.8 * math.cos(math.radians(60)), theta_0=theta_0, method='orbit')
        images, images_alpha, images_mask = fast_texture.uv_to_image(1024)
        images_mask_inv = torch.logical_and(dilate_erode(~images_mask), images_alpha)
        Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows_inpainting, n_cols=n_cols_inpainting)[..., :4]).save(f'__debug_images_re_init_{iter:04d}.png')
        Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images_mask_inv[:, [0,0,0], :, :].to(dtype=torch.float32), images_alpha], dim=1)), n_rows=n_rows_inpainting, n_cols=n_cols_inpainting)[..., :4]).save(f'__debug_images_re_init_mask_{iter:04d}.png')

        images = fast_texture.inpaint_images(torch.cat([images[:, :3, :, :], images_alpha], dim=1), images_mask_inv)
        Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows_inpainting, n_cols=n_cols_inpainting)[..., :4]).save(f'__debug_images_re_inpaint_{iter:04d}.png')

        fast_texture.set_images(images)
        map_Kd, map_Kd_alpha, map_Kd_mask = fast_texture.image_to_uv_v2(2048)
        Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))[..., :4]).save(f'__debug_map_Kd_re_inpaint_re_{iter:04d}.png')
        Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd_mask[[0,0,0], :, :].to(dtype=torch.float32), map_Kd_alpha], dim=0))[..., :4]).save(f'__debug_map_Kd_re_inpaint_re_mask_{iter:04d}.png')

        fast_texture.set_map_Kd(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))
        fast_texture.set_map_Kd_mask(map_Kd_mask)
        fast_texture.outpaint_map_Kd()
    if n_iters_inpainting > 0:
        fast_texture.set_c2ws(n_views=n_views, radius=2.8, height=0.0, method='orbit')
        images, images_alpha, images_mask = fast_texture.uv_to_image(512)
        Image.fromarray(tensor_to_ndarray(torch.cat([fast_texture.map_Kd_mask, map_Kd_alpha], dim=0))[..., :4]).save('__debug_map_Kd_re_inpaint_re_mask_outpainting.png')
        Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images_re_inpaint_re.png')
        Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images_mask[:, [0,0,0], :, :].to(dtype=torch.float32), images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images_re_inpaint_re_mask.png')

    #### stage 3 ####
    map_Kd = fast_texture.map_Kd
    Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))[..., :4]).save('__debug_before_upscaler.png')
    map_Kd = fast_texture.upscale_map_Kd(fast_texture.map_Kd)
    Image.fromarray(tensor_to_ndarray(torch.cat([map_Kd[:3, :, :], map_Kd_alpha], dim=0))[..., :4]).save('__debug_after_upscaler.png')
    fast_texture.set_map_Kd(map_Kd)
    fast_texture.outpaint_map_Kd()

    fast_texture.set_c2ws(n_views=n_views, radius=2.8, height=0.0, method='orbit')
    images, images_alpha, images_mask = fast_texture.uv_to_image(512)
    Image.fromarray(make_grid(tensor_to_ndarray(torch.cat([images[:, :3, :, :], images_alpha], dim=1)), n_rows=n_rows, n_cols=n_cols)[..., :4]).save('__debug_images_upscaler.png')

    texture = fast_texture.export_texture()
    export_video(texture, '__debug_mesh_video_inpaint.mp4')
    texture.reset_map_Kd_mask()
    texture.outpaint_map_Kd()
    texture.export('__debug_mesh_inpaint.glb')

    # mse = torch.masked_select(images_1 - images_3, images_1_mask).square().mean()
    # psnr = 10 * torch.log10(1 / mse)
    # print('n_views', n_views, 'mse', mse.item(), 'psnr', psnr.item())

