# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor, 
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, EXAMPLE_DOC_STRING
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux_prior_redux import FluxPriorReduxPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class PBRFluxPriorReduxPipelineOutput(BaseOutput):
    prompt_embeds: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None


class PBRFluxPriorReduxPipeline(FluxPriorReduxPipeline):
    @torch.no_grad()
    def __call__(self, image: PipelineImageInput, return_dict=True) -> PBRFluxPriorReduxPipelineOutput:
        '''
        image: 
            * Image or List[Image]
            * Tensor: [B, C, H, W], range(0.0, 1.0)
        prompt_embeds: [B, L, C]
        text_ids: [L, 3]
        '''
        if isinstance(image, torch.Tensor):
            H = W = self.image_encoder.config.image_size
            image = torch.nn.functional.interpolate(image, size=(H, W), mode='bicubic', antialias=True)
            image = image.mul(2.0).sub(1.0).to(dtype=self.image_encoder.dtype, device=self.image_encoder.device)
            prompt_embeds = self.image_encoder(image).last_hidden_state
        else:
            image = self.feature_extractor.preprocess(
                images=image, 
                do_resize=True, 
                return_tensors="pt", 
                do_convert_rgb=True
            )
            prompt_embeds = self.image_encoder(**image).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.image_embedder.dtype, device=self.image_embedder.device)
        prompt_embeds = self.image_embedder(prompt_embeds).image_embeds
        text_ids = torch.zeros((prompt_embeds.shape[1], 3), device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if not return_dict:
            return prompt_embeds, text_ids
        return PBRFluxPriorReduxPipelineOutput(prompt_embeds=prompt_embeds, text_ids=text_ids)


class PBRFluxPipelineOutput(BaseOutput):
    images: Optional[List[Image.Image]] = None
    albedo_images: Optional[List[Image.Image]] = None
    metallic_roughness_images: Optional[List[Image.Image]] = None
    bump_images: Optional[List[Image.Image]] = None


class PBRFluxPipeline(FluxPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        redux_pipeline: PBRFluxPriorReduxPipeline = None,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128
        self.redux_pipeline = redux_pipeline

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, pixel_shuffle=True):
        if pixel_shuffle:
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        else:
            latents = latents.permute(0, 2, 3, 1)
            latents = latents.reshape(batch_size, height * width, num_channels_latents)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype, offset_x=0, offset_y=0, offset_z=0):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(offset_y, offset_y + height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(offset_x, offset_x + width)[None, :]
        if offset_z != 0:
            latent_image_ids[..., 0] = latent_image_ids[..., 0] + offset_z
        latent_image_ids = latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)
        return latent_image_ids

    def prepare_latents_and_image_ids(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        dual_image=None,
        redux_image=None,
        control_image=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        HL = 2 * (int(height) // (self.vae_scale_factor * 2))
        WL = 2 * (int(width) // (self.vae_scale_factor * 2))
        noise_latents = randn_tensor((batch_size, num_channels_latents, HL, WL), generator=generator, device=device, dtype=dtype)
        noise_latents = self._pack_latents(
            noise_latents, 
            batch_size, 
            num_channels_latents, 
            HL, 
            WL,
            pixel_shuffle=True,
        )
        noise_ids = self._prepare_latent_image_ids(
            batch_size, 
            HL // 2, 
            WL // 2, 
            device, 
            dtype,
            offset_x=0,
            offset_y=0,
            offset_z=0,
        )

        if dual_image is not None:
            WD, HD = dual_image.size
            dual_image = self.image_processor.preprocess(dual_image, height=HD, width=WD)
            dual_image = dual_image.to(dtype=self.vae.dtype, device=device)
            dual_latents = self._encode_vae_image(image=dual_image, generator=generator)
            dual_latents = dual_latents.to(dtype=dtype)
            if dual_latents.shape[0] == 1 and batch_size > 1:
                dual_latents = dual_latents.repeat(batch_size, 1, 1, 1)
            BDL, CDL, HDL, WDL = dual_latents.shape
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            assert HDL == 2 * (HD // (self.vae_scale_factor * 2)) and \
                WDL == 2 * (WD // (self.vae_scale_factor * 2))
            dual_latents = self._pack_latents(
                dual_latents, 
                batch_size, 
                dual_latents.shape[1], 
                HDL, 
                WDL, 
                pixel_shuffle=True, 
            )
            dual_ids = self._prepare_latent_image_ids(
                batch_size, 
                HDL // 2, 
                WDL // 2, 
                device,
                dtype,
                offset_x=WL // 2,
                offset_y=HL // 2,
                offset_z=0,
            )
        else:
            dual_latents = None
            dual_ids = None

        if redux_image is not None:
            if self.redux_pipeline is not None:
                redux_out = self.redux_pipeline(redux_image)
                redux_embeds = redux_out.redux_image_embeds
                if redux_embeds.shape[0] == 1 and batch_size > 1:
                    redux_embeds = redux_embeds.repeat(batch_size, 1, 1)
                redux_ids = redux_out.redux_image_ids
            else:
                redux_embeds = None
                redux_ids = None
        else:
            redux_embeds = None
            redux_ids = None

        if control_image is not None:
            WC, HC = control_image.size
            control_image = self.image_processor.preprocess(control_image, height=HC, width=WC)
            control_image = control_image.to(dtype=self.vae.dtype, device=device)
            control_latents = self._encode_vae_image(image=control_image, generator=generator)
            control_latents = control_latents.to(dtype=dtype)
            if control_latents.shape[0] == 1 and batch_size > 1:
                control_latents = control_latents.repeat(batch_size, 1, 1, 1)
            BCL, CCL, HCL, WCL = control_latents.shape
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            assert HCL == 2 * (HC // (self.vae_scale_factor * 2)) and \
                WCL == 2 * (WC // (self.vae_scale_factor * 2))
            control_latents = self._pack_latents(
                control_latents, 
                batch_size, 
                control_latents.shape[1], 
                HCL, 
                WCL, 
                pixel_shuffle=True, 
            )
            control_ids = self._prepare_latent_image_ids(
                batch_size, 
                HCL // 2, 
                WCL // 2, 
                device,
                dtype,
                offset_x=0,
                offset_y=HL // 2,
                offset_z=0,
            )
        else:
            control_latents = None
            control_ids = None
        return (
            noise_latents, noise_ids,
            dual_latents, dual_ids,
            redux_embeds, redux_ids,
            control_latents, control_ids,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        dual_image: Optional[Image.Image] = None,
        redux_image: Optional[Image.Image] = None,
        control_image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        n_rows = n_rows or 2
        n_cols = n_cols or 2

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # NOTE: if hf hooks are not registered, default _execution_device is cpu
        device = self._execution_device

        # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        if self.text_encoder is None:
            # clip text encoder hidden size is 768
            pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=torch.bfloat16)
        if self.text_encoder_2 is None:
            # t5 encoder hidden size is 4096, max_sequence_length is 512
            prompt_embeds = torch.zeros((batch_size, max_sequence_length, 4096), device=device, dtype=torch.bfloat16)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        (
            noise_latents, noise_ids,
            dual_latents, dual_ids,
            redux_embeds, redux_ids,
            control_latents, control_ids,
        ) = self.prepare_latents_and_image_ids(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            dual_image=dual_image,
            redux_image=redux_image,
            control_image=control_image,
        )
        latents = noise_latents
        latent_image_ids = noise_ids
        if dual_latents is not None and control_latents is not None:
            condition_latents = torch.cat([control_latents, dual_latents], dim=1)
            condition_ids = torch.cat([control_ids, dual_ids], dim=0)
        elif dual_latents is not None and control_latents is None:
            condition_latents = dual_latents
            condition_ids = dual_ids
        elif dual_latents is None and control_latents is not None:
            condition_latents = control_latents
            condition_ids = control_ids
        else:
            condition_latents = None
            condition_ids = None

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        is_transformer_compiled = is_compiled_module(self.transformer)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        if redux_embeds is not None:
            prompt_image_embeds = redux_embeds
            text_image_ids = redux_ids
        else:
            prompt_image_embeds = prompt_embeds
            text_image_ids = text_ids
        if condition_latents is not None:
            latents = torch.cat([latents, condition_latents], dim=1)
            latent_image_ids = torch.cat([latent_image_ids, condition_ids], dim=0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Relevant thread: https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if is_transformer_compiled and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if condition_latents is not None:
                    latents = torch.cat([latents[:, :-condition_latents.shape[1]], condition_latents], dim=1)
                noise_pred = self.transformer(
                    hidden_states=latents,  # vae: [1, 8192, 64], change dim to 3072
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,  # clip: [1, 768]
                    encoder_hidden_states=prompt_image_embeds,  # t5: [1, 512, 4096], redux: [1, 729, 4096], change dim to 3072
                    txt_ids=text_image_ids,  # t5 pe: [512, 3], redux: [729, 3]
                    img_ids=latent_image_ids,  # vae pe: [8192, 3]
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if condition_latents is not None:
            latents = latents[:, :-condition_latents.shape[1]]
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(dtype=self.vae.dtype, device=self.vae.device)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PBRFluxPipelineOutput(images=image)