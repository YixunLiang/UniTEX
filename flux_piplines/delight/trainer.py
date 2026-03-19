#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# Copied from diffusers/examples/dreambooth/train_dreambooth_lora_flux.py
import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"adding path: {os.path.dirname(os.path.abspath(__file__))}")
import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.utils import save_image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers import FluxTransformer2DModel  
# from pipeline_flux_fill_mv import PBRFluxPipeline
from transformers import CLIPTextModel
from transformers import T5EncoderModel

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from safetensors.torch import save_file, load_file
from pipeline import PBRFluxPipeline, PBRFluxPriorReduxPipeline
from attention_processor import NativeFluxAttnProcessor2_0, RandomDropFluxAttnProcessor2_0

def get_mha_processor(args, device='cuda'):
    text_tokens = 512
    if args.six_views_or_four_views:
        condition_token_num = (1+6)*32*32
    else:
        condition_token_num = (1+4)*32*32
    drop_rate = args.attn_random_drop_noise_probability
    mha_native_processor = NativeFluxAttnProcessor2_0()
    if not args.attn_random_drop_noise:
        double_random_drop_processor = NativeFluxAttnProcessor2_0()
        single_random_drop_processor = NativeFluxAttnProcessor2_0()
    else:
        double_random_drop_processor = RandomDropFluxAttnProcessor2_0(
            drop_rate = drop_rate,
            text_token_num = text_tokens,
            condition_token_num = condition_token_num
        )
        single_random_drop_processor = RandomDropFluxAttnProcessor2_0(
            drop_rate = drop_rate,
            text_token_num = text_tokens,
            condition_token_num = condition_token_num
        )
    return mha_native_processor, double_random_drop_processor, single_random_drop_processor


class PBRTrainer(PBRFluxPipeline):
    def __init__(self, 
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        args,
        accelerator, 
        logger,
        dataloader):
        super().__init__(scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer)
        self.args = args
        # set parameters in all models as false
        self.set_requires_grad()
        # set all parameters as half precision
        self.set_weight_type_and_device(accelerator)

        # parameters in adaptor will be type of float32, 
        # requires_grad will be True
        self.add_LORA(accelerator)
        ## enable dataloader
        self.train_dataloader = dataloader
        self.set_optimizer(logger)
        self.set_lr_schedular(accelerator)
        self.prepare_training(accelerator, logger)


    @staticmethod
    def from_args(args, 
        accelerator, 
        logger,
        train_dataloader):
        # Load the tokenizers
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )

        # Load scheduler and models
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # load the text_encoder in flux, "CLIPTextModel" and "T5EncoderModel"
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=args.revision, 
            variant=args.variant,
            quantization_config=None,
            torch_dtype=torch.bfloat16,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder_2", 
            revision=args.revision, 
            variant=args.variant,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )


        ## add random drop technique for diffusion training acceleration![paper novelty!]
        # NOTE: hardcode here
        (
            mha_native_processor, 
            double_random_drop_processor, 
            single_random_drop_processor, 
        ) = get_mha_processor(args=args, device=accelerator.device)
        mha_dict = dict()
        for k, m in transformer.attn_processors.items():
            n = k.split('.')
            if 'transformer_blocks' in n:
                mha_dict[k] = double_random_drop_processor
            elif 'single_transformer_blocks' in n:
                mha_dict[k] = single_random_drop_processor
            else:
                mha_dict[k] = mha_native_processor
        transformer.set_attn_processor(mha_dict)

        return PBRTrainer(scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer, args, accelerator, logger, train_dataloader)

    def set_requires_grad(self):
        # We only train the additional adapter LoRA layers and disable follows:
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)


    
    def set_weight_type_and_device(self, accelerator):
        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if torch.backends.mps.is_available() and self.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.vae.to(accelerator.device, dtype=self.weight_dtype)
        self.transformer.to(accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(accelerator.device)


    def prepare_training(self, accelerator, logger):
        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        if accelerator.is_main_process:
            PBRTrainer.show_parameters(self.transformer, 'transformer')
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

        if self.args.gradient_checkpointing:  # enable gradient caching
            self.transformer.enable_gradient_checkpointing()

        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * accelerator.num_processes
            )

        # Prepare everything with our `accelerator`.
        # NOTE: deepspeed will break if initialize more than one model
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)


    def add_LORA(self, accelerator):
        if self.args.lora_layers is not None:  # default as None
            target_modules = [layer.strip() for layer in self.args.lora_layers.split(",")]
            if 'x_embedder' not in target_modules:
                target_modules.append('x_embedder')
        else:
            target_modules = [  # train LoRA for these layers
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
            #2 行 1个打开x_embedder 1个注视diao
            modules_to_save = [
                "x_embedder", 
                'norm1.norm',
                'norm1_context.norm',
                'norm2',
                'norm2_context',
                'norm.norm'
            ]
            
            # NOTE: https://github.com/huggingface/diffusers/pull/10130
            # examples/control-lora/train_control_lora_flux.py#L832

            # now we will add new LoRA weights the transformer layers
            transformer_lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                modules_to_save = modules_to_save
            )
            # NOTE: add_adapter changes requires_grad of modules except loras to False
            self.transformer.add_adapter(transformer_lora_config)

        self.load_LoRA_from_checkpoint(accelerator)

    def set_optimizer(self, logger):
        # parameters to optimize
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        # learn_param_name_ls = []
        # for iii in self.transformer.named_parameters():
        #     if iii[1].requires_grad:
        #         learn_param_name_ls.append(iii[0])
        # self.learn_param_name_ls = learn_param_name_ls
        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters, 
            "lr": self.args.learning_rate,
        }

        # Optimization parameters
        params_to_optimize = [transformer_parameters_with_lr]
        
        # Optimizer creation
        if self.args.use_8bit_adam and not self.args.optimizer.lower() == "adamw":
            logger.warning(
                f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                f"set to {self.args.optimizer.lower()}"
            )

        if self.args.optimizer.lower() == "adamw":
            if self.args.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                    )

                optimizer_class = bnb.optim.AdamW8bit
            else:
                optimizer_class = torch.optim.AdamW

            self.optimizer = optimizer_class(
                params_to_optimize,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )

        elif self.args.optimizer.lower() == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

            optimizer_class = prodigyopt.Prodigy

            if self.args.learning_rate <= 0.1:
                logger.warning(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )

            self.optimizer = optimizer_class(
                params=params_to_optimize,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                beta3=self.args.prodigy_beta3,
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
                decouple=self.args.prodigy_decouple,
                use_bias_correction=self.args.prodigy_use_bias_correction,
                safeguard_warmup=self.args.prodigy_safeguard_warmup,
            )

            return self.optimizer
        
        else:
            raise NotImplementedError(f"optimizer not implemented: {self.args.optimizer.lower()}")
        
        return 

    def set_lr_schedular(self, accelerator):
        # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        return self.lr_scheduler

    def get_sigmas(self, timesteps, n_dim=4, device='cuda:0', dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def load_LoRA_from_checkpoint(self, accelerator):
        # Potentially load in the weights and states from a previous save
        global_step = 0
        first_epoch = 0

        if self.args.resume_from_checkpoint is not None:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                # breakpoint()
                lora_state_dict = PBRFluxPipeline.lora_state_dict(os.path.join(self.args.output_dir, path))
                # learn_param_name_ls = self.learn_param_name_ls
                transformer_state_dict = {
                    f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
                }

                
                incompatible_keys = set_peft_model_state_dict(self.transformer, transformer_state_dict, adapter_name="default")  

                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )

                # Make sure the trainable params are in float32. This is again needed since the base models
                # are in `weight_dtype`. More details:
                # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
                models = [self.transformer]
                # only upcast trainable parameters (LoRA) into fp32, in case the state_dict is not full precision
                cast_training_params(models, dtype=torch.float32)
                global_step = int(path.split("-")[1])

                initial_global_step = global_step

        else:
            initial_global_step = 0

        self.initial_global_step = initial_global_step
        self.first_epoch = first_epoch

        return initial_global_step, first_epoch

    def save_lora_layers(self, accelerator, save_path):
        os.makedirs(save_path, exist_ok=True)

        transformer_lora_layers = get_peft_model_state_dict(self.unwrap_model(self.transformer, accelerator))

        if transformer_lora_layers is not None:
            PBRFluxPipeline.save_lora_weights(
                save_directory=save_path,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=None,
            )  

    def inference(self, steps, accelerator, logger, load_path=None, output_dir=None):
        # Final inference
        # Load previous pipeline
        if load_path is not None:
            pipeline = PBRFluxPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                revision=self.args.revision,
                variant=self.args.variant,
                torch_dtype=self.weight_dtype,
            )
            pipeline.load_lora_weights(self.args.output_dir)
        else:
            pipeline = PBRFluxPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        vae=self.vae,
                        text_encoder=accelerator.unwrap_model(self.text_encoder).eval(),
                        text_encoder_2=accelerator.unwrap_model(self.text_encoder_2).eval(),
                        transformer=accelerator.unwrap_model(self.transformer).eval(),
                        redux_pipeline=None,
                        revision=self.args.revision,
                        variant=self.args.variant,
                        torch_dtype=self.weight_dtype,
            )
        
        # run inference
        pipeline_args = {"prompt": self.args.validation_prompt}
        pipeline = pipeline.to(accelerator.device)
        images = self.log_validation(
                        pipeline=pipeline,
                        args=self.args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=steps,
                        torch_dtype=self.weight_dtype,
                        logger = logger
                    )
        del pipeline
        return images

    @staticmethod
    def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        epoch,
        torch_dtype,
        logger,
        is_final_validation=False,
    ):
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
        autocast_ctx = nullcontext()
        pipeline_args['guidance_scale'] = 3.5  # args.guidance_scale
        pipeline_args['height'], pipeline_args['width'] = args.resolution
        pipeline_args['n_rows'], pipeline_args['n_cols'] = args.n_rows, args.n_cols
        # if args.train_controlnet:
        #     pipeline_args['controlnet_blocks_repeat'] = args.controlnet_blocks_repeat
        #     pipeline_args['controlnet_single_blocks_repeat'] = args.controlnet_single_blocks_repeat

        if os.path.isfile(args.validation_prompt):
            with open(args.validation_prompt, 'r') as f:
                validation_prompt = json.load(f)
            eval_count = 0
            for validation_prompt_item in tqdm(validation_prompt, total=len(validation_prompt), desc='log_validation'):
                images = []
                if args.instance_prompt is None:
                    prompt = validation_prompt_item['prompt']
                    with open(prompt, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    pipeline_args["prompt"] = prompt
                else:
                    pipeline_args["prompt"] = args.instance_prompt
                if args.dual_image:
                    image = validation_prompt_item['image']
                    image = Image.open(image).convert('RGB')
                    pipeline_args["dual_image"] = image
                    images += [image]
                if args.control_image:
                    if args.both_ccm_normal_condition:
                        image1 = validation_prompt_item['control_image']['rgbs']
                        image2 = validation_prompt_item['control_image']['rgbs']
                        if isinstance(image1, str):
                            image = Image.fromarray((0.5*np.array(Image.open(image1).convert('RGB')) + 0.5*np.array(Image.open(image2).convert('RGB'))).astype(np.uint8))
                            images += [image]
                        else:
                            image = [Image.open(im).convert('RGB') for im in image]
                            images += [im for im in image]                        
                    else:
                        image = validation_prompt_item['control_image']['rgbs']
                        if isinstance(image, str):
                            image = Image.open(image).convert('RGB')
                            images += [image]
                        else:
                            image = [Image.open(im).convert('RGB') for im in image]
                            images += [im for im in image]
                    # NOTE: hardcode here
                    if args.six_views_or_four_views:
                        image = Image.fromarray(np.array(image).reshape(2, 512, 3, 512, -1).transpose(1, 0, 2, 3, 4).reshape(512, 2*3*512, -1))
                    else:
                        image = Image.fromarray(np.array(image).reshape(2, 512, 2, 512, -1).transpose(1, 0, 2, 3, 4).reshape(512, 2*2*512, -1))
                    pipeline_args["control_image"] = image
                if args.redux_image:
                    image = validation_prompt_item['image']
                    image = Image.open(image).convert('RGB')
                    pipeline_args["redux_image"] = image
                    images += [image]
                with autocast_ctx:
                    images += [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]
                if accelerator.is_main_process:
                    if not os.path.exists(args.output_dir+f'/eval_result_step_{epoch}'):
                        os.mkdir(args.output_dir+f'/eval_result_step_{epoch}')
                    if not os.path.exists(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}'):
                        os.mkdir(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}')
                    # images[0].save(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}'+f'/input_image.png')
                    images[0].save(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}'+f'/control_image.png')
                    images[1].save(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}'+f'/gen_image.png')
                eval_count += 1

        else:
            with autocast_ctx:
                images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]
                if not os.path.exists(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}'):
                    os.mkdir(args.output_dir+f'/eval_result_step_{epoch}/{eval_count}')
                # images[0].save(args.output_dir+f'/eval_result_step_{epoch}'+f'/input_image_{eval_count}.png')
                images[0].save(args.output_dir+f'/eval_result_step_{epoch}'+f'/control_image_{eval_count}.png')
                images[1].save(args.output_dir+f'/eval_result_step_{epoch}'+f'/gen_image_{eval_count}.png')
                eval_count += 1
        # NOTE: do validation in every rank to avoid nccl timeout(600s).
        # if accelerator.is_main_process:
        #     for tracker in accelerator.trackers:
        #         phase_name = "test" if is_final_validation else "validation"
        #         if tracker.name == "tensorboard":
        #             np_images = np.stack([np.asarray(img) for img in images])
        #             tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        #         if tracker.name == "wandb":
        #             tracker.log(
        #                 {
        #                     phase_name: [
        #                         wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
        #                     ]
        #                 }
        #             )

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return images


    def _encode_prompt_with_t5(
        self,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
    ):
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizer_2 is not None:
            text_inputs = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
        
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device))[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    @staticmethod
    def show_parameters(model: torch.nn.Module, name: str = None):
        print('#' * 8, f'{name if name is not None else model.__class__.__name__} parameters', '#' * 8)
        for i, (k, p) in enumerate(model.named_parameters()):
            if p.requires_grad:
                print('#' * 4, i, k)

    @torch.no_grad()
    def compute_text_embeddings(self, 
            prompt, 
            device,       
            weight_dtype, 
            num_images_per_prompt: int = 1,
            text_input_ids_list=None):
        
        with torch.no_grad():  

            prompt = [prompt] if isinstance(prompt, str) else prompt
            dtype = self.text_encoder.dtype

            pooled_prompt_embeds = self._encode_prompt_with_clip(
                prompt=prompt,
                device=device if device is not None else self.text_encoder.device,
                num_images_per_prompt=num_images_per_prompt,
                text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
            )

            prompt_embeds = self._encode_prompt_with_t5(
                max_sequence_length=self.args.max_sequence_length,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device if device is not None else self.text_encoder_2.device,
                text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
            )

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

            prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=weight_dtype)
            text_ids = text_ids.to(device, dtype=weight_dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
        
    @staticmethod
    def unwrap_model(model, accelerator):
        model = accelerator.unwrap_model(model) # correspond to prepare()
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    def _encode_vae_image():
        return 
    def train(self, accelerator, logger):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_name = "flux-dev-lora"
            accelerator.init_trackers(tracker_name, config=dict(filter(lambda x: isinstance(x[1], (int, float, str, bool)), vars(self.args).items())))

        total_batch_size = self.args.train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0
        # Potentially load in the weights and states from a previous save
        # global_step, first_epoch = self.load_LoRA_from_checkpoint(accelerator)
        global_step = self.initial_global_step
        first_epoch = global_step // self.num_update_steps_per_epoch

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        vae_config_shift_factor = self.vae.config.shift_factor
        vae_config_scaling_factor = self.vae.config.scaling_factor
        vae_config_block_out_channels = self.vae.config.block_out_channels
        vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
        ### inference at the begining:
        #images = self.inference(global_step, accelerator, output_dir=self.args.output_dir, logger = logger)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.transformer.train()

            for step, batch in enumerate(self.train_dataloader):
                models_to_accumulate = [self.transformer]

                with accelerator.accumulate(models_to_accumulate):
                    if self.args.random_prompt:
                        prompts = batch["uids"]
                    else:
                        prompts = batch["prompts"]
                    # 1. Time 90step
                    # print("keys of dataset: ", batch.keys())
                    bsz = batch["rgbs"].shape[0] 
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=self.args.logit_mean,
                        logit_std=self.args.logit_std,
                        mode_scale=self.args.mode_scale,
                    )
                    indices = (u * self.scheduler.config.num_train_timesteps).long()
                    timesteps = self.scheduler.timesteps[indices].to(device=accelerator.device)

                    # 2. Encode Prompt   
                    prompt_embeds, pooled_prompt_embeds, text_ids = self.compute_text_embeddings(
                        prompts, accelerator.device, self.weight_dtype
                    )

                    # 3. Prepare latents
 
                    # breakpoint()
                    pixel_values = batch["albedos"].to(device=accelerator.device)
                    #pixel_values = batch["normals"].to(device=accelerator.device)
                    # prepare input images
                    with torch.no_grad():
                        model_input = self.vae.encode(pixel_values.mul(2.0).sub(1.0).to(dtype=self.vae.dtype)).latent_dist.sample()
                        model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                        model_input = model_input.to(dtype=self.weight_dtype)
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input).to(accelerator.device)
                    batch_size = bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=self.args.logit_mean,
                        logit_std=self.args.logit_std,
                        mode_scale=self.args.mode_scale,
                    )
                    indices = (u * self.scheduler.config.num_train_timesteps).long()
                    timesteps = self.scheduler.timesteps[indices].to(device=accelerator.device)

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(device=accelerator.device)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    BL, CL, HL, WL = noisy_model_input.shape
                    latents = self._pack_latents(
                        noisy_model_input,
                        BL,
                        CL,
                        HL,
                        WL,
                    )
                    latent_image_ids = self._prepare_latent_image_ids(
                        BL,
                        HL // 2,
                        WL // 2,
                        accelerator.device,
                        self.weight_dtype,
                        offset_x=0,
                        offset_y=0,
                        offset_z=0,
                    )

                    # handle guidance
                    if accelerator.unwrap_model(self.transformer).config.guidance_embeds:
                        guidance = torch.tensor([self.args.guidance_scale], device=accelerator.device)
                        guidance = guidance.expand(latents.shape[0])
                    else:
                        guidance = None
                    # get dul images
                    if self.args.dual_image:
                        dual_image = batch["rgbs_ip"]
                        _, _, HD, WD = dual_image.shape
                        dual_image = dual_image.to(dtype=self.vae.dtype, device=accelerator.device)
                        with torch.no_grad():
                            dual_latents = self.vae.encode(dual_image.mul(2.0).sub(1.0)).latent_dist.sample()
                            dual_latents = (dual_latents - vae_config_shift_factor) * vae_config_scaling_factor
                            dual_latents = dual_latents.to(dtype=self.weight_dtype)
                        if dual_latents.shape[0] == 1 and batch_size > 1:
                            dual_latents = dual_latents.repeat(batch_size, 1, 1, 1)
                        BDL, CDL, HDL, WDL = dual_latents.shape
                        # VAE applies 8x compression on images but we must also account for packing which requires
                        # latent height and width to be divisible by 2.
                        assert HDL == 2 * (HD // (vae_scale_factor * 2)) and \
                            WDL == 2 * (WD // (vae_scale_factor * 2))
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
                            accelerator.device,
                            self.weight_dtype,
                            offset_x=WL // 2,
                            offset_y=HL // 2,
                            offset_z=0,
                        )
                    else:
                        dual_latents = None
                        dual_ids = None
                    # prepare redux image condition [NOT SUPPORT YET!]
                    redux_embeds = None
                    redux_ids = None
                    # prepare control images
                    if self.args.control_image:
                        control_image = batch["rgbs"]
                        _, _, HC, WC = control_image.shape
                        control_image = control_image.to(dtype=self.vae.dtype, device=accelerator.device)
                        with torch.no_grad():
                            control_latents = self.vae.encode(control_image.mul(2.0).sub(1.0)).latent_dist.sample()
                            control_latents = (control_latents - vae_config_shift_factor) * vae_config_scaling_factor
                            control_latents = control_latents.to(dtype=self.weight_dtype)
                        if control_latents.shape[0] == 1 and batch_size > 1:
                            control_latents = control_latents.repeat(batch_size, 1, 1, 1)
                        BCL, CCL, HCL, WCL = control_latents.shape
                        # VAE applies 8x compression on images but we must also account for packing which requires
                        # latent height and width to be divisible by 2.
                        assert HCL == 2 * (HC // (vae_scale_factor * 2)) and \
                            WCL == 2 * (WC // (vae_scale_factor * 2))
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
                            accelerator.device,
                            self.weight_dtype,
                            offset_x=0,
                            offset_y=HL // 2,
                            offset_z=0,
                        )
                    else:
                        control_latents = None
                        control_ids = None
                    # prepare condition components
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

                    if redux_embeds is not None:
                        prompt_image_embeds = redux_embeds
                        text_image_ids = redux_ids
                    else:
                        prompt_image_embeds = prompt_embeds
                        text_image_ids = text_ids
                    # using random drop tricks
                    if self.args.random_drop_noise and 0.0 <= self.args.random_drop_noise_probability <= 1.0:
                        if random.random() < 0.7:
                            random_drop_noise = True
                            mask_noise = (torch.rand(size=(latents.shape[1],), dtype=self.weight_dtype, device=accelerator.device) <= 1.0 - self.args.random_drop_noise_probability)
                            index_noise = torch.where(mask_noise)[0]
                            model_pred_zero = torch.zeros_like(latents)
                            latents = latents[:, index_noise, :]
                            latent_image_ids = latent_image_ids[index_noise, :]
                        else:
                            random_drop_noise = False
                    if condition_latents is not None:
                        if self.args.random_drop_condition and 0.0 <= self.args.random_drop_condition_probability <= 1.0:
                            if random.random() < 0.7:
                                random_drop_condition = True
                                mask_condition = (torch.rand(size=(condition_latents.shape[1],), dtype=self.weight_dtype, device=accelerator.device) <= 1.0 - self.args.random_drop_condition_probability)
                                index_condition = torch.where(mask_condition)[0]
                                condition_latents = condition_latents[:, index_condition, :]
                                condition_ids = condition_ids[index_condition, :]
                            else:
                                random_drop_condition = False
                        latents = torch.cat([latents, condition_latents], dim=1)
                        latent_image_ids = torch.cat([latent_image_ids, condition_ids], dim=0)
                    # 4. predict
                    model_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_image_embeds,
                    txt_ids=text_image_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    )[0]

                    if condition_latents is not None:
                        model_pred = model_pred[:, :-condition_latents.shape[1]]
                        
                    model_pred_mask = torch.ones_like(model_pred)
                    if self.args.random_drop_noise and 0.0 <= self.args.random_drop_noise_probability <= 1.0:
                        if random_drop_noise:
                            model_pred_zero = model_pred_zero.to(dtype=model_pred.dtype, device=model_pred.device)
                            index_noise = index_noise.unsqueeze(0).unsqueeze(-1).repeat(model_pred_zero.shape[0], 1, model_pred_zero.shape[-1])
                            model_pred = model_pred_zero.scatter(1, index_noise, model_pred)
                            model_pred_mask = model_pred_zero.scatter(1, index_noise, torch.ones_like(model_pred))
                    # model_pred = self._unpack_latents(
                    #     model_pred,
                    #     height=height,
                    #     width=width,
                    #     vae_scale_factor=vae_scale_factor,
                    # )
                    model_pred = self._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                    )

                    model_pred_mask = self._unpack_latents(
                    model_pred_mask,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                    )
                    # 5. flow matching loss
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.args.weighting_scheme, sigmas=sigmas)
                    # flow matching loss
                    target = noise - model_input


                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()

                    # Compute regular loss.
                    loss = torch.sum((weighting.float() * ((model_pred * model_pred_mask).float() - (target * model_pred_mask).float()) ** 2))/model_pred_mask.sum()

                    if self.args.with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss

                    if self.args.preconditioning_loss:
                        # compute the previous noisy sample x_t -> x_0
                        # (1) latents_pred = noise - model_pred
                        # (2) latents_pred = model_pred * (-sigmas) + noisy_model_input
                        latents_pred = model_pred * (-sigmas) + noisy_model_input
                        latents_pred = (latents_pred / vae_config_scaling_factor) + vae_config_shift_factor
                        pixel_values_pred = self.vae.decode(latents_pred.to(dtype=vae.dtype), return_dict=False)[0].mul(0.5).add(0.5)
                        preconditioning_loss = torch.mean((pixel_values_pred.float() - pixel_values.float()) ** 2)
                        loss = loss + args.preconditioning_loss_weight * preconditioning_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = [self.transformer.parameters()]
                        accelerator.clip_grad_norm_(itertools.chain(*params_to_clip), self.args.max_grad_norm)
                        del params_to_clip

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    # breakpoint()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            self.save_lora_layers(accelerator, save_path)

                            logger.info(f"Saved state to {save_path}")
                    if self.args.validation_prompt is not None and global_step % self.args.validation_steps == 0:
                        images = self.inference(global_step, accelerator, output_dir=self.args.output_dir, logger = logger)
                        self.transformer.train()
                    #     pipeline_args = {}
                    #     images, masks = self.log_validation(
                    #         pipeline=self,
                    #         args=self.args,
                    #         accelerator=accelerator,
                    #         pipeline_args=pipeline_args,
                    #         steps=global_step,
                    #         is_final_validation=True
                    #     )
                    # accelerator.wait_for_everyone()
                    # if accelerator.is_main_process:
                    #     breakpoint()
                    #     save_images_grids(images, masks, global_step, output_dir=self.args.output_dir)


                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break


        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.save_lora_layers(accelerator, self.args.output_dir)

        # Final inference
        if self.args.validation_config and self.args.num_validation_images > 0:
            images = self.inference(steps=global_step, accelerator=accelerator, output_dir=self.args.output_dir, logger = logger)
        #     pipeline_args = {}
        #     images, masks = self.log_validation(
        #         pipeline=self,
        #         args=self.args,
        #         accelerator=accelerator,
        #         pipeline_args=pipeline_args,
        #         steps=global_step,
        #         is_final_validation=True
        #     )
        # if accelerator.is_main_process:
        #     save_images_grids(images, masks, global_step, output_dir=self.args.output_dir)

        accelerator.end_training()
