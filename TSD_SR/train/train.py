#!/usr/bin/env python
# coding=utf-8
import argparse
import copy
import datetime
import logging
import math
import os
import sys
sys.path.append(os.getcwd())
import shutil
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F

import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
import pyiqa
import diffusers
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.image_processor import  VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from models.autoencoder_kl import AutoencoderKL
from utils.util import load_lora_state_dict
from data.data import Real_ESRGAN_Dataset

if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

def log_validation(
    log_dict,
    args,
    accelerator,
    vae_scale,
):
    logger.info(
        f"Running validation... \n Generating images with prompt:"
    )

    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
        
    with autocast_ctx:
        hq = torch.chunk(log_dict["hq"], args.train_batch_size, dim=0)[0]
        lq = torch.chunk(log_dict["lq"], args.train_batch_size, dim=0)[0]
        image_stu = torch.chunk(log_dict["image_stu"], args.train_batch_size, dim=0)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=2 ** (vae_scale - 1))
        image = image_processor.postprocess(hq.detach())[0]
        images.append(image)
        
        image = image_processor.postprocess(lq.detach())[0]
        images.append(image)
        
        image = image_processor.postprocess(image_stu.detach())[0]
        images.append(image)

    labels = ["hq", "lq", "image_stu"]
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{labels[i]}: {log_dict['prompt'][0]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--default_embedding_dir", 
        type=str, 
        default="dataset/default", 
        help='path to prompt embeddings'
    )
    parser.add_argument(
        "--null_embedding_dir", 
        type=str, 
        default="dataset/null", 
        help='path to prompt embeddings'
    )
    parser.add_argument(
        "--use_default_prompt",
        action="store_true",
        help="use default prompt",
    )
    parser.add_argument(
        "--use_dasm",
        action="store_true",
        help="use dasm",
    )
    parser.add_argument(
        "--use_teacher_lora",
        action="store_true",
        help="use fine-tune lora weights for teacher model",
    )
    parser.add_argument(
        "--use_random_bias",
        action="store_true",
        help="add ramdom to DASM step",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the SR model LoRA update matrices."),
    )
    parser.add_argument(
        "--rank_lora",
        type=int,
        default=16,
        help=("The dimension of the auxiliary model LoRA update matrices."),
    )
    parser.add_argument(
        "--rank_vae",
        type=int,
        default=64,
        help=("The dimension of the vae LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsdsr-checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="tsdsr",
        help="log_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--log_code",
        action="store_true",
        help="log code",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help="Number of steps to log validation images.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_reg",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' , `"wandb" (default)` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def collate_fn(examples, weight_dtype=torch.float16):
    lr_img = [example["lr_img"] for example in examples]
    hr_img = [example["hr_img"] for example in examples]
    latent_hr = [example["latent_hr"] for example in examples]
    
    prompts = [example["prompt_text"] for example in examples]
    prompt_embeds = torch.stack([example["prompt_embeds_input"] for example in examples])
    pooled_prompt_embeds = torch.stack([example["pooled_prompt_embeds_input"] for example in examples])

    lr_img = torch.stack(lr_img)
    hr_img = torch.stack(hr_img)
    latent_hr = torch.stack(latent_hr)
    
    batch = {
        "lr_img": lr_img.to(dtype=weight_dtype),
        "hr_img": hr_img.to(dtype=weight_dtype),
        "latent_hr": latent_hr.to(dtype=weight_dtype),
        "prompts": prompts,
        "prompt_embeds": prompt_embeds.to(dtype=weight_dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=weight_dtype),
             }
    return batch

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_tea = copy.deepcopy(noise_scheduler)
    
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    transformer_reg = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # now we will add new LoRA weights to the DiT model
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"],
    )
    # add the LoRA weights to the SR model
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    if args.checkpoint_path is not None:
        stu_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.checkpoint_path ,weight_name="transformer.safetensors")
        load_lora_state_dict(stu_lora_state_dict, transformer)
    transformer.enable_adapters()
    
    # add the LoRA weights to the teacher model if use fine-tuned lora weights
    if args.use_teacher_lora:
        transformer_reg.add_adapter(transformer_lora_config, adapter_name="default")    
        tea_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.teacher_lora_path, weight_name="teacher.safetensors")
        load_lora_state_dict(tea_lora_state_dict, transformer_reg)
    transformer_reg.requires_grad_(False)
    
    # add the LoRA weights to the lora model
    transformer_lora_config_reg = LoraConfig(
        r=args.rank_lora,
        lora_alpha=args.rank_lora,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj"],
    )
    transformer_reg.add_adapter(transformer_lora_config_reg, adapter_name="reg")    
    transformer_reg.enable_adapters()
    
    # add the LoRA weights to the vae encoder
    vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1', 'encoder.down_blocks.0.resnets.0.conv2', 'encoder.down_blocks.0.resnets.1.conv1', 
                          'encoder.down_blocks.0.resnets.1.conv2', 'encoder.down_blocks.0.downsamplers.0.conv', 'encoder.down_blocks.1.resnets.0.conv1',
                          'encoder.down_blocks.1.resnets.0.conv2', 'encoder.down_blocks.1.resnets.0.conv_shortcut', 'encoder.down_blocks.1.resnets.1.conv1', 'encoder.down_blocks.1.resnets.1.conv2', 
                          'encoder.down_blocks.1.downsamplers.0.conv', 'encoder.down_blocks.2.resnets.0.conv1', 'encoder.down_blocks.2.resnets.0.conv2',
                          'encoder.down_blocks.2.resnets.0.conv_shortcut', 'encoder.down_blocks.2.resnets.1.conv1', 'encoder.down_blocks.2.resnets.1.conv2', 'encoder.down_blocks.2.downsamplers.0.conv',
                          'encoder.down_blocks.3.resnets.0.conv1', 'encoder.down_blocks.3.resnets.0.conv2', 'encoder.down_blocks.3.resnets.1.conv1', 'encoder.down_blocks.3.resnets.1.conv2', 
                          'encoder.mid_block.attentions.0.to_q', 'encoder.mid_block.attentions.0.to_k', 'encoder.mid_block.attentions.0.to_v', 'encoder.mid_block.attentions.0.to_out.0', 
                          'encoder.mid_block.resnets.0.conv1', 'encoder.mid_block.resnets.0.conv2', 'encoder.mid_block.resnets.1.conv1', 'encoder.mid_block.resnets.1.conv2', 'encoder.conv_out', 'quant_conv']
    vae_lora_config = LoraConfig(
        r=args.rank_vae,
        lora_alpha=args.rank_vae,
        init_lora_weights="gaussian",
        target_modules=vae_target_modules
    )
    vae.add_adapter(vae_lora_config, adapter_name="default")
    if args.checkpoint_path is not None:
        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.checkpoint_path ,weight_name="vae.safetensors")
        load_lora_state_dict(vae_lora_state_dict, vae)
    vae.enable_adapters()

    # Set the model in the correct device and dtype.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer_reg.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Save and load model hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for index, model in enumerate(models):
                if isinstance(model, type(unwrap_model(transformer))) and index == 1:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"transformer.safetensors"
            ) 
                elif isinstance(model, type(unwrap_model(transformer_reg))) and index == 2:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="reg")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"transformer_reg.safetensors"
            )
                elif isinstance(model, type(unwrap_model(vae))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"vae.safetensors"
            )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        transformer_reg = models.pop()
        transformer = models.pop()
        vae = models.pop()

        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="vae.safetensors")
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="transformer.safetensors")
        reg_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="transformer_reg.safetensors")

        load_lora_state_dict(vae_lora_state_dict, vae)
        load_lora_state_dict(transformer_lora_state_dict, transformer)
        load_lora_state_dict(reg_lora_state_dict, transformer_reg, "reg")

        if args.mixed_precision == "fp16":
            models = [transformer, vae, transformer_reg]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # trainable parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) + list(filter(lambda p: p.requires_grad, vae.parameters()))
    transformer_reg_lora_parameters = list(filter(lambda p: p.requires_grad, transformer_reg.parameters())) 

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer, vae, transformer_reg]
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    
    transformer_reg_parameters_with_lr = {"params": transformer_reg_lora_parameters, "lr": args.learning_rate_reg}
    params_to_optimize_reg = [transformer_reg_parameters_with_lr]

    optimizer_class = torch.optim.AdamW
    optimizer_g = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_reg = optimizer_class(
        params_to_optimize_reg,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = Real_ESRGAN_Dataset(device=accelerator.device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_g = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_g,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    lr_scheduler_reg = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_reg,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if getattr(accelerator.state, "fsdp_plugin", None):
        fsdp_plugin = accelerator.state.fsdp_plugin
        fsdp_plugin.ignored_modules = [vae]
        
    # Prepare everything with our `accelerator`.
    vae, transformer, transformer_reg  = accelerator.prepare(vae, transformer, transformer_reg)
    train_dataloader, optimizer_g, lr_scheduler_g, optimizer_reg, lr_scheduler_reg  = accelerator.prepare(train_dataloader, optimizer_g, lr_scheduler_g, optimizer_reg, lr_scheduler_reg)        

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "tsdsr"
        log_name = args.log_name
        time = datetime.datetime.now().strftime('%m-%d_%H:%M')
        accelerator.init_trackers(tracker_name, config=vars(args), 
                                  init_kwargs={"wandb": {"name": f"{log_name}_lr{args.learning_rate}_{time}"}}
                                  )
        if args.log_code and is_wandb_available():
            wandb.run.log_code(".", log_name,
                           include_fn=lambda path: path.endswith(".py") or path.endswith(".sh"),
                           exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(".history/"))
        
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            if "latest-checkpoint" in dirs:
                path  = "latest-checkpoint"
            else:
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif path == "latest-checkpoint":
            accelerator.print(f"Resuming from checkpoint {path} and starting a new training run.")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Define the function to get the sigmas
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_tea.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_tea.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # cfg for transformer
    def compute_with_cfg(transformer, noisy_model_input, timesteps, encoder_hidden_states, pooled_projections, guidance_scale):
        noisy_model_input = torch.cat([noisy_model_input] * 2)
        timesteps_input = torch.cat([timesteps] * 2)
        prompt_embeds = torch.cat([prompt_embeds_null, encoder_hidden_states], dim=0)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_null, pooled_projections], dim=0) 
        
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
            )[0]
        pred_uncond, pred_cond = model_pred.chunk(2)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    
    log_dict = {}
    lpips = pyiqa.create_metric('lpips', as_loss=True).cuda()
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
    if args.use_default_prompt:
        prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "prompt_embeds.pt") , map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
        pooled_prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "pool_embeds.pt"), map_location=accelerator.device).repeat(args.train_batch_size, 1)
    prompt_embeds_null = torch.load(os.path.join(args.null_embedding_dir, "prompt_embeds.pt"), map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
    pooled_prompt_embeds_null = torch.load(os.path.join(args.null_embedding_dir, "pool_embeds.pt"), map_location=accelerator.device).repeat(args.train_batch_size, 1)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        transformer_reg.train()
        vae.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [vae, transformer, transformer_reg]
            with accelerator.accumulate(models_to_accumulate):
                lr_values = batch["lr_img"]
                hr_values = batch["hr_img"]
                prompts = batch["prompts"]
                prompt_embeds = batch["prompt_embeds"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                log_dict["lq"] = lr_values.float().cpu()
                log_dict["hq"] = hr_values.float().cpu()
                log_dict["prompt"] = prompts
                with autocast_ctx:
                    # Sample the latent space
                    model_input = unwrap_model(vae).encode(lr_values).latent_dist.sample() * unwrap_model(vae).config.scaling_factor
                    model_input_hr = batch["latent_hr"]
                    if not args.use_default_prompt:
                        prompt_embeds_default = prompt_embeds
                        pooled_prompt_embeds_default = pooled_prompt_embeds
                    
                    bsz = model_input.shape[0]
                    
                    timesteps = torch.tensor([1000.], device=accelerator.device)
                    # Sample a random timesteps for the teacher 
                    indices = torch.randint(50, 950, (bsz,))
                    timesteps_tea = noise_scheduler_tea.timesteps[indices].to(device=model_input.device)
                    
                    # Config for TSD
                    total_step = 4 if args.use_dasm else 1
                    weight_sigmas = [1.0, 0.3, 0.3, 0.3]
                    step_size = 50
                    random_bias = torch.randint(-step_size // 2, step_size // 2, (1,))[0] if args.use_random_bias else 0
                    
                    # Sample the timesteps for DASM
                    right_indices = min(indices[0] + step_size * total_step + random_bias, 999)
                    indices = torch.linspace(indices[0], right_indices, total_step + 1).long()
                    indices_cur, indices_next = indices[:total_step] , indices[1:] 
                    timestep_next = [noise_scheduler_tea.timesteps[torch.tensor([idx] * bsz)].to(device=model_input.device) for idx in indices_next]
                    timestep_cur = [noise_scheduler_tea.timesteps[torch.tensor([idx] * bsz)].to(device=model_input.device) for idx in indices_cur]
                    
                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_default,
                        pooled_projections=pooled_prompt_embeds_default,
                        return_dict=False,
                    )[0]
                    
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    latent_stu = model_pred * (-sigmas) + model_input
                    
                    # Add noise according to flow matching.
                    noise = torch.randn_like(model_input, device=model_input.device)
                    sigmas_tea = get_sigmas(timesteps_tea, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_input = sigmas_tea * noise + (1.0 - sigmas_tea) * latent_stu
                    noisy_hr = sigmas_tea * noise + (1.0 - sigmas_tea) * model_input_hr
                    
                    # Compute the TSD grad
                    grad_vsd = 0.0
                    grad_tsm = 0.0
                    iter_count = 0
                    with torch.no_grad():
                        for current_step,next_step,weight_sigma in zip(timestep_cur, timestep_next, weight_sigmas):
                            # Switch to the teacher model
                            if args.use_teacher_lora:
                                unwrap_model(transformer_reg).set_adapter("default")
                            else:
                                unwrap_model(transformer_reg).disable_adapters()
                            # Compute epsilon_psi(z^hat)
                            model_pred_tea = compute_with_cfg(
                                transformer_reg, 
                                noisy_input, 
                                current_step, 
                                prompt_embeds, 
                                pooled_prompt_embeds, 
                                args.guidance_scale)
                            # Compute epsilon_psi(z)
                            model_pred_tea_hr = compute_with_cfg(
                                transformer_reg, 
                                noisy_hr, 
                                current_step, 
                                prompt_embeds, 
                                pooled_prompt_embeds, 
                                args.guidance_scale)
                            
                            # Switch to the lora model
                            if args.use_teacher_lora:
                                unwrap_model(transformer_reg).set_adapter("reg")
                            else:
                                unwrap_model(transformer_reg).enable_adapters()
                            # Compute epsilon_phi(z^hat)
                            model_pred_reg = compute_with_cfg(
                                transformer_reg, 
                                noisy_input, 
                                current_step, 
                                prompt_embeds, 
                                pooled_prompt_embeds, 
                                args.guidance_scale)

                            sigmas_current = get_sigmas(current_step, n_dim=model_input.ndim, dtype=model_input.dtype)
                            sigmas_next = get_sigmas(next_step, n_dim=model_input.ndim, dtype=model_input.dtype)
                            # VSD grad -- SD3
                            grad_vsd += (model_pred_tea - model_pred_reg) * sigmas_current ** 2 * weight_sigma
                            grad_vsd = torch.nan_to_num(grad_vsd)
                            # TSM grad 
                            grad_tsm += (model_pred_tea - model_pred_tea_hr) * sigmas_current ** 2 * weight_sigma
                            grad_tsm = torch.nan_to_num(grad_tsm)
                            iter_count += 1
                            if iter_count == total_step:
                                break
                            # Update the noisy model input accross the sampling trajectory
                            noisy_input = noisy_input - (sigmas_current - sigmas_next) * model_pred_reg
                            noisy_hr = noisy_hr - (sigmas_current - sigmas_next) * model_pred_tea_hr    
                                                                
                        # Compute TSD loss
                        lambda_tsd = 0.7
                        grad = lambda_tsd * grad_vsd + (1 - lambda_tsd) * grad_tsm
                        tsddiff = (latent_stu - grad).detach()
                        tsd_loss = 0.5 * F.mse_loss(latent_stu.float(), tsddiff.float(), reduction='mean')
                    
                    image_stu = unwrap_model(vae).decode(latent_stu / unwrap_model(vae).config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                    if accelerator.is_main_process and step % args.validation_steps == 49:
                        log_dict["image_stu"] = image_stu.cpu()
                        
                    image_stu = (image_stu * 0.5 + 0.5)
                    hr_values = (hr_values * 0.5 + 0.5)
                    
                    # Remove mse loss when getting no regular spot but oversmoothed results
                    mse_loss = F.mse_loss(latent_stu.float(), model_input_hr.float().detach(), reduction='mean') 
                    lpips_loss = lpips(image_stu, hr_values) 
                    data_loss =  1 * lpips_loss.float() + 1 * mse_loss
                    data_loss = data_loss.mean()

                    # Compute total loss
                    loss_g = data_loss + tsd_loss

                # backward
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()
                # Clear the cache to avoid memory leak
                del image_stu, lr_values, hr_values, grad_vsd, grad_tsm, grad, tsddiff, model_pred_reg, model_pred_tea, model_pred_tea_hr, noisy_input, noisy_hr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Lora Model
                with autocast_ctx:
                    # Sample the noise and add it to the latent space
                    indices_lora = torch.randint(50, 950, (bsz,))
                    timesteps_lora = noise_scheduler_tea.timesteps[indices_lora].to(device=model_input.device)
                    noise = torch.randn_like(model_input, device=model_input.device)
                    sigmas_lora = get_sigmas(timesteps_lora, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_student = sigmas_lora * noise + (1.0 - sigmas_lora) * latent_stu
                    noisy_student = noisy_student.detach()
                    
                    # Switch to the lora model
                    if args.use_teacher_lora:
                        unwrap_model(transformer_reg).set_adapter("reg")
                    else:
                        unwrap_model(transformer_reg).enable_adapters()
                    model_pred_reg = transformer_reg(
                            hidden_states=noisy_student,
                            timestep=timesteps_lora,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                    model_pred_reg = model_pred_reg * (-sigmas_lora) + noisy_student

                    # Compute the weighting for the loss
                    if args.weighting_scheme == "sigma_sqrt":
                        weighting = (sigmas_lora**-2.0)
                    elif args.weighting_scheme == "logit_normal":
                        u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(bsz,), device=accelerator.device)
                        weighting = torch.nn.functional.sigmoid(u)
                    elif args.weighting_scheme == "mode":
                        u = torch.rand(size=(bsz,), device=accelerator.device)
                        weighting = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

                    # Compute the diffusion loss based on flow matching
                    epsilon_loss = 0.5 * weighting.view(-1,1,1,1) *  \
                    F.mse_loss(model_pred_reg.float(), latent_stu.detach().float(), reduction='mean')
                    epsilon_loss = torch.nan_to_num(epsilon_loss).mean()
                    
                    # Compute total lora model loss
                    loss_reg = epsilon_loss
                     
                # backward
                accelerator.backward(loss_reg)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_reg_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad()
                del model_pred_reg, noisy_student, prompt_embeds, pooled_prompt_embeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 500:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                    "epsilon": epsilon_loss.detach().item(),
                    "tsd_loss": tsd_loss.detach().item(),
                    "lpips_loss": lpips_loss.detach().item(), 
                    "mse_loss": mse_loss.detach().item(), 
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process and step % args.validation_steps == 49:
                log_validation(log_dict, args,accelerator, len(unwrap_model(vae).config.block_out_channels))

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
