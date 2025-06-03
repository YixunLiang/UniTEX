import glob
import time
import os
import sys
sys.path.append(".")
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig
from diffusers import (
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from TSD_SR.models.autoencoder_kl import AutoencoderKL
from TSD_SR.utils.vaehook import _init_tiled_vae
from TSD_SR.utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from TSD_SR.utils.util import load_lora_state_dict

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers/", help='path to the pretrained sd3')
    parser.add_argument("--lora_dir", type=str, default="TSD-SR/checkpoint/tsdsr/", help='path to tsd-sr lora weights')
    parser.add_argument("--embedding_dir", type=str, default="TSD-SR/dataset/default", help='path to prompt embeddings')
    parser.add_argument("--rank", type=int, default=64, help='rank for transformer')
    parser.add_argument("--rank_vae", type=int, default=64, help='rank for vae')
    
    parser.add_argument("--is_use_tile", type=bool, default=True, help='whether to use tiled vae')
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=128, help='tiled size for tiled vae decoder') 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=512, help='tiled size for tiled vae encoder') 
    parser.add_argument("--latent_tiled_size", type=int, default=64, help='tiled size for transformer latent')
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help='tiled overlap for transformer latent')
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4, help='upscale factor')
    parser.add_argument("--process_size", type=int, default=512, help='process size for images')
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet', help='color alignment method')
    return parser.parse_args()


def _gaussian_weights(tile_width, tile_height, nbatches, transformer):
    """Generates a gaussian mask of weights for tile contributions"""
    from numpy import pi, exp, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=transformer.device), (nbatches, transformer.config.in_channels, 1, 1))

def tile_sample(lq_latent, lq, transformer, timesteps, prompt_embeds, pooled_prompt_embeds, weight_dtype, args):
    with torch.no_grad():
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (args.latent_tiled_size, args.latent_tiled_overlap)
        # print(h,w,tile_size,tile_overlap)
        if h * w <= tile_size * tile_size:
            # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred =  transformer(
                                hidden_states=lq_latent,
                                timestep=timesteps,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                return_dict=False,
                            )[0]
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = _gaussian_weights(tile_size, tile_size, 1,transformer)
            tile_size = min(tile_size, min(h, w))
            tile_weights = _gaussian_weights(tile_size, tile_size, 1,transformer)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # Extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # Input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # Input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0).to(args.device, dtype=weight_dtype)
                        # print(input_list_t.shape)
                        model_out =  transformer(
                            hidden_states=input_list_t,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        input_list = []
                        
                    noise_preds.append(model_out)
                    
            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=args.device)
            contributors = torch.zeros(lq_latent.shape, device=args.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # Extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # Input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred
            
    return model_pred.to(args.device, dtype=weight_dtype)



class TSDSRPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        args = parse_args()
        self.args = args
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        self.weight_dtype = weight_dtype
        # Load the pretrained models
        transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", torch_dtype=weight_dtype)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
        
        if args.is_use_tile:
            _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)
        
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

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
        vae.add_adapter(vae_lora_config)
        vae.enable_adapters()
        
        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="vae.safetensors")
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="transformer.safetensors")
        
        load_lora_state_dict(vae_lora_state_dict, vae)
        load_lora_state_dict(transformer_lora_state_dict, transformer)

        self.vae = vae.to(args.device, dtype=weight_dtype)
        self.transformer = transformer.to(args.device, dtype=weight_dtype)
            # Load the prompt embeddings
        self.prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        self.prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
        self.pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
    

    def forward(self,lr):
        ori_width, ori_height = lr.size
        upscale = self.args.upscale
        process_size = self.args.process_size
        
        # Resize the image if it is not valid
        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale*ori_width), int(scale*ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale*new_width, upscale*new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8
            
        lr_scale = lr.resize((int(ori_width*self.args.upscale), int(ori_height*self.args.upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(self.args.device, dtype=self.weight_dtype)
        size = (new_height, new_width)
        start_time = time.time()
        with torch.no_grad():
            # Preprocess the input image
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(self.args.device, dtype=self.weight_dtype).clamp(-1,1)
            # Encode the input image
            model_input = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
            model_input = model_input.to(self.args.device, dtype=self.weight_dtype)
            timesteps = torch.tensor([1000.], device=self.args.device, dtype=self.weight_dtype)
            # Predict
            model_pred = tile_sample(model_input, pixel_values, self.transformer, timesteps, self.prompt_embeds, self.pooled_prompt_embeds, self.weight_dtype, self.args)
            latent_stu = model_input - model_pred
            
            # Decode the output
            image = self.vae.decode(latent_stu / self.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
            
        #image = self.main(pixel_values, (new_height, new_width))
        end_time = time.time()
        image_pil_image = transforms.ToPILImage()(image.cpu() / 2 + 0.5)      
        total_time = (end_time - start_time)
        if resize_flag:
            image_pil_image = image_pil_image.resize((int(ori_width*self.args.upscale), int(ori_height*self.args.upscale)))
        
        if self.args.align_method == 'adain':
            image_pil_image = adain_color_fix(target=image_pil_image, source=lr)
        elif self.args.align_method == 'wavelet':
            image_pil_image = wavelet_color_fix(target=image_pil_image, source=lr_scale)
        else:
            pass
        return image_pil_image

if __name__ == "__main__":
    args = parse_args()
    srpipe = TSDSRPipeline(args)
    image_name = 'projects/test_lucid3/rembg_image.png'
    lr = Image.open(image_name).convert('RGB')
    sr = srpipe(lr)
    breakpoint()