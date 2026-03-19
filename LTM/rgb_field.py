import os
import sys
from typing import List, Optional, Union
from einops import rearrange
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
import trimesh


from LTM.craftsman.utils.config import load_config
from LTM.craftsman.models.autoencoders.flux_autoencoder_unet2 import FluxAutoencoderUnet


class RGBFieldVAE(nn.Module):
    def __init__(self, flux_path=None, cfg_path=None, pretrain_ckpt=None):
        super().__init__()
        cfg = OmegaConf.load(cfg_path)
        if flux_path is not None:
            cfg.system.shape_model.pretrain_flux_name_or_path = flux_path
        if pretrain_ckpt is not None:
            cfg.system.shape_model.pretrained_model_name_or_path = pretrain_ckpt
        cfg.system.shape_model.sample_type = 'mode_sample'  # mode_sample or mode
        cfg.system.shape_model.use_checkpoint = False
        self.shape_model = FluxAutoencoderUnet(cfg.system.shape_model)

    @property
    def shape_model_dtype(self):
        return next(self.shape_model.parameters()).dtype

    def encode_geometry(
        self,
        alpha_fbrltd: torch.Tensor, 
        albedo_fbrltd: torch.Tensor, 
        ccm_fbrltd: torch.Tensor, 
        surface: torch.Tensor, 
        sharp_surface: torch.Tensor=None,
    ) -> torch.Tensor:
        B, N, C, H, W = albedo_fbrltd.shape
        pixel_values = torch.cat([albedo_fbrltd, ccm_fbrltd, alpha_fbrltd], dim=2).reshape(B*N, 2*C+1, H, W)
        if H != 512 or W != 512:
            pixel_values[:, :2*C, :, :] = torch.nn.functional.interpolate(pixel_values[:, :2*C, :, :], size=(512, 512))
            pixel_values[:, 2*C:, :, :] = torch.nn.functional.interpolate(pixel_values[:, 2*C:, :, :], size=(512, 512), mode='nearest-exact')
        H = W = 512
        pixel_values[:, :2*C, :, :] = pixel_values[:, :2*C, :, :] * 2.0 - 1.0
        pixel_values = pixel_values.reshape(B, N, 2*C+1, H, W).permute(0, 2, 3, 1, 4).reshape(B, 2*C+1, H, N*W)
        surface[:, :, 3:] = torch.where((surface[:, :, 3:] == 2.0).all(dim=-1, keepdim=True), 2.0, surface[:, :, 3:] * 2.0 - 1.0)
        if sharp_surface is not None:
            sharp_surface[:, :, 3:] = torch.where((sharp_surface[:, :, 3:] == 2.0).all(dim=-1, keepdim=True), 2.0, sharp_surface[:, :, 3:] * 2.0 - 1.0)

        ################ for debug ################
        # surface[..., 3:] = torch.where(surface[..., [2]] > 0.0, 2.0, surface[..., 3:])
        # save_image(pixel_values[:, C:2*C:, :, :].mul(0.5).add(0.5), 'debug.png')
        # trimesh.Trimesh(vertices=surface[:, :, :3].reshape(-1, 3).detach().cpu().numpy(), vertex_colors=surface[:, :, 3:].mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).reshape(-1, 3).detach().cpu().numpy().astype(np.uint8), process=False).export('debug.ply')
        ################ for debug ################

        latents = self.shape_model.decode(
            latents=pixel_values.to(dtype=self.shape_model_dtype, device=self.shape_model.device), 
            surface=surface.to(dtype=self.shape_model_dtype, device=self.shape_model.device), 
            sharp_surface=sharp_surface.to(dtype=self.shape_model_dtype, device=self.shape_model.device) if sharp_surface is not None else None,
        )
        return latents

    def decode_field(
        self,
        queries:torch.Tensor,
        latents:torch.Tensor,
        batch_size:Optional[int]=None,
        return_pcds=True,
    ) -> Union[trimesh.Trimesh, torch.Tensor]:
        if batch_size is None:
            logits = self.shape_model.query(queries, latents)
        else:
            B, L, C = queries.shape
            logits = None
            for idx in range(0, L, batch_size):
                logits_batch = self.shape_model.query(queries[:, idx:idx+batch_size, :], latents)
                if logits is None:
                    logits = torch.zeros((B, L, logits_batch.shape[-1]), dtype=logits_batch.dtype, device=logits_batch.device)
                logits[:, idx:idx+batch_size, :] = logits_batch
        logits = logits * 0.5 + 0.5
        if return_pcds:
            queries_np = queries.detach().cpu().numpy()
            logits_np = logits.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
            pcds = [
                trimesh.Trimesh(
                    vertices=query, 
                    vertex_colors=logit,
                    process=False,
                ) for query, logit in zip(queries_np, logits_np)
            ]
            return logits, pcds
        return logits




if __name__ == '__main__':
    # project_path=$(realpath -es .)
    # export PYTHONPATH=$PYTHONPATH:${project_path}/mvdataset
    from pbr_diffusion.rgb_field_dataloader_6_views_v2 import GeometryDataset, GeometryDatasetConfig
    data_root = '/mnt/jfs/algo/demos-texture/test_data/sketchfab_free_geometry_diffusion/'
    save_dir = 'test_result/test_new_field/debug'

    ds = GeometryDataset(
        data_root,
        GeometryDatasetConfig(
            image_ext='.png',
            skip_broken=False,
        ),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    flux_path = "/mnt/jfs/kunming/projects/pret_models/black-forest-labs/FLUX.1-dev"
    cfg_path = '/mnt/jfs/kunming/projects/Texman/LTM/configs/texture_model/textureman.yaml'
    pretrain_ckpt = '/mnt/jfs/kunming/projects/pret_models/textureman/mvdiffusion_training_release/flux_fields/test_ckpt_last_wfield/pytorch_model.bin' 
    vae = RGBFieldVAE(flux_path=flux_path, cfg_path=cfg_path, pretrain_ckpt=pretrain_ckpt)  # NOTE: peak memory ~ 8 GB
    vae = vae.eval().requires_grad_(False).cuda()

    for i, batch in tqdm(enumerate(dl), total=len(dl)):
        uids = batch['uid']
        surface = batch['surface'][..., :6].cuda()
        batch['ccm_fbrltd'] = (batch['ccm_fbrltd'].permute(0,1,3,4,2) * 2.0 - 1.0) @ torch.tensor([
            [ 1, 0, 0],
            [ 0, 0, -1],
            [ 0, 1, 0],
        ]).to(batch['ccm_fbrltd']) * 0.5 + 0.5  
        batch['ccm_fbrltd'] = batch['ccm_fbrltd'].permute(0,1,4,2,3)
        kl_embed = vae.encode_geometry(
            alpha_fbrltd=batch['alpha_fbrltd'].cuda(),
            albedo_fbrltd=batch['albedo_fbrltd'].cuda(), 
            ccm_fbrltd=batch['ccm_fbrltd'].cuda(), 
            surface=surface, 
            sharp_surface=None, 
        )
        _, pcds = vae.decode_field(
            latents=kl_embed,
            queries=surface[..., :3],
            batch_size=None,
            return_pcds=True,
        )

        for j, uid in enumerate(uids):
            os.makedirs(os.path.dirname(f'{save_dir}/{uid}.png'), exist_ok=True)
            pcds[j].export(f'{save_dir}/{uid}.obj')

