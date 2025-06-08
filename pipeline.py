'''
full pipeline for pbr generation (v2)
'''
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'online'))
from typing import Tuple
from tqdm import tqdm
from glob import glob
import numpy as np
import rembg
import trimesh
import fpsample
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation
from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
from torchvision.utils import save_image

###texture tools for inpainting
from TextureTools.texturetools.geometry.uv.uv_atlas import preprocess_blank_mesh
from TextureTools.texturetools.geometry.sampling import geomerty_sampling
from TextureTools.texturetools.video.export_nvdiffrast_video import VideoExporter
from TextureTools.texturetools.render.nvdiffrast.renderer_inverse import NVDiffRendererInverse
from TextureTools.texturetools.render.nvdiffrast.renderer_utils import tensor_to_image, image_to_tensor
from TextureTools.texturetools.image.process_image import preprocess
from TextureTools.texturetools.utils.timer import CPUTimer


from TSD_SR.sr_pipeline import TSDSRPipeline

class RMBG2:
    def __init__(self, pretrain_models=None):
        if pretrain_models is None:
            pretrain_models = pretrain_models_default
        ckpt_id = f"{pretrain_models}/briaai/RMBG-2.0"
        model = AutoModelForImageSegmentation.from_pretrained(ckpt_id, trust_remote_code=True)
        # torch.set_float32_matmul_precision(['high', 'highest'][0])
        model.to('cuda').eval()
        self.model = model

        # Data settings
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_tensor = transforms.ToPILImage()

    def __call__(self, image:Image.Image) -> Image.Image:
        image = image.convert('RGB')
        input_images = self.transform_image(image).unsqueeze(0).to('cuda')
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = self.transform_tensor(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image


def build_rembg():
    return rembg.new_session(
        model_name="bria-rmbg",
        providers=[
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'HEURISTIC',
            }),
            'CPUExecutionProvider',
        ],
    )


def build_pipeline(pretrain_models=None, pipeline_name='texture_plus', model='rgb', super_resolutions=False,add_lora_path=None,add_lora_weights=None, speedup_mode=False):
    from flux_piplines.texturing.pipeline import PBRFluxPipeline as FluxPipeline
    lora_id = f"{pretrain_models}/UniTex/texture_gen/pytorch_lora_weights.safetensors"
    lora_id_delight = f"{pretrain_models}/UniTex/delight/pytorch_lora_weights.safetensors"
    ckpt_id = f"{pretrain_models}/black-forest-labs/FLUX.1-dev"
    redux_id = f"{pretrain_models}/black-forest-labs/FLUX.1-Redux-dev"
    if torch.cuda.get_device_properties(0).total_memory > 30 * (1024 ** 3):
        quantization_config = None
        cpu_offload = False
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    pipeline:FluxPipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=FluxTransformer2DModel.from_pretrained(
            ckpt_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        ),
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=torch.bfloat16,
    )
    pipeline.load_lora_weights(lora_id, adapter_name='texture')
    pipeline.load_lora_weights(lora_id_delight, adapter_name='delight')
    weights_for_texture = [1.,0.]
    weights_for_delight = [0.,1.]
    adapter_names = ['texture','delight']
    if add_lora_path is not None:
        for i in range(len(add_lora_path)):
            pipeline.load_lora_weights(add_lora_path[i], adapter_name=f'add_lora_{i}')
            adapter_names.append(f'add_lora_{i}')
            weights_for_texture.append(add_lora_weights[i])
            weights_for_delight.append(add_lora_weights[i])
    pipeline.transformer.__class__.__name__ = 'FluxTransformer2DModel'
    pipeline._num_inference_steps = 28
    if cpu_offload:
        pipeline._exclude_from_cpu_offload = ['transformer']
        pipeline.enable_model_cpu_offload(device='cuda')
    else:
        pipeline.transformer.to(device='cuda')
        pipeline.to(device='cuda')
    return pipeline,weights_for_texture,weights_for_delight,adapter_names


    
def build_ltm(pretrain_models=None):
    from LTM.rgb_field import RGBFieldVAE
    color_field_vae = RGBFieldVAE(cfg_path='LTM/configs/texture_model/textureman.yaml')
    color_field_vae = color_field_vae.eval().requires_grad_(False).to(device='cpu')
    color_field_vae_cpu_offload = True
    color_field_vae_sharp_edge = False
    return color_field_vae, color_field_vae_cpu_offload, color_field_vae_sharp_edge



class RGBTextureFullPipelineBase:
    def __init__(self, pretrain_models=None, pipeline_name='texture_plus',super_resolutions=False, seed=0, speedup_mode=None,add_lora_path=None, add_lora_weights=None, enable_rembg=False):
        if pretrain_models is None:
            pretrain_models = pretrain_models_default
        pipeline,weights_for_texture,weights_for_delight,adapter_names = build_pipeline(pretrain_models=pretrain_models, pipeline_name=pipeline_name,speedup_mode = speedup_mode, add_lora_path=add_lora_path,add_lora_weights=add_lora_weights, model='rgb')
        if enable_rembg:
            rembg_session = build_rembg()
        else:
            rembg_session = RMBG2(pretrain_models=pretrain_models)
        video_exporter = VideoExporter()
        inverse_renderer = NVDiffRendererInverse(device='cuda')
        generator = torch.Generator().manual_seed(seed)


        self.weights_for_texture = weights_for_texture
        self.weights_for_delight = weights_for_delight
        self.adapter_names = adapter_names
        self.pipeline_name = pipeline_name
        self.pipeline = pipeline
        self.rembg_session = rembg_session
        self.video_exporter = video_exporter
        self.inverse_renderer = inverse_renderer
        self.generator = generator
        self.super_resolutions = super_resolutions
        if self.super_resolutions:
            self.sr_pipeline = TSDSRPipeline()

        ###### build LTM #######

    @CPUTimer('preprocess_blank_mesh')
    def preprocess_blank_mesh(self, save_dir, input_mesh_path, min_faces=20_000, max_faces=200_000, scale=0.95):
        output_mesh_path = os.path.join(save_dir, 'processed_mesh.obj')
        preprocess_blank_mesh(
            input_mesh_path,
            output_mesh_path,
            min_faces=min_faces,
            max_faces=max_faces,
            scale=scale,
        )


    @CPUTimer('preprocess_reference_image')
    def preprocess_reference_image(self, save_dir, input_image_path, scale=0.95, color='grey'):
        input_image = Image.open(input_image_path).convert('RGB').resize((1024, 1024))
        output_image = preprocess(
            input_image,
            alpha=None,
            H=1024,
            W=1024,
            scale=scale,
            color=color,
            return_alpha=False,
            rembg_session=self.rembg_session,
        )
        output_image.save(os.path.join(save_dir, 'rembg_image.png'))
        output_image.convert('RGB').resize((512, 512)).save(os.path.join(save_dir, 'processed_image.png'))


    @CPUTimer('render_geometry_images')
    def render_geometry_images(self, save_dir, input_mesh_path, geometry_scale=0.95, scale=1.0, color='grey', four_or_six=False, flatten=False):
        out = self.video_exporter.export_condition(
            input_mesh_path,
            geometry_scale=geometry_scale,
            n_views=4 if four_or_six else 6,
            n_rows=1 if flatten else 2,
            n_cols=(4 if flatten else 2) if four_or_six else (6 if flatten else 3),
            H=512,
            W=512,
            fov_deg=49.1,
            scale=scale,
            perspective=False,
            orbit=False,
            background=color,
            return_info=False,
            return_image=True,
            return_mesh=False,
            return_camera=True,
        )

        out['alpha'].save(os.path.join(save_dir, 'mv_alpha.png'))
        out['ccm'].save(os.path.join(save_dir, 'mv_ccm.png'))
        out['normal'].save(os.path.join(save_dir, 'mv_normal.png'))
        camera_info = {
            'c2ws': out['c2ws'],
            'intrinsics': out['intrinsics'],
            'perspective': out['perspective'],
        }
        torch.save(camera_info, os.path.join(save_dir, 'camera_info.pth'))


    @CPUTimer('infer_mv')
    def infer_mv(self, save_dir, input_image_path, input_mv_image_path, add_input_mv_image_path):
        reference_image = Image.open(input_image_path).convert('RGB')
        control_image = Image.open(input_mv_image_path).convert('RGB')
        control_image_add = Image.open(add_input_mv_image_path).convert('RGB')
        num_inference_steps = getattr(self.pipeline, '_num_inference_steps', 28)
        if self.pipeline_name == 'texture_plus':
            # NOTE: frtbld -> frbltd
            img_temp = np.array(control_image).reshape(2, 512, 3, 512, -1)
            img_temp_add = np.array(control_image_add).reshape(2, 512, 3, 512, -1)
            img_temp = 0.5*img_temp + 0.5*img_temp_add
            img_temp = img_temp.astype(np.uint8)
            img_temp[1,:,2] = img_temp[1,::-1,2,::-1]
            control_image = Image.fromarray(img_temp.transpose(0, 2, 1, 3, 4).reshape(2*3, 512, 512, -1)[[0, 4,  1, 3, 2, 5]].transpose(1, 0, 2, 3).reshape(512, 2*3*512, -1))
            self.pipeline.set_adapters(adapter_names=self.adapter_names, adapter_weights=self.weights_for_texture)
            out = self.pipeline(
                prompt='[MVFLUX]',
                control_image=control_image,
                dual_image=reference_image,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                height=512,
                width=3072,
                n_rows=1,
                n_cols=6,
                num_inference_steps=num_inference_steps,
                guidance_scale=3.5,
                max_sequence_length=512,
                generator=self.generator,
            )
            out_image = out.images[0]
            out_image.save(os.path.join(save_dir, 'mv_rgb_w_light.png'))
            self.pipeline.set_adapters(adapter_names=self.adapter_names, adapter_weights=self.weights_for_delight)
            # NOTE: frbltd -> frtbld
            out = self.pipeline(
                prompt='[MVFLUX]',
                control_image=out_image,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                height=512,
                width=3072,
                n_rows=1,
                n_cols=6,
                num_inference_steps=num_inference_steps,
                guidance_scale=3.5,
                max_sequence_length=512,
                generator=self.generator,
            )
            out_delighted_image = out.images[0]
            img_temp = np.array(out_delighted_image).reshape(512, 6, 512, -1)
            img_temp[:,5] = img_temp[::-1,5,::-1]
            mv_rgb = Image.fromarray(img_temp.transpose(1, 0, 2, 3)[[0, 2, 4, 3, 1, 5]].reshape(2, 3, 512, 512, -1).transpose(0, 2, 1, 3, 4).reshape(2*512, 3*512, -1))
            
            if self.super_resolutions:
                mv_rgb.save(os.path.join(save_dir, 'mv_rgb_lr.png'))
                mv_rgb = self.sr_pipeline(mv_rgb)
                mv_rgb.save(os.path.join(save_dir, 'mv_rgb.png'))
            else:
                mv_rgb.save(os.path.join(save_dir, 'mv_rgb.png'))
        else:
            raise NotImplementedError(f'pipeline_name {self.pipeline_name} is not supported')


    @CPUTimer('export_video')
    def export_video(self, save_dir, input_mesh_path, output_video_name):
        output_video_path = os.path.join(save_dir, os.path.splitext(output_video_name)[0] + '.mp4')
        self.video_exporter.export_orbit_video(
            input_mesh_path,
            output_video_path,
            n_frames=120,
            enhance_mode=None,
            perspective=True,
            video_type='rgb',
            save_frames=False,
            save_grid=False,
            save_cover=False,
            save_camera=False,
            rename_with_euler=False,
        )


    @CPUTimer('reproject_and_query_field')
    def reproject_and_query_field(self, save_dir, input_mesh_path, input_mv_image_path, camera_info_path, four_or_six=False, flatten=False, method='reproject', inpainting=False):
        assert method in ['kdtree', 'reproject']
        n_views = 4 if four_or_six else 6
        n_rows = 1 if flatten else 2
        n_cols = (4 if flatten else 2) if four_or_six else (6 if flatten else 3)

        input_mv_image = Image.open(input_mv_image_path).convert('RGB')
        input_mv_image = image_to_tensor(input_mv_image, device='cuda')
        H, W, C = input_mv_image.shape
        HP, WP = H // n_rows, W // n_cols
        HT, WT = 2048, 2048
        image_attrs = input_mv_image.reshape(n_rows, HP, n_cols, WP, C).permute(0, 2, 1, 3, 4).reshape(n_views, HP, WP, C)
        camera_info = torch.load(camera_info_path, weights_only=True, map_location='cuda')
        c2ws = camera_info['c2ws']
        intrinsics = camera_info['intrinsics']
        perspective = camera_info['perspective']

        self.inverse_renderer.update_from_file(input_mesh_path)
        # if self.color_field_vae is not None and self.color_field_vae_cpu_offload:
        #     self.color_field_vae.cuda()
        textured_mesh, reprojected_uv, visable_mask, completed_uv_map = self.inverse_renderer.infer(
            input_mesh_path,
            c2ws=c2ws,
            intrinsics=intrinsics,
            image_attrs=image_attrs,
            perspective=perspective,
            H=HP, W=WP, H2D=HT, W2D=WT,
            method=method,
            kdtree_inpainting=inpainting,
            reproject_inpainting=inpainting,
            kdtree_n_neighbors = 8,
            kdtree_n_neighbors_visiable = 4,
            grad_norm_threhold=0.15,
            ray_normal_angle_threhold = 100,
            filt_gradient_points=inpainting
        )
        # if self.color_field_vae is not None and self.color_field_vae_cpu_offload:
        #     self.color_field_vae.cpu()
        textured_mesh.export(os.path.join(save_dir, 'textured_mesh.glb'))

        visable_uv_mask = reprojected_uv.any(dim=0,keepdim=True).float().permute(0,3,1,2)
        valid_uv_mask = visable_mask.float().permute(0,3,1,2)
        completed_uv_map = completed_uv_map.permute(0,3,1,2)
        save_image(visable_uv_mask,os.path.join(save_dir, 'visable_uv_mask.png'))
        save_image(valid_uv_mask,os.path.join(save_dir, 'valid_uv_mask.png'))
        save_image(completed_uv_map,os.path.join(save_dir, 'completed_uv.png'))
        self.inverse_renderer.clear()
        torch.cuda.empty_cache()


    @CPUTimer('sampling_on_mesh')
    def sampling_on_mesh(
        self, save_dir, input_mesh_path,
        scale=1.0, N=200000, N_fps=32768, angle=15.0,
    ):
        '''
        document: https://lightillusions.yuque.com/xzfrzd/cfv6oz/kyotxvygmuu0yder#W8LHk
        '''
        output_sharp_path = os.path.join(save_dir, 'sharp_pcd.ply')
        output_coarse_path = os.path.join(save_dir, 'coarse_pcd.ply')
        output_sharp_fps_path = os.path.join(save_dir, 'sharp_pcd_fps.ply')
        output_coarse_fps_path = os.path.join(save_dir, 'coarse_pcd_fps.ply')

        geomerty_sampling(
            input_mesh_path,
            output_sharp_path,
            output_coarse_path,
            scale=scale,
            N=N,
            angle_threhold_deg=angle,
            method='equal_steps',
            merge_close_vertices=True,
        )

        sharp_surface = trimesh.load(output_sharp_path, process=False)
        sharp_surface = getattr(sharp_surface, 'vertices', None)
        if sharp_surface is not None and sharp_surface.shape[0] > 0:
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(sharp_surface, n_samples=N_fps, h=5)
            fps_sharp_surface = sharp_surface[kdline_fps_samples_idx]
            fps_sharp_surface = np.nan_to_num(fps_sharp_surface, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            fps_sharp_surface = np.full((N_fps, 3), fill_value=1.0)
        fps_sharp_surface = trimesh.Trimesh(vertices=fps_sharp_surface, process=False)
        fps_sharp_surface.export(output_sharp_fps_path)

        coarse_surface = trimesh.load(output_coarse_path, process=False)
        coarse_surface = getattr(coarse_surface, 'vertices', None)
        if coarse_surface is not None and coarse_surface.shape[0] > 0:
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface, n_samples=N_fps, h=5)
            fps_coarse_surface = coarse_surface[kdline_fps_samples_idx]
            fps_coarse_surface = np.nan_to_num(fps_coarse_surface, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            fps_coarse_surface = np.full((N_fps, 3), fill_value=1.0)
        fps_coarse_surface = trimesh.Trimesh(vertices=fps_coarse_surface, process=False)
        fps_coarse_surface.export(output_coarse_fps_path)


    @CPUTimer('infer_field')
    def infer_field(
        self, save_dir, input_mesh_path, sharp_fps_path, coarse_fps_path,
        mv_alpha_image_path, mv_ccm_image_path, mv_rgb_image_path,
        color='gray', four_or_six=False, input_four_or_six=False, input_flatten=False, N_fps=32768,
        base_or_inpainting=False, test_query_field=False, save_point_clouds=False,
    ):
        n_views = 4 if four_or_six else 6
        input_n_views = 4 if input_four_or_six else 6
        input_n_rows = 1 if input_flatten else 2
        input_n_cols = (4 if input_flatten else 2) if input_four_or_six else (6 if input_flatten else 3)

        # prepare image latents
        mv_alpha_image = Image.open(mv_alpha_image_path).convert('L')
        mv_ccm_image = Image.open(mv_ccm_image_path).convert('RGB')
        mv_rgb_image = Image.open(mv_rgb_image_path).convert('RGB')
        mv_ccm_image_processed = Image.new('RGB', mv_ccm_image.size, color)
        mv_ccm_image_processed.paste(mv_ccm_image, mask=mv_alpha_image)
        mv_rgb_image_processed = Image.new('RGB', mv_rgb_image.size, color)
        mv_rgb_image_processed.paste(mv_rgb_image, mask=mv_alpha_image)
        alpha_frtbld = image_to_tensor(mv_alpha_image, device='cuda').mean(dim=-1, keepdim=True)
        ccm_frtbld = image_to_tensor(mv_ccm_image_processed, device='cuda')
        albedo_frtbld = image_to_tensor(mv_rgb_image_processed, device='cuda')
        H, W, C = albedo_frtbld.shape
        HP, WP = H // input_n_rows, W // input_n_cols
        ccm_frtbld = (ccm_frtbld * 2.0 - 1.0) @ torch.tensor([
            [ 1, 0, 0],
            [ 0, 0, 1],
            [ 0, -1, 0],
        ]).to(ccm_frtbld) * 0.5 + 0.5
        if four_or_six:
            alpha_fbrl = alpha_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 1).permute(0, 2, 4, 1, 3).reshape(input_n_views, 1, HP, WP)[[0, 3, 1, 4]]
            ccm_fbrl = ccm_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 3).permute(0, 2, 4, 1, 3).reshape(input_n_views, 3, HP, WP)[[0, 3, 1, 4]]
            albedo_fbrl = albedo_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 3).permute(0, 2, 4, 1, 3).reshape(input_n_views, 3, HP, WP)[[0, 3, 1, 4]]
            for item in [alpha_fbrl, ccm_fbrl, albedo_fbrl]:
                item[1, :, :, :] = item[1, :, :, :].flip(2)
                item[3, :, :, :] = item[3, :, :, :].flip(2)
                item[5, :, :, :] = item[5, :, :, :].flip(1)
        else:
            # NOTE: frtbld -> fbrltd
            alpha_fbrltd = alpha_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 1).permute(0, 2, 4, 1, 3).reshape(input_n_views, 1, HP, WP)[[0, 3, 1, 4, 2, 5]]
            ccm_fbrltd = ccm_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 3).permute(0, 2, 4, 1, 3).reshape(input_n_views, 3, HP, WP)[[0, 3, 1, 4, 2, 5]]
            albedo_fbrltd = albedo_frtbld.reshape(input_n_rows, HP, input_n_cols, WP, 3).permute(0, 2, 4, 1, 3).reshape(input_n_views, 3, HP, WP)[[0, 3, 1, 4, 2, 5]]
            for item in [alpha_fbrltd, ccm_fbrltd, albedo_fbrltd]:
                item[1, :, :, :] = item[1, :, :, :].flip(2)
                item[3, :, :, :] = item[3, :, :, :].flip(2)
                item[5, :, :, :] = item[5, :, :, :].flip(2)

        # prepare surface points
        surface_pcd = trimesh.load(coarse_fps_path, process=False)  # NOTE: no check
        surface_pcd = torch.as_tensor(surface_pcd.vertices, dtype=torch.float32, device='cuda')
        sharp_pcd = trimesh.load(sharp_fps_path, process=False)  # NOTE: no check
        sharp_pcd = torch.as_tensor(sharp_pcd.vertices, dtype=torch.float32, device='cuda')
        if base_or_inpainting:
            surface = torch.cat([surface_pcd, torch.full_like(surface_pcd, fill_value=2.0)], dim=-1)
            if self.color_field_vae_sharp_edge:
                sharp_surface = torch.cat([sharp_pcd, torch.full_like(sharp_pcd, fill_value=2.0)], dim=-1)
            else:
                sharp_surface = None
            if four_or_six:
                kl_embed = self.color_field_vae.encode_geometry(
                    alpha_fbrl=alpha_fbrl.unsqueeze(0),
                    albedo_fbrl=albedo_fbrl.unsqueeze(0),
                    ccm_fbrl=ccm_fbrl.unsqueeze(0),
                    surface=surface.unsqueeze(0),
                    sharp_surface=sharp_surface.unsqueeze(0),
                    sample_posterior=True,
                )
            else:
                kl_embed = self.color_field_vae.encode_geometry(
                    alpha_fbrltd=alpha_fbrltd.unsqueeze(0),
                    albedo_fbrltd=albedo_fbrltd.unsqueeze(0),
                    ccm_fbrltd=ccm_fbrltd.unsqueeze(0),
                    surface=surface.unsqueeze(0),
                    sharp_surface=sharp_surface.unsqueeze(0),
                )

            # register query_field function
            def query_field(v_vis:torch.Tensor, c_vis:torch.Tensor, v_invis:torch.Tensor):
                logits = self.color_field_vae.decode_field(
                    latents=kl_embed,
                    queries=v_invis.unsqueeze(0),
                    batch_size=1024*1024,
                    return_pcds=False,
                )
                c_invis = logits.squeeze(0).clamp(0.0, 1.0)
                if save_point_clouds:
                    pcd_input = trimesh.Trimesh(vertices=v_vis.detach().cpu().numpy(), vertex_colors=c_vis.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    pcd_output = trimesh.Trimesh(vertices=v_invis.detach().cpu().numpy(), vertex_colors=c_invis.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    pcd_input.export(os.path.join(save_dir, f'pcd_input.ply'))
                    pcd_output.export(os.path.join(save_dir, f'pcd_output.ply'))
                return c_invis
        else:
            if self.color_field_vae_sharp_edge:
                sharp_surface = torch.cat([sharp_pcd, torch.full_like(sharp_pcd, fill_value=2.0)], dim=-1)
            else:
                sharp_surface = None
            def query_field(v_vis:torch.Tensor, c_vis:torch.Tensor, v_invis:torch.Tensor):
                coarse_surface = torch.cat([
                    torch.cat([v_vis, c_vis], dim=-1), 
                    torch.cat([v_invis, torch.full_like(v_invis, fill_value=2.0)], dim=-1),
                ], dim=0)
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface[:, :3].detach().cpu().numpy(), n_samples=N_fps, h=5)
                kdline_fps_samples_idx = torch.as_tensor(kdline_fps_samples_idx, dtype=torch.int64, device=coarse_surface.device)
                surface = coarse_surface[kdline_fps_samples_idx, :]
                if save_point_clouds:
                    pcd_input = trimesh.Trimesh(vertices=v_vis.detach().cpu().numpy(), vertex_colors=c_vis.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    pcd_input_coarse = trimesh.Trimesh(vertices=coarse_surface[:, :3].detach().cpu().numpy(), vertex_colors=coarse_surface[:, 3:].clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    pcd_input_coarse_fps = trimesh.Trimesh(vertices=surface[:, :3].detach().cpu().numpy(), vertex_colors=surface[:, 3:].clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    if sharp_surface is not None:
                        pcd_input_sharp = None
                        pcd_input_sharp_fps = trimesh.Trimesh(vertices=sharp_surface[:, :3].detach().cpu().numpy(), vertex_colors=sharp_surface[:, 3:].clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    else:
                        pcd_input_sharp = None
                        pcd_input_sharp_fps = None
                    pcd_input.export(os.path.join(save_dir, f'pcd_input.ply'))
                    pcd_input_coarse.export(os.path.join(save_dir, f'pcd_input_coarse.ply'))
                    pcd_input_coarse_fps.export(os.path.join(save_dir, f'pcd_input_coarse_fps.ply'))
                    if pcd_input_sharp is not None:
                        pcd_input_sharp.export(os.path.join(save_dir, f'pcd_input_sharp.ply'))
                    if pcd_input_sharp_fps is not None:
                        pcd_input_sharp_fps.export(os.path.join(save_dir, f'pcd_input_sharp.ply'))
                if four_or_six:
                    kl_embed = self.color_field_vae.encode_geometry(
                        alpha_fbrl=alpha_fbrl.unsqueeze(0),
                        albedo_fbrl=albedo_fbrl.unsqueeze(0),
                        ccm_fbrl=ccm_fbrl.unsqueeze(0),
                        surface=surface.unsqueeze(0),
                        sharp_surface=sharp_surface.unsqueeze(0) if sharp_surface is not None else None,
                        sample_posterior=True,
                    )
                else:
                    kl_embed = self.color_field_vae.encode_geometry(
                        alpha_fbrltd=alpha_fbrltd.unsqueeze(0),
                        albedo_fbrltd=albedo_fbrltd.unsqueeze(0),
                        ccm_fbrltd=ccm_fbrltd.unsqueeze(0),
                        surface=surface.unsqueeze(0),
                        sharp_surface=sharp_surface.unsqueeze(0) if sharp_surface is not None else None,
                    )
                logits = self.color_field_vae.decode_field(
                    latents=kl_embed,
                    queries=v_invis.unsqueeze(0),
                    batch_size=1024*1024,
                    return_pcds=False,
                )
                c_invis = logits.squeeze(0).clamp(0.0, 1.0)
                if save_point_clouds:
                    pcd_output = trimesh.Trimesh(vertices=v_invis.detach().cpu().numpy(), vertex_colors=c_invis.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), process=False)
                    pcd_output.export(os.path.join(save_dir, f'pcd_output.ply'))
                return c_invis
        self.inverse_renderer.register_query_field(query_field=query_field)
        if test_query_field:
            test_mesh_gt, test_mesh_pred = self.inverse_renderer.test_query_field(blank_mesh=input_mesh_path)
            test_mesh_gt.export(os.path.join(save_dir, 'test_mesh_gt.glb'))
            test_mesh_pred.export(os.path.join(save_dir, 'test_mesh_pred.glb'))



class RGBTextureFullPipeline(RGBTextureFullPipelineBase):
    def step_1_1(self, cache_dir:str, input_image_path:str, input_mesh_path:str, clear_cache=False,super_resolutions=False, seed=0, *args, **kwargs):
        print(f"step_1_1: {input_image_path}, {input_mesh_path}")
        self.preprocess_blank_mesh(cache_dir, input_mesh_path)
        self.preprocess_reference_image(cache_dir, input_image_path)
        self.render_geometry_images(cache_dir, input_mesh_path)
        #self.infer_mv(cache_dir, os.path.join(cache_dir, 'processed_image.png'), os.path.join(cache_dir, 'mv_normal.png'))
        self.infer_mv(cache_dir, os.path.join(cache_dir, 'processed_image.png'), os.path.join(cache_dir, 'mv_normal.png'), os.path.join(cache_dir, 'mv_ccm.png'))

    def step_2_1(self, cache_dir:str, input_image_path:str, input_mesh_path:str, clear_cache=False, *args, **kwargs):
        self.reproject_and_query_field(cache_dir, os.path.join(cache_dir, 'processed_mesh.obj'), os.path.join(cache_dir, 'mv_rgb.png'), os.path.join(cache_dir, 'camera_info.pth'), inpainting=False)
        if not clear_cache:
            self.export_video(cache_dir, os.path.join(cache_dir, 'textured_mesh.glb'), 'textured_mesh.mp4')

    def step_2_2(self, cache_dir:str, input_image_path:str, input_mesh_path:str, clear_cache=False, *args, **kwargs):
        self.sampling_on_mesh(cache_dir, os.path.join(cache_dir, 'processed_mesh.obj'))
        self.infer_field(
            cache_dir, os.path.join(cache_dir, 'processed_mesh.obj'), os.path.join(cache_dir, 'sharp_pcd_fps.ply'), os.path.join(cache_dir, 'coarse_pcd_fps.ply'), 
            os.path.join(cache_dir, 'mv_alpha.png'), os.path.join(cache_dir, 'mv_ccm.png'), os.path.join(cache_dir, 'mv_rgb.png'),
            base_or_inpainting=False, test_query_field=False, save_point_clouds=False,
        )
        self.reproject_and_query_field(cache_dir, os.path.join(cache_dir, 'processed_mesh.obj'), os.path.join(cache_dir, 'mv_rgb.png'), os.path.join(cache_dir, 'camera_info.pth'), inpainting=True)
        if not clear_cache:
            self.export_video(cache_dir, os.path.join(cache_dir, 'textured_mesh.glb'), 'textured_mesh.mp4')

    step_seq = ['step_1_1', 'step_2_1']
    def __call__(
        self, 
        save_dir:str, 
        input_image_path:str, 
        input_mesh_path:str, 
        clear_cache=False,
    ) -> Tuple[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        cache_dir = os.path.join(os.path.abspath(save_dir), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        kwargs = dict(
            cache_dir=cache_dir, 
            input_image_path=input_image_path, 
            input_mesh_path=input_mesh_path, 
            clear_cache=clear_cache,
        )
        for step in self.step_seq:
            getattr(self, step)(**kwargs)
        shutil.copy(os.path.join(cache_dir, 'rembg_image.png'), os.path.join(save_dir, 'rembg_image.png'))
        shutil.copy(os.path.join(cache_dir, 'mv_rgb.png'), os.path.join(save_dir, 'mv_rgb.png'))
        shutil.copy(os.path.join(cache_dir, 'textured_mesh.glb'), os.path.join(save_dir, 'textured_mesh.glb'))
        if clear_cache:
            shutil.rmtree(cache_dir)
        return os.path.join(save_dir, 'rembg_image.png'), os.path.join(save_dir, 'textured_mesh.glb')


class CustomRGBTextureFullPipeline(RGBTextureFullPipeline):
    step_seq = ['step_1_1','step_2_ablition']

    # NOTE: ONLY wo LTM before paper is accepted
    def step_2_ablition(self, cache_dir:str, input_image_path:str, input_mesh_path:str, clear_cache=False, *args, **kwargs):
        os.makedirs(os.path.join(cache_dir, 'wo_LTM'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'w_LTM'), exist_ok=True)
        self.reproject_and_query_field(os.path.join(cache_dir, 'wo_LTM'), os.path.join(cache_dir, 'processed_mesh.obj'), os.path.join(cache_dir, 'mv_rgb.png'), os.path.join(cache_dir, 'camera_info.pth'), inpainting=False)
        self.export_video(os.path.join(cache_dir, 'wo_LTM'), os.path.join(cache_dir, 'wo_LTM/textured_mesh.glb'), 'textured_mesh.mp4')
        shutil.copy(os.path.join(cache_dir, 'wo_LTM/textured_mesh.glb'), os.path.join(cache_dir, 'textured_mesh.glb'))

        ############ w_LTM ##################
        print("The second stage (LTM part) will be released after the paper is accepted.")
