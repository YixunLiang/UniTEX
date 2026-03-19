import argparse
from datetime import datetime
from glob import glob
import os
import gradio as gr
import numpy as np
from PIL import Image

from texturetools.video.export_nvdiffrast_video import VideoExporter
from texturetools.renderers.nvdiffrast.renderer_pbr import NVDiffRendererPBR


def call_render_base(input_mesh, video_type):
    assert video_type in ['world_position', 'world_normal']
    task_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '-' + scope
    current_dir = os.path.join(cache_dir, task_id)
    os.makedirs(current_dir, exist_ok=True)
    output_video_path = os.path.join(current_dir, 'output_video.mp4')
    output_image_path = os.path.join(current_dir, 'output_video_grid.png')
    print(f"current_dir = {current_dir}")

    n_views, n_rows, n_cols, H, W = (4, 2, 2, 1024, 1024)
    info = exporter.export_info(
        input_mesh, 
        n_views=n_views, scale=0.85, fov_deg=49.1, perspective=False, orbit=True,
        background='white',
    )
    result = renderer.render_base(
        info['texture'].mesh,
        info['texture'].map_Kd,
        info['texture'].map_Ks,
        info['c2ws'],
        info['intrinsics'],
        render_size=(H, W),
    )
    result['background'] = info['background']
    im = result[video_type].mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    im = Image.fromarray(im.reshape(n_rows, n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, 3), mode='RGB')
    im.save(output_image_path)
    return output_image_path, gr.update(choices=renderer.index_list)


def call_render_pbr(
    index, video_type,
    lambda_albedo_r=1.0,
    lambda_albedo_g=1.0,
    lambda_albedo_b=1.0,
    lambda_matellic=1.0,
    lambda_roughness=1.0,
    lambda_diffuse=1.0, 
    lambda_specular=1.0,
):
    assert index is not None and index in renderer.index_list
    assert video_type in ['rgb', 'diffuse', 'specular']
    lambda_albedo_r = float(lambda_albedo_r)
    lambda_albedo_g = float(lambda_albedo_g)
    lambda_albedo_b = float(lambda_albedo_b)
    lambda_matellic = float(lambda_matellic)
    lambda_roughness = float(lambda_roughness)
    lambda_diffuse = float(lambda_diffuse)
    lambda_specular = float(lambda_specular)
    task_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '-' + scope
    current_dir = os.path.join(cache_dir, task_id)
    os.makedirs(current_dir, exist_ok=True)
    output_video_path = os.path.join(current_dir, 'output_video.mp4')
    output_image_path = os.path.join(current_dir, 'output_video_grid.png')
    print(f"current_dir = {current_dir}")

    n_views, n_rows, n_cols, H, W = (4, 2, 2, 1024, 1024)
    result = renderer.render_pbr(
        index=index,
        lambda_albedo_r=lambda_albedo_r,
        lambda_albedo_g=lambda_albedo_g,
        lambda_albedo_b=lambda_albedo_b,
        lambda_matellic=lambda_matellic,
        lambda_roughness=lambda_roughness,
        lambda_diffuse=lambda_diffuse, 
        lambda_specular=lambda_specular,
    )
    im = result[video_type].clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    im = Image.fromarray(im.reshape(n_rows, n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(n_rows * H, n_cols * W, 3), mode='RGB')
    im.save(output_image_path)
    return output_image_path


def parse_parameters():
    parser = argparse.ArgumentParser("mvdiffusion_gardio")
    parser.add_argument('--host', default="0.0.0.0", type=str)
    parser.add_argument('--port', default=7861, type=int)
    parser.add_argument('--cache_dir', default=f'/tmp/mvdiffusion_renderer/{os.getpid()}', type=str)
    return parser.parse_args()


parsed_args = parse_parameters()
scope = 'renderer'
cache_dir = parsed_args.cache_dir
os.makedirs(cache_dir, exist_ok=True)
exporter = VideoExporter()
renderer = NVDiffRendererPBR()


with gr.Blocks() as gradio:
    gradio_index = gr.Dropdown(
        label="Render Index",
        choices=renderer.index_list,
        value=None,
        interactive=True,
    )
    gradio_render_base = gr.Interface(
        fn=call_render_base,
        inputs=[
            gr.Model3D(
                label="Input Model (glb)",
                display_mode="solid",
                camera_position=[90, 90, 2.5],  # alpha, beta, radius
                interactive=True,
            ),
            gr.Dropdown(
                label="Render Type",
                choices=[
                    'world_position', 
                    'world_normal', 
                ],
                value='world_position',
                interactive=True,
            )
        ],
        outputs=[
            gr.Image(
                label='Output Image',
                image_mode="RGBA",
                format='png',
                type="pil",
                interactive=False,
            ),
            gradio_index,
        ],
        title=f"MVDiffusion: Renderer Demo (Base)",
        description="",
        examples=[[m] for m in glob('gradio_examples_mesh/*/raw_mesh.glb')],
        cache_examples=False,
        flagging_mode='never',
        delete_cache=None,  # (86400, 86400),
    )
    gradio_render_pbr = gr.Interface(
        fn=call_render_pbr,
        inputs=[
            gradio_index,
            gr.Dropdown(
                label="Render Type",
                choices=[
                    'rgb', 
                    'diffuse',
                    'specular',
                ],
                value='rgb',
                interactive=True,
            ),
            gr.Slider(
                label="Albedo-R",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Albedo-G",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Albedo-B",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Matellic",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Roughness",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Diffuse",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
            gr.Slider(
                label="Specular",
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.01,
                interactive=True,
            ),
        ],
        outputs=[
            gr.Image(
                label='Output Image',
                image_mode="RGBA",
                format='png',
                type="pil",
                interactive=False,
            ),
        ],
        title=f"MVDiffusion: Renderer Demo (PBR)",
        description="",
        examples=[],
        cache_examples=False,
        flagging_mode='never',
        delete_cache=None,  # (86400, 86400),
    )
    gradio_button = gr.Button(
        value="Clear Cache",
    ).click(
        fn=lambda: renderer.clear_cache(),
    ).success(
        fn=lambda: gr.update(choices=renderer.index_list),
        outputs=[gradio_index],
    )

gradio.queue(max_size=1)
gradio.launch(
    server_name=parsed_args.host, server_port=parsed_args.port, 
    allowed_paths=[cache_dir],
)


