from datetime import datetime
import os
from typing import Optional, Tuple, Union
import numpy as np
import torch


BLENDER_PATH = os.environ.get('BLENDER_PATH', os.path.expanduser("~/blender"))
BLENDER_CACHE = os.environ.get('BLENDER_CACHE', "/tmp/blender_cache")


def blender_rendering(
    input_mesh_path:str,
    output_dir:str, 
    c2ws:torch.Tensor, 
    intrinsics:torch.Tensor, 
    render_size:Union[int, Tuple[int]],
    perspective:bool=True,
    env_hdr_path:Optional[str]=None,
):
    input_mesh_path = os.path.abspath(os.path.normpath(input_mesh_path))
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    task_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    blender_cache = os.path.join(BLENDER_CACHE, task_id)
    os.makedirs(blender_cache, exist_ok=True)
    c2ws_path = os.path.join(blender_cache, 'c2ws.npy')
    intrinsics_path = os.path.join(blender_cache, 'intrinsics.npy')
    np.save(c2ws_path, c2ws.detach().cpu().numpy())
    np.save(intrinsics_path, intrinsics.detach().cpu().numpy())
    height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
    cmd = f'''
    {BLENDER_PATH} -b --python {os.path.dirname(__file__)}/render_blender.py -- \
        -i {input_mesh_path} \
        -o {output_dir} \
        --c2ws {c2ws_path} \
        --intrinsics {intrinsics_path} \
        --height {height} \
        --width {width} \
    '''
    if perspective:
        cmd += '--perspective \\\n'
    if env_hdr_path is not None:
        cmd += f'--env_hdr_path {env_hdr_path} \\\n'
    return os.system(cmd)

