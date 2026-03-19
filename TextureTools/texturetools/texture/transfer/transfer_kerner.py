from datetime import datetime
import os
import shutil
import sys
from typing import Optional, Union
import trimesh


BLENDER_PATH = os.environ.get('BLENDER_PATH', os.path.expanduser("~/blender"))
TRANSFER_CACHE = os.environ.get('BLENDER_CACHE', "/tmp/transfer_cache")
PYTHON_PATH = sys.executable


def color_transfer_v_uv(
    src_mesh:Union[str, trimesh.Trimesh], 
    dst_mesh:Union[str, trimesh.Trimesh], 
    output_mesh_path:Optional[str]=None, 
    blender_path=BLENDER_PATH, 
    python_path=PYTHON_PATH
) -> Optional[Union[str, trimesh.Trimesh]]:
    '''
    transfer src v_color to dst uv_color and export as output
    '''
    task_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    cache_dir = os.path.join(TRANSFER_CACHE, task_id)
    os.makedirs(cache_dir, exist_ok=True)
    if isinstance(src_mesh, str):
        src_mesh_path = os.path.abspath(os.path.normpath(src_mesh))
        assert os.path.splitext(src_mesh_path)[1] == '.obj'
    else:
        src_mesh_path = os.path.join(cache_dir, 'src_mesh.obj')
        src_mesh.export(src_mesh_path)
    if isinstance(dst_mesh, str):
        dst_mesh_path = os.path.abspath(os.path.normpath(dst_mesh))
        assert os.path.splitext(dst_mesh_path)[1] == '.obj'
    else:
        dst_mesh_path = os.path.join(cache_dir, 'dst_mesh.obj')
        dst_mesh.export(dst_mesh_path)
    if output_mesh_path is not None:
        output_mesh_path = os.path.abspath(os.path.normpath(output_mesh_path))
        assert os.path.splitext(output_mesh_path)[1] == '.obj'
        os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    blender_path = os.path.abspath(os.path.normpath(blender_path))
    python_path = os.path.abspath(os.path.normpath(python_path))
    cmd_1 = f"{blender_path} --background --python {os.path.join(os.path.dirname(__file__), 'transfer_blender.py')} -- -d {dst_mesh_path} -o {os.path.join(cache_dir, 'mesh_with_uv.obj')}"
    cmd_2 = f"{python_path} {os.path.join(os.path.dirname(__file__), 'transfer_meshlab.py')} -- -s {src_mesh_path} -d {os.path.join(cache_dir, 'mesh_with_uv.obj')} -o {os.path.join(cache_dir, 'mesh_with_texture.obj')}"
    os.system(cmd_1)
    os.system(cmd_2)
    if os.path.isfile(os.path.join(cache_dir, 'mesh_with_texture.obj')):
        if output_mesh_path is not None:
            shutil.copy(os.path.join(cache_dir, 'mesh_with_texture.obj'), output_mesh_path)
            return output_mesh_path
        else:
            output_mesh = trimesh.load(os.path.join(cache_dir, 'mesh_with_texture.obj'), force='mesh', process=False)
            return output_mesh

