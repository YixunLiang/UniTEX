import os


BLENDER_PATH = os.environ.get('BLENDER_PATH', os.path.expanduser("~/blender"))
BLENDER_CACHE = os.environ.get('BLENDER_CACHE', "/tmp/blender_cache")


def preprocess_blank_mesh(input_obj_path, output_obj_path, min_faces=20_000, max_faces=200_000, scale=1.0):
    input_obj_path = os.path.normpath(input_obj_path)
    output_obj_path = os.path.normpath(output_obj_path)
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    cmd = f'{BLENDER_PATH} --background --python {os.path.dirname(__file__)}/uv_blender.py -- -i {input_obj_path} -o {output_obj_path}'
    return os.system(cmd)

