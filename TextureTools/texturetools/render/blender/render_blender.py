'''
This scripts is used for reading and writing obj or glb/gltf.

environments:
    python -m pip install -r requirements.txt
requirements:
    blender < 4.0
usage:
    blender_path=${HOME}/blender
    input_mesh_path="gradio_examples_mesh/0941f2938b3147c7bfe16ec49d3ac500/raw_mesh.glb"
    env_hdr_path="texturetools/renderers/envmaps/lilienstein_1k.hdr"
    blender_state_path="debug.blend"
    output_dir="test_result/test_renderer/blender/0941f2938b3147c7bfe16ec49d3ac500"
    ${blender_path} -b --python texturetools/renderers/blender/blender_scripts.py -- \
        -i ${input_mesh_path} \
        -o ${output_dir} \
        --c2ws camera_info/c2ws_ortho_views_8.npy \
        --intrinsics camera_info/intrinsics_persp_views_8.npy \
        --height 1024 \
        --width 1024 \
        --perspective \
        --env_hdr_path ${env_hdr_path} \
        --blender_state_path ${blender_state_path} \
'''

import os
import sys
# print(sys.exec_prefix)  # /snap/blender/4461/3.6/python
import shutil
import argparse
import math
from time import perf_counter
from typing import Union, Tuple
import numpy as np
import bpy # type: ignore
import bmesh # type: ignore
import mathutils # type: ignore
# print(bpy.app.version)  # (3, 6, 15)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mesh_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--c2ws', type=str, required=True)
    parser.add_argument('--intrinsics', type=str, required=True)
    parser.add_argument('-H', '--height', type=int, required=True)
    parser.add_argument('-W', '--width', type=int, required=True)
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--env_hdr_path', type=str, default=None, required=False)
    parser.add_argument('--blender_state_path', type=str, default=None, required=False)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else '')
    return args


def import_mesh(input_mesh_path):
    # sanity check
    input_mesh_ext = os.path.splitext(input_mesh_path)[1]
    assert input_mesh_ext in ['.glb', '.gltf', '.obj'], \
        f'support [.glb, .gltf, .obj] ext only, but input ext is {input_mesh_ext}'

    # delete Camera/Cube/Light
    for k in ['Camera', 'Cube', 'Light']:
        obj = bpy.data.objects.get(k, None)
        if obj is not None:
            obj.select_set(True)
            bpy.ops.object.delete()

    # import scene
    # https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.obj_import
    # https://docs.blender.org/api/current/bpy.ops.import_scene.html#bpy.ops.import_scene.gltf
    if input_mesh_ext == '.obj':
        bpy.ops.wm.obj_import(
            filepath=input_mesh_path,
        )
    elif input_mesh_ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(
            filepath=input_mesh_path,
        )
    else:
        raise NotImplementedError

    # unselect all objects
    for obj in bpy.data.objects.values():
        obj.select_set(False)


def export_mesh(output_mesh_path):
    # sanity check
    output_mesh_ext = os.path.splitext(output_mesh_path)[1]
    assert output_mesh_ext in ['.glb', '.gltf', '.obj'], \
        f'support [.glb, .gltf, .obj] ext only, but output ext is {output_mesh_ext}'

    # delete Camera/Cube/Light
    for k in ['Camera', 'Cube', 'Light']:
        obj = bpy.data.objects.get(k, None)
        if obj is not None:
            obj.select_set(True)
            bpy.ops.object.delete()

    # select all objects
    for obj in bpy.data.objects.values():
        obj.select_set(True)

    # export scene
    # https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.obj_export
    # https://docs.blender.org/api/current/bpy.ops.export_scene.html#bpy.ops.export_scene.gltf
    if output_mesh_ext == '.obj':
        bpy.ops.wm.obj_export(
            filepath=output_mesh_path,
            path_mode="COPY",
        )
    elif output_mesh_ext in ['.glb', '.gltf']:
        merge_normal()
        bpy.ops.export_scene.gltf(
            filepath=output_mesh_path,
        )
    else:
        raise NotImplementedError

    # delete all objects
    bpy.ops.object.delete()


def apply_transform():
    # select all objects
    for obj in bpy.data.objects.values():
        obj.select_set(True)

    # apply all transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # unselect all objects
    for obj in bpy.data.objects.values():
        obj.select_set(False)


def add_transform(transform, after=True):
    # select all objects
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        obj.select_set(True)
        if after:
            obj.matrix_world = mathutils.Matrix(np.matmul(transform, np.asarray(obj.matrix_world)))
        else:
            obj.matrix_world = mathutils.Matrix(np.matmul(np.asarray(obj.matrix_world), transform))

    # unselect all objects
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        obj.select_set(False)


def add_init_transform():
    scale_length = bpy.context.scene.unit_settings.scale_length

    # select all objects
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        obj.select_set(True)

        # NOTE: default mesh rotation_euler in blender is [-90, 0, 0], 
        # but default mesh rotation_euler in bpy is [0, 0, 0].
        obj.rotation_mode = "XYZ"
        obj.rotation_euler.x = obj.rotation_euler.x + scale_length * math.radians(-90)

    # unselect all objects
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        obj.select_set(False)


def remove_image():
    for img in bpy.data.images:
        bpy.data.images.remove(img)


def merge_normal():
    active_obj = bpy.context.view_layer.objects.active
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        obj.select_set(True)
        bpy.ops.mesh.select_all(action='SELECT')

        bpy.ops.mesh.merge_normals()

        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
    bpy.context.view_layer.objects.active = active_obj


def unwarp_uv():
    active_obj = bpy.context.view_layer.objects.active
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        obj.select_set(True)
        bm = bmesh.from_edit_mesh(obj.data)
        uv = bm.loops.layers.uv
        if len(uv) == 0:
            bpy.ops.mesh.select_all(action='SELECT')

            print(f"UV unwrapping, V={len(bm.verts)}, F={len(bm.faces)}, may take a while ...")
            t = perf_counter()
            bpy.ops.uv.smart_project()
            print(f"UV unwrapping wastes {perf_counter() - t} sec")
            
            bpy.ops.mesh.select_all(action='DESELECT')
            # bpy.ops.mesh.uv_texture_add()
            # bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode="OBJECT")
        if not obj.data.materials:
            obj.data.materials.append(bpy.data.materials.new(name="EmptyMaterial"))
        obj.select_set(False)
    bpy.context.view_layer.objects.active = active_obj


def intrinsic_to_camera_data(camera_data, intrinsic:np.ndarray, perspective=True):
    '''
    intrinsic: [3, 3]
    '''
    f_x, f_y, c_x, c_y = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    if perspective:
        fov_rad = 2 * math.atan(1 / (2 * f_y))
        camera_data.type = "PERSP"
        camera_data.lens_unit = 'FOV'
        # NOTE: do not use angle_x, angle_y here
        # NOTE: when H != W, nvdiffrast will resize image, blender will crop and pad image
        camera_data.angle = fov_rad
    else:
        # NOTE: [0, 1] to [-1, 1]
        scale = 2.0 / f_y
        camera_data.type = "ORTHO"
        # NOTE: when H != W, nvdiffrast will resize image, blender will crop and pad image
        camera_data.ortho_scale = scale
    # NOTE: [0, 1] to [-1, 1]
    camera_data.shift_x = c_x * 2.0 - 1.0
    camera_data.shift_x = c_y * 2.0 - 1.0
    return camera_data


def camera_data_to_intrinsic(camera_data):
    '''
    intrinsic: [3, 3]
    '''
    if camera_data.type == "PERSP":
        perspective = True
        camera_data.lens_unit = 'FOV'
        fov_rad = float(camera_data.angle)
        f_x = f_y = 1.0 / (2.0 * math.tan(fov_rad / 2))
    elif camera_data.type == "ORTHO":
        perspective = False
        scale = float(camera_data.ortho_scale)
        # NOTE: [-1, 1] to [0, 1]
        f_x = f_y = 2.0 / scale
    else:
        raise NotImplementedError(f'camera_data.type {camera_data.type} is not supported')
    # NOTE: [-1, 1] to [0, 1]
    c_x = float(camera_data.shift_x) * 0.5 + 0.5
    c_y = float(camera_data.shift_y) * 0.5 + 0.5
    intrinsic = np.asarray([
        [f_x, 0.0, c_x],
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return intrinsic, perspective


def c2w_to_camera(camera, c2w:np.ndarray):
    '''
    c2w: [4, 4]
    '''
    camera.matrix_world = mathutils.Matrix(c2w)
    return camera


def camera_to_c2w(camera):
    '''
    c2w: [4, 4]
    '''
    c2w = np.asarray(camera.matrix_world, dtype=np.float32)
    return c2w


def c2ws_intrinsics_to_cameras(c2ws:np.ndarray, intrinsics:np.ndarray, perspective=True):
    batch_shape = np.broadcast_shapes(c2ws.shape[:-2], intrinsics.shape[:-2])
    c2ws = np.broadcast_to(c2ws, batch_shape+(4,4))
    intrinsics = np.broadcast_to(intrinsics, batch_shape+(3,3))
    cameras = []
    for idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
        # https://docs.blender.org/api/current/bpy.types.Camera.html
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_data = intrinsic_to_camera_data(camera_data, intrinsic, perspective=perspective)
        camera = bpy.data.objects.new(f'Camera-{idx:04d}', camera_data)
        camera = c2w_to_camera(camera, c2w)
        # NOTE: link camera to collection objects
        # * Object can't be selected because it is not in View Layer
        # * ViewLayer does not contain object
        bpy.context.collection.objects.link(camera)
        cameras.append(camera)
    return cameras


def compute_bbox():
    bbox_list = []
    for obj in bpy.data.objects.values():
        if obj.type != 'MESH':
            continue
        bbox = np.asarray(obj.bound_box)  # [8, 3]
        bbox = np.stack([bbox[0], bbox[-2]], axis=0)  # [2, 3]
        bbox_list.append(bbox)
    bbox_list = np.stack(bbox_list, axis=0)  # [N, 2, 3]
    bbox = np.stack([bbox_list[:, 0, :].min(axis=0), bbox_list[:, 1, :].max(axis=0)], axis=0)  # [2, 3]
    return bbox


def normalize_scene(largest=True):
    # bake transform to object
    add_init_transform()
    apply_transform()

    # compute bbox and add transform
    bbox = compute_bbox()
    bbox_center = bbox.mean(axis=0)
    bbox_half_length = (bbox[1, :] - bbox[0, :]) / 2.0
    scale = bbox_half_length.max() if largest else bbox_half_length.min()
    transform = np.eye(4)
    transform[[0, 1, 2], [0, 1, 2]] = 1 / scale
    transform[:3, 3] = - bbox_center / scale
    add_transform(transform)

    # bake transform to object
    apply_transform()


def set_env_hdr(env_hdr_path):
    '''
    Texture Coordinate -> Mapping -> Environment Texture -> Background -> World Output
    (ShaderNodeTexCoord) (ShaderNodeMapping) (ShaderNodeTexEnvironment) (ShaderNodeBackground) (ShaderNodeOutputWorld)
    '''
    env_hdr_path = os.path.abspath(os.path.normpath(env_hdr_path))
    node_tree = bpy.context.scene.world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    environment_texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = bpy.data.images.load(env_hdr_path)
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.inputs['Rotation'].default_value = mathutils.Euler((math.radians(90.0), 0.0, 0.0), 'XYZ')
    nodes.new(type="ShaderNodeTexCoord")
    links.new(nodes["Texture Coordinate"].outputs[0], nodes["Mapping"].inputs[0])
    links.new(nodes["Mapping"].outputs[0], nodes["Environment Texture"].inputs[0])
    links.new(nodes["Environment Texture"].outputs[0], nodes["Background"].inputs[0])
    links.new(nodes["Background"].outputs[0], nodes["World Output"].inputs[0])


def save_blender_state(blender_state_path):
    blender_state_path = os.path.abspath(os.path.normpath(blender_state_path))
    os.makedirs(os.path.dirname(blender_state_path), exist_ok=True)
    bpy.ops.wm.save_mainfile(filepath=blender_state_path)


def blender_rendering(output_dir:str, render_size:Union[int, Tuple[int]]):
    height, width = (render_size, render_size) if isinstance(render_size, int) else render_size
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = True
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    idx = 0
    for obj in bpy.context.collection.objects:
        if obj.type == 'CAMERA':
            bpy.context.scene.camera = obj
            bpy.ops.render.render()
            image = bpy.data.images["Render Result"]
            image.save_render(os.path.join(output_dir, f"{idx:04d}_rgb.png"))
            idx += 1


if __name__ == '__main__':
    args = parse_args()
    import_mesh(input_mesh_path=args.input_mesh_path)
    normalize_scene()
    if args.env_hdr_path is not None:
        set_env_hdr(env_hdr_path=args.env_hdr_path)
    c2ws = np.load(args.c2ws)
    intrinsics = np.load(args.intrinsics)
    c2ws_intrinsics_to_cameras(c2ws=c2ws, intrinsics=intrinsics, perspective=args.perspective)
    if args.blender_state_path is not None:
        save_blender_state(args.blender_state_path)
    blender_rendering(output_dir=args.output_dir, render_size=(args.height, args.width))

