'''
This scripts is used for reading and writing obj or glb/gltf.

environments:
    python -m pip install -r requirements.txt
requirements:
    blender < 4.0
usage:
    obj_path=./3d_model/doll.obj
    glb_path=./3d_model/doll.glb
    blender --background --python texturetools/io/blender_scripts.py -- -i ${obj_path} -o ${glb_path}
'''

import os
import sys
# print(sys.exec_prefix)  # /snap/blender/4461/3.6/python
import shutil
import argparse
import math
from time import perf_counter
import numpy as np
import bpy # type: ignore
import bmesh # type: ignore
import mathutils # type: ignore
# print(bpy.app.version)  # (3, 6, 15)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
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


def transform(scale=None, euler=None, translation=None):
    scale_length = bpy.context.scene.unit_settings.scale_length

    # select all objects
    for obj in bpy.data.objects.values():
        obj.select_set(True)

        # set all transforms
        if scale is not None:
            obj.scale = (
                obj.scale[0] * scale[0], 
                obj.scale[1] * scale[1], 
                obj.scale[2] * scale[2],
            )
        if euler is not None:
            # https://docs.blender.org/api/current/bpy_types_enum_items/object_rotation_mode_items.html
            # blender xyz: forward, right, up
            # meshlab xyz: right, up, forward
            obj.rotation_mode = "XYZ"
            obj.rotation_euler.x = obj.rotation_euler.x + scale_length * math.radians(euler[0])
            obj.rotation_euler.y = obj.rotation_euler.y + scale_length * math.radians(euler[1])
            obj.rotation_euler.z = obj.rotation_euler.z + scale_length * math.radians(euler[2])
        if translation is not None:
            obj.location = (
                obj.location[0] + translation[0], 
                obj.location[1] + translation[1], 
                obj.location[2] + translation[2],
            )

    # apply all transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # unselect all objects
    for obj in bpy.data.objects.values():
        obj.select_set(False)


def remove_material():
    for obj in bpy.data.objects.values():
        obj.select_set(True)

        if len(obj.material_slots) > 0:
            for material in obj.material_slots:
                matname = material.name
                matindex = obj.data.materials.find(matname)
                if matname in obj.data.materials:
                    obj.data.materials.pop(index=matindex)

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


def edit_material():
    '''
    create link:
    * from: 
        - Principled BSDF(BSDF_PRINCIPLED) - Base Color
        - Principled BSDF(BSDF_PRINCIPLED) - Emission
        - Emission(EMISSION) - Color
    * to: Material Output(OUTPUT_MATERIAL) - Surface

    NOTE: 
    1. skip complex render graph, support Principled BSDF only;
    2. remove transparent component if alpha < 1.0;
    3. link Base Color or Emission to Surface.
    '''
    for obj in bpy.data.objects.values():
        obj.select_set(True)

        remove_obj = False
        if len(obj.material_slots) > 0:
            for material_slot in obj.material_slots:
                matname = material_slot.name
                material = obj.data.materials.get(matname)
                node = material.node_tree.nodes.get('Principled BSDF', None)
                if node is None:
                    raise ValueError(f'node error: Principled BSDF')
                if node.inputs['Alpha'].default_value < 1.0:
                    remove_obj = True
                    break
                socket = node.inputs['Base Color']
                if len(socket.links) == 0:
                    new_socket = node.inputs['Emission']
                    if len(new_socket.links) == 0:
                        default_value = socket.default_value
                        if default_value == 0.0:
                            default_value = new_socket.default_value
                        new_node = material.node_tree.nodes.new('ShaderNodeRGB')
                        new_node.outputs['Color'].default_value = default_value
                        from_socket = new_node.outputs['Color']
                    elif len(new_socket.links) == 1:
                        from_socket = new_socket.links[0].from_socket
                    else:
                        raise ValueError(f'link_limit error: {new_socket.link_limit}')
                elif len(socket.links) == 1:
                    from_socket = socket.links[0].from_socket
                else:
                    raise ValueError(f'link_limit error: {socket.link_limit}')
                to_socket = material.node_tree.nodes['Material Output'].inputs['Surface'].links[0].to_socket
                material.node_tree.links.new(from_socket, to_socket)

        if remove_obj:
            bpy.ops.object.delete()
        else:
            obj.select_set(False)


if __name__ == '__main__':
    args = parse_args()
    import_mesh(args.input)
    # transform(euler=(0, 0, 90))
    remove_image()
    unwarp_uv()
    export_mesh(args.output)

