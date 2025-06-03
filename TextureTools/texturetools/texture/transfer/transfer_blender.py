'''
This scripts is used for reading and writing obj or glb/gltf.

environments:
    python -m pip install -r requirements.txt
requirements:
    blender < 4.0
usage:
    src="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh.obj"
    dst="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh_low_poly.obj"
    output="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh_low_poly_with_uv.obj"
    ${HOME}/blender --background --python texturetools/texture/color_transfer/transfer_blender.py -- -d ${dst} -o ${output}
'''

import os
import sys
# print(sys.exec_prefix)  # /snap/blender/4461/3.6/python
import argparse
import bpy # type: ignore
import bmesh # type: ignore
# print(bpy.app.version)  # (3, 6, 15)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--src', type=str, required=True)
    parser.add_argument('-d', '--dst', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else '')
    return args


def main(args):
    for k in ['Camera', 'Cube', 'Light']:
        obj = bpy.data.objects.get(k, None)
        if obj is not None:
            obj.select_set(True)
            bpy.ops.object.delete()
    bpy.ops.wm.obj_import(filepath=args.dst)
    dst = bpy.data.objects[list(bpy.data.objects.keys())[0]]
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
            bpy.ops.uv.smart_project()
            bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode="OBJECT")
        if not obj.data.materials:
            obj.data.materials.append(bpy.data.materials.new(name="EmptyMaterial"))
        obj.select_set(False)
    dst.select_set(True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bpy.ops.wm.obj_export(filepath=args.output, path_mode="COPY")



if __name__ == '__main__':
    args = parse_args()
    main(args)

