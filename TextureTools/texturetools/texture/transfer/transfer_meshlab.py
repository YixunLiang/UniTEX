'''
This scripts is used for reading and writing obj or glb/gltf.

usage:
    src="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh.obj"
    dst="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh_low_poly_with_uv.obj"
    output="/mnt/jfs/wangzihao/Projects/PanoIndoor3DGeneration/examples/koolai_room_1574_64_v2/pano_estimation_debug/layout_mesh_low_poly_with_texture.obj"
    python texturetools/texture/color_transfer/transfer_meshlab.py -- -s ${src} -d ${dst} -o ${output}
'''

import os
import sys
import argparse
import pymeshlab as ml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True)
    parser.add_argument('-d', '--dst', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else '')
    return args


def main(args):
    meshset_ml = ml.MeshSet()
    meshset_ml.load_new_mesh(args.src)
    meshset_ml.load_new_mesh(args.dst)
    meshset_ml.apply_filter(
        'transfer_attributes_to_texture_per_vertex', 
        sourcemesh=0,
        targetmesh=1,
        attributeenum=0,
        textw=2048,
        texth=2048,
        textname=os.path.basename(args.output)+'.png',
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    meshset_ml.save_current_mesh(args.output, save_polygonal=False)
    with open(args.output+'.mtl', 'r+') as f:
        lines = f.readlines()
        lines.append(f'map_Kd {os.path.basename(args.output)}.png')
        f.seek(0, 0)
        f.writelines(lines)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)

