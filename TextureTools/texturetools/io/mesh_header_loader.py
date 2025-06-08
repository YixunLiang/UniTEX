import json
import os
import numpy as np
import trimesh


# magic numbers which have meaning in GLTF
# most are uint32's of UTF-8 text
_magic = {"gltf": 1179937895, "json": 1313821514, "bin": 5130562}


def load_mesh_header(mesh_path):
    ext = os.path.splitext(mesh_path)[1]

    if ext == '.glb':
        with open(mesh_path, 'rb') as file_obj:
            # read the first 20 bytes which contain section lengths
            head_data = file_obj.read(20)
            head = np.frombuffer(head_data, dtype="<u4")

            # check to make sure first index is gltf magic header
            if head[0] != _magic["gltf"]:
                raise ValueError("incorrect header on GLB file")

            # and second value is version: should be 2 for GLTF 2.0
            if head[1] != 2:
                raise NotImplementedError(f"only GLTF 2 is supported not `{head[1]}`")

            # overall file length
            # first chunk length
            # first chunk type
            length, chunk_length, chunk_type = head[2:]

            # first chunk should be JSON header
            if chunk_type != _magic["json"]:
                raise ValueError("no initial JSON header!")

            # uint32 causes an error in read, so we convert to native int
            # for the length passed to read, for the JSON header
            json_data = file_obj.read(int(chunk_length))
            # convert to text
            if hasattr(json_data, "decode"):
                json_data = trimesh.util.decode_text(json_data)
            # load the json header to native dict
            header = json.loads(json_data)

    elif ext == '.gltf':
        with open(mesh_path, 'r') as file_obj:
            json_data = file_obj.read()
            json_data = trimesh.util.decode_text(json_data)
            header = json.loads(json_data)
            if 'buffers' in header.keys():
                header.pop('buffers')

    else:
        # raise NotImplementedError(f'ext {ext} is not supported now')
        return {'meshes': []}

    return header


def parse_mesh_info(mesh_path):
    vl = fl = 0
    h = load_mesh_header(mesh_path)
    nc = len(h['meshes'])
    nm = len(h['materials']) if 'materials' in h.keys() else 0
    for i in range(len(h['meshes'])):
        for j in range(len(h['meshes'][i]['primitives'])):
            vi = h['meshes'][i]['primitives'][j]['attributes']['POSITION']
            fi = h['meshes'][i]['primitives'][j]['indices']
            vl += h['accessors'][vi]['count']
            fl += h['accessors'][fi]['count']
    return {
        'V': vl,
        'F': fl // 3,  # triangle indices
        'NC': nc,
        'NM': nm,
    }


