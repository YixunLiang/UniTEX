import math
import os
import pygltflib
import torch
import trimesh


def dump_glb(vertices:torch.Tensor, faces:torch.Tensor, output_path:str, backend=1):
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    '''
    if backend == 0:
        vertices = vertices.detach().cpu().numpy()
        faces = faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
    elif backend == 1:
        triangles_binary_blob = faces.cpu().numpy().astype(np.int32).flatten().tobytes()
        triangles_count = math.prod(faces.shape)
        triangles_min = [faces.min().item()]
        triangles_max = [faces.max().item()]
        points_binary_blob = vertices.detach().cpu().numpy().astype(np.float32).tobytes()
        points_count = vertices.shape[0]
        points_min = vertices.min(dim=0).values.tolist()
        points_max = vertices.max(dim=0).values.tolist()
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(POSITION=1), indices=0
                        )
                    ]
                )
            ],
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,
                    count=triangles_count,
                    type=pygltflib.SCALAR,
                    max=triangles_max,
                    min=triangles_min,
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=points_count,
                    type=pygltflib.VEC3,
                    max=points_max,
                    min=points_min,
                ),
            ],
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(triangles_binary_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(triangles_binary_blob),
                    byteLength=len(points_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(triangles_binary_blob) + len(points_binary_blob)
                )
            ],
        )
        gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gltf.save(output_path)
    else:
        raise NotImplementedError(f'backend {backend} is not supported')


