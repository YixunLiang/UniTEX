# Texture Tools

Core codes of texture generation based on pytorch, take full advantage of GPU acceleration.

## Installation

Check `requirements.txt` and install texturetools from source codes.
``` sh
pip install .
```

## Usage

Before using the classes and functions as follows, you need to fully grasp the 3D related knowledge and review all details and scopes of source codes, and then write the application-layer interface and integrate it into the specific applications as you like.

* **Read and Write**: Accurately read or write triangle mesh vertex attributes, face index, pbr material map from or to ply/obj/glb format file, build-in data type is pytorch tensor.
* **Camera**: Generate internal and external parameters of the camera, mutual conversion between various rotation quantities, coordinate system conversion, projection and its inversion, discretization and its inversion.
* **Rendering**: Rendering of geometry and textures based on rasterization, pbr materials rendering, inverse rendering or backprojection based on rasterization, global seamless backprojection, converting between cubemap and panorama, rendering and inverse rendering of cubemap and panorama.
* **Ray tracing**: Flexible, fast and accurate ray tracing.
* **Watertight**: Fast conversion of non-watertight triangle mesh into high-precision watertight triangle mesh.
* **Geometry calculation**: Remeshing and integration based on normal images, sparse voxel marching cubes, selection of connected components, selection of collision components, surface or spatial sampling with texture on multiple components, selection of sharp edges, geometry image and triangle mesh conversion, UV unwrapping, selection of UV seams on triangle mesh, texture map expansion and mip-texture seamless.
* **Useful tools**: Release memory collected by pytorch, CPU timing, create image grid or video grid, etc.

## Changelogs

* 2024-03-29: rewrite codes.

