# Build Docker Image of MVDiffusion

1. install full [docker](https://docs.docker.com/engine/install/ubuntu/) or `docker.io` only, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. pull base docker image
* pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
3. download packages
* [blender-4.2.3-linux-x64.tar.xz](https://www.blender.org/download/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz)
* [nvdiffrast-729261dc64c4241ea36efda84fbf532cc8b425b8.zip](https://github.com/NVlabs/nvdiffrast/archive/729261dc64c4241ea36efda84fbf532cc8b425b8.zip)
* [pytorch3d-0.7.8.zip](https://github.com/facebookresearch/pytorch3d/releases/tag/V0.7.8)
* [NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh](https://developer.nvidia.com/designworks/optix/downloads/legacy)
* [trimesh-ray-optix-a5f625258e1aae78972344bf82e93366577c656f.zip](https://github.com/lcp29/trimesh-ray-optix/archive/a5f625258e1aae78972344bf82e93366577c656f.zip)
* [pybind11-bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip](https://github.com/pybind/pybind11/archive/bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip)
* [torch_kdtree-86961f7def35a2d818916e343807f1867d0622c0.zip](https://github.com/thomgrand/torch_kdtree/archive/86961f7def35a2d818916e343807f1867d0622c0.zip)
* [PyNanoInstantMeshes-9aaaf584973e6d6d960b04d69cac9bae32f54538.zip](https://github.com/vork/PyNanoInstantMeshes/archive/9aaaf584973e6d6d960b04d69cac9bae32f54538.zip)
* [taming-transformers-master.zip](https://github.com/CompVis/taming-transformers/archive/refs/heads/master.zip)
* [TextureTools-main.zip](https://github.com/lightillusions/TextureTools/archive/refs/heads/main.zip)
* [TextureMan-main.zip](https://github.com/lightillusions/TextureMan/archive/refs/heads/main.zip)
4. build docker image with tag `mvdiffusion/mvdiffusion:v1.0`.

## Appendix

If you do not want to run mvdiffusion in docker container, you can install python environment as follows.

1. (optional) install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), create and activate a clean conda environment with name `mvdiffusion`.
``` sh
conda create -n mvdiffusion python=3.10
conda activate mvdiffusion
```
2. install and check [cuda](https://developer.nvidia.com/cuda-downloads) and [cudnn](https://developer.nvidia.com/rdp/cudnn-download).
3. install all requirements in [Dockerfile](./Dockerfile), refer to [install.sh](./install.sh).

