FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    zip \
    unzip \
    libsystemd0 \
    libudev1 \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    xorg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install torch==2.5.1 -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --no-cache-dir
RUN pip install -U diffusers transformers datasets accelerate peft gradio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple  --no-cache-dir

COPY blender-4.2.3-linux-x64.tar.xz blender-4.2.3-linux-x64.tar.xz
RUN tar -xvf blender-4.2.3-linux-x64.tar.xz
ENV PATH=${PATH}:/workspace/blender-4.2.3-linux-x64

COPY nvdiffrast-729261dc64c4241ea36efda84fbf532cc8b425b8.zip nvdiffrast-729261dc64c4241ea36efda84fbf532cc8b425b8.zip
RUN unzip nvdiffrast-729261dc64c4241ea36efda84fbf532cc8b425b8.zip \
    && mv nvdiffrast-729261dc64c4241ea36efda84fbf532cc8b425b8 nvdiffrast \
    && cd nvdiffrast && pip install . && cd ..

COPY pytorch3d-0.7.8.zip pytorch3d-0.7.8.zip
RUN unzip pytorch3d-0.7.8.zip \
    && cd pytorch3d-0.7.8 && pip install . && cd ..

COPY NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
COPY trimesh-ray-optix-a5f625258e1aae78972344bf82e93366577c656f.zip trimesh-ray-optix-a5f625258e1aae78972344bf82e93366577c656f.zip
RUN unzip trimesh-ray-optix-a5f625258e1aae78972344bf82e93366577c656f.zip \
    && mv trimesh-ray-optix-a5f625258e1aae78972344bf82e93366577c656f trimesh-ray-optix \
    && bash NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh \
    && cd trimesh-ray-optix && pip install . && cd ..
ENV OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64

COPY pybind11-bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip pybind11-bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip
COPY torch_kdtree-86961f7def35a2d818916e343807f1867d0622c0.zip torch_kdtree-86961f7def35a2d818916e343807f1867d0622c0.zip
RUN unzip pybind11-bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip \
    && unzip torch_kdtree-86961f7def35a2d818916e343807f1867d0622c0.zip \
    && mv torch_kdtree-86961f7def35a2d818916e343807f1867d0622c0 torch_kdtree \
    && mv pybind11-bb05e0810b87e74709d9f4c4545f1f57a1b386f5 pybind11 \
    && mv pybind11 torch_kdtree \
    && cd torch_kdtree && pip install . && cd ..

COPY PyNanoInstantMeshes-9aaaf584973e6d6d960b04d69cac9bae32f54538.zip PyNanoInstantMeshes-9aaaf584973e6d6d960b04d69cac9bae32f54538.zip
RUN unzip PyNanoInstantMeshes-9aaaf584973e6d6d960b04d69cac9bae32f54538.zip \
    && mv PyNanoInstantMeshes-9aaaf584973e6d6d960b04d69cac9bae32f54538 PyNanoInstantMeshes \
    && cd PyNanoInstantMeshes && pip install . && cd ..

COPY taming-transformers-master.zip taming-transformers-master.zip
RUN unzip taming-transformers-master.zip \
    && cd taming-transformers-master && python setup.py install && cd ..

COPY TextureTools-main.zip TextureTools-main.zip
RUN unzip TextureTools-main.zip \
    && cd TextureTools-main && pip install . && cd ..

COPY TextureMan-main.zip TextureMan-main.zip
RUN unzip TextureMan-main.zip \
    && cd TextureMan-main && pip install . && cd ..
