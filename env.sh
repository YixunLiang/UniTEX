conda create -n unitex python=3.10 --yes
conda activate unitex
conda install cudatoolkit=11.8 --yes # check nvcc -V, it should be 11.8

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

wget https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu118/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl
pip install kaolin-0.17.0-cp310-cp310-linux_x86_64.whl
rm kaolin-0.17.0-cp310-cp310-linux_x86_64.whl

pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers bitsandbytes==0.45 tokenizers
pip install tqdm ftfy regex safetensors thop opencv-python opencv-contrib-python datetime bypy lpips matplotlib imageio tensorboard
pip install datasets accelerate datetime diffusers einops lpips transformers h5py
pip install calflops
pip install OmegaConf
pip install icecream
pip install blenderproc==2.5.0
pip install objaverse==0.0.7
pip install blender bpy mathutils==3.3.0 boto3==1.26.105
pip install pymcubes==0.1.6 pymeshlab==2023.12.post3 onnxruntime onnxruntime-gpu 
pip install trimesh==3.20.2
pip install pyhocon
pip install deepspeed
pip install git+https://github.com/NVlabs/nvdiffrast.git 
pip install rembg fpsample triton==3.0.0
pip install open3d
pip install xatlas==0.0.10
pip install gpytoolbox==0.3.3
pip install cupy-cuda11x==13.4.1
pip install async-timeout==5.0.1
pip install timeout-decorator==0.5.0
pip install peft==0.15.2
pip install jaxtyping==0.3.1
pip install sentencepiece==0.2.0
pip install timm==0.6.13
pip install kornia==0.8.0
pip install slangtorch==1.3.7
pip install pyexr==0.5.0
pip install git+https://github.com/thomgrand/torch_kdtree.git 
pip install imageio==2.37
pip install imageio-ffmpeg==0.4.7
