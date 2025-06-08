# UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes

[Yixun Liang](https://yixunliang.github.io/)$^{\*}$, [Kunming Luo](https://coolbeam.github.io)$^{{\*}}$, [Xiao Chen]()$^{{*}}$, [Rui Chen](https://aruichen.github.io), [Hongyu Yan](https://scholar.google.com/citations?user=TeKnXhkAAAAJ&hl=zh-CN), [Weiyu Li](https://weiyuli.xyz),[Jiarui Liu](), [Ping Tan](https://pingtan.people.ust.hk/index.html)$†$

$\*$: Equal contribution. $†$: Corrsponding author.


<a href="https://arxiv.org/abs/2505.23253"><img src="https://img.shields.io/badge/ArXiv-2505.23253-brightgreen"></a> 
---
## :tv: Video
</div>
<div align=center>

[![UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes](assets/unitex_demo.jpg)](https://youtu.be/O8G1XqfIxck "UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes")

Please click to watch the 3-minute video introduction of our project.

</div>

## 🎏 Abstract

We present a 2 stage texturing framework, named the *UniTEX*, to achieve high-fidelity textures from any 3D shapes.

<details><summary>CLICK for the full abstract</summary>

> We present UniTEX, a novel two-stage 3D texture generation framework to create high-quality, consistent textures for 3D assets.
Existing approaches predominantly rely on UV-based inpainting to refine textures after reprojecting the generated multi-view images onto the 3D shapes, which introduces challenges related to topological ambiguity. To address this, we propose to bypass the limitations of UV mapping by operating directly in a unified 3D functional space. Specifically, we first propose a novel framework that lifts texture generation into 3D space via Texture Functions (TFs)—a continuous, volumetric representation that maps any 3D point to a texture value based solely on surface proximity, independent of mesh topology. Then, we propose to predict these TFs directly from images and geometry inputs using a transformer-based Large Texturing Model (LTM). To further enhance texture quality and leverage powerful 2D priors, we develop an advanced LoRA-based strategy for efficiently adapting large-scale Diffusion Transformers (DiTs) for high-quality multi-view texture synthesis as our first stage. Extensive experiments demonstrate that UniTEX achieves superior visual quality and texture integrity compared to existing approaches, offering a generalizable and scalable solution for automated 3D texture generation.

</details>

<div align=center>
<img src="assets/pipeline.png" width="95%"/>  
</div>

## 🚧 Todo

- [x] Release the basic texturing codes with flux lora checkpoints
- [x] Release the training code of flux (lora) ([UniTEX-FLUX](https://github.com/YixunLiang/UniTEX-FLUX))
- [ ] Release LTM checkpoints [after paper accepted]

**Note** Our framework filters out the geometry edge and some conflicting points and uses LTM to inpaint them. Therefore, the current results without LTM may contain more artifacts compared to those presented in the paper. we will release full pipeline after paper is accpeted. 
## 🔧 Installation

run  ```bash env.sh``` to prepare your environment.

**Note** We noticed that some users encountered errors when using slangtorch==1.3.7. If you encounter the same issue, you can try reinstalling slangtorch==1.3.4, which should resolve the problem. (Check [This issue](https://github.com/YixunLiang/UniTEX/issues/2#issuecomment-2939913090), it also about how to use our repo under cu121, thanks to [HwanHeo](https://github.com/hwanhuh))

Download [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) and [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/tree/main) and the checkpoints of our LoRA in [UniTex](https://huggingface.co/coolbeam/UniTex) from Hugging Face. and prepare your ``pretrain_models folder`` following bellow structure:
```
{pretrain_models_root}
├──black-forest-labs
    ├── FLUX.1-dev
    ├── FLUX.1-Redux-dev
    └── ...  
├──UniTex
    ├── delight
    └── texture_gen
...                
```

and then replace the Ln 3 in ''run.py'' as:
```
rgb_tfp = CustomRGBTextureFullPipeline(pretrain_models={pretrain_models_root},
                                        super_resolutions=False,
                                        seed = 63)        
```
## How to use？
Run the following code after your prepared the lora weights and set the corresponding ``dir`` in ``pretrain_models``:
```
from pipeline import CustomRGBTextureFullPipeline
import os
rgb_tfp = CustomRGBTextureFullPipeline(pretrain_models={pretrain_models_root},
                                        super_resolutions=False,
                                        seed = 63)

test_image_path = {your reference image}
test_mesh_path = {your input mesh}
save_root = 'outputs/{your save folder}'
os.makedirs(save_root, exist_ok=True)
rgb_tfp(save_root, test_image_path, test_mesh_path, clear_cache=False)
```

you can also use
```
python run.py
```
to run our given example.

SR:
if you want to use super_resolutions, prepare the ckpts of SR model [TSD_SR](https://github.com/Microtreei/TSD-SR) 
and change the default dir in `` TSD_SR/sr_pipeline.py ln 30-32``
```
parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers/", help='path to the pretrained sd3')
parser.add_argument("--lora_dir", type=str, default="your_lora_dir", help='path to tsd-sr lora weights')
parser.add_argument("--embedding_dir", type=str, default="your_emb_dir", help='path to prompt embeddings')
```

Then, tune ``super_resolutions``  in ``run.py`` to true.

## Training your own FLUX lora
We also provide training code for texture generation and de-lighting, which can be adapted for other tasks as well. Please refer to ([UniTEX-FLUX](https://github.com/YixunLiang/UniTEX-FLUX)) for more details.

## 📍 Citation 
If you find this project useful for your research, please cite: 

```
@article{liang2025UnitTEX,
  title={UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes},
  author={Yixun Liang and Kunming Luo and Xiao Chen and Rui Chen and Hongyu Yan and Weiyu Li and Jiarui Liu and Ping Tan},
  journal={arXiv preprint arXiv:2505.23253},
  year={2025}
}
```
## 7. Acknowledgments
We would like to thank the following projects: [FLUX](https://github.com/black-forest-labs/flux), [DINOv2](https://github.com/facebookresearch/dinov2), [CLAY](https://arxiv.org/abs/2406.13897), [Michelango](https://github.com/NeuralCarver/Michelangelo), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), [TripoSG](https://github.com/VAST-AI-Research/TripoSG), [Dora](https://github.com/Seed3D/Dora), [Hunyuan3D 2.0](https://github.com/Tencent/Hunyuan3D-2), [TSD_SR](https://github.com/Microtreei/TSD-SR),[Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) for their open exploration and contributions. We would also like to express our gratitude to the closed-source 3D generative platforms [Tripo](https://www.tripo3d.ai/), [Rodin](https://hyper3d.ai/), and [Hunyuan2.5](https://3d.hunyuan.tencent.com/) for providing such impressive geometry resources to the community. We sincerely appreciate their efforts and contributions.
