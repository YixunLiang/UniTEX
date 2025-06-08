from pipeline import CustomRGBTextureFullPipeline
import os
rgb_tfp = CustomRGBTextureFullPipeline(pretrain_models='/mnt/jfs/kunming/projects/pret_models/textureman',
                                        super_resolutions=False,
                                        seed = 63)

test_image_path = "test_cases/teaser_robot/image.png"
test_mesh_path = "test_cases/gamda_style/inputmesh2.glb"
save_root = 'outputs/test'
os.makedirs(save_root, exist_ok=True)
rgb_tfp(save_root, test_image_path, test_mesh_path, clear_cache=False)