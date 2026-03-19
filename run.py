from pipeline import CustomRGBTextureFullPipeline
import os
rgb_tfp = CustomRGBTextureFullPipeline(super_resolutions=False,
                                        filt_gradient_points=False,
                                        filt_large_angle_points=True,
                                        seed = 63)

test_image_path = "test_cases/teaser_robot/image.png"
test_mesh_path = "test_cases/teaser_robot/inputmesh.obj"
save_root = 'outputs/test'
os.makedirs(save_root, exist_ok=True)
rgb_tfp(save_root, test_image_path, test_mesh_path, clear_cache=False)