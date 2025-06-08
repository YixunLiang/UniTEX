from glob import glob
import os
import cv2
import imageio
from tqdm import tqdm


#### SDXL ####

H, W = 2*1024, 2*1024
output_dir = 'geometry_to_image_controlnet_v22'

# input_image_path_list = glob(os.path.join(output_dir, 'validation_result/*/*/*.jpg'))
# input_image_path_list = [(p, int(os.path.basename(os.path.dirname(p)))) for p in input_image_path_list]
# input_image_path_list = sorted(input_image_path_list, key=lambda x:x[1])
# input_image_list = [
#     cv2.putText(
#         cv2.resize(
#             cv2.imread(p), 
#             (W, H), 
#             interpolation=cv2.INTER_LINEAR_EXACT,
#         ), 
#         f'step: {idx:04d}', 
#         [0, 20], 0, 1, [255, 0, 255], 2,
#     ) for p, idx in tqdm(input_image_path_list)
# ]

# output_video_path = os.path.join(output_dir, 'train_result.mp4')
# imageio.mimsave(output_video_path, input_image_list)



#### FLUX ####

H, W = 2*1024, 2*1024
output_dir = 'geometry_to_image_flux_v3'


input_image_path_list = glob(os.path.join(output_dir, 'validation_result/*_prompt_0_res.png'))
input_image_path_list = [(p, int(os.path.basename(p).split('_')[0])) for p in input_image_path_list]
input_image_path_list = sorted(input_image_path_list, key=lambda x:x[1])
input_image_list = [
    cv2.putText(
        cv2.resize(
            cv2.imread(p), 
            (W, H), 
            interpolation=cv2.INTER_LINEAR_EXACT,
        ), 
        f'step: {idx:04d}', 
        [0, 20], 0, 1, [255, 0, 255], 2,
    ) for p, idx in tqdm(input_image_path_list)
]

output_video_path = os.path.join(output_dir, 'train_result_0.mp4')
imageio.mimsave(output_video_path, input_image_list)



input_image_path_list = glob(os.path.join(output_dir, 'validation_result/*_prompt_1_res.png'))
input_image_path_list = [(p, int(os.path.basename(p).split('_')[0])) for p in input_image_path_list]
input_image_path_list = sorted(input_image_path_list, key=lambda x:x[1])
input_image_list = [
    cv2.putText(
        cv2.resize(
            cv2.imread(p), 
            (W, H), 
            interpolation=cv2.INTER_LINEAR_EXACT,
        ), 
        f'step: {idx:04d}', 
        [0, 20], 0, 1, [255, 0, 255], 2,
    ) for p, idx in tqdm(input_image_path_list)
]

output_video_path = os.path.join(output_dir, 'train_result_1.mp4')
imageio.mimsave(output_video_path, input_image_list)



