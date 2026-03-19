'''
https://github.com/cnr-isti-vclab/meshlab/blob/main/src/meshlabplugins/filter_texture/pushpull.h
'''

import math
import torch


def pull_push_mip(map_Kd:torch.Tensor, map_mask:torch.Tensor):
    '''
    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    B, C, H, W = map_Kd.shape
    map_alpha = map_mask.to(dtype=torch.float32)
    map_alpha_mip = torch.nn.functional.avg_pool2d(map_alpha, 2, 2, 0).permute(0, 2, 3, 1)
    map_Kd_mip = torch.nn.functional.avg_pool2d(map_Kd, 2, 2, 0).permute(0, 2, 3, 1)
    map_mask_mip_boundary = torch.logical_and(map_alpha_mip > 0.0, map_alpha_mip < 1.0)
    map_alpha_mip_boundary = torch.masked_select(map_alpha_mip, map_mask_mip_boundary).reshape(-1, 1)
    map_Kd_mip_boundary = torch.masked_select(map_Kd_mip, map_mask_mip_boundary).reshape(-1, C)
    map_Kd_mip_boundary = torch.div(map_Kd_mip_boundary, map_alpha_mip_boundary)
    map_Kd_mip = torch.masked_scatter(map_Kd_mip, map_mask_mip_boundary, map_Kd_mip_boundary).permute(0, 3, 1, 2)
    map_mask_mip = (map_alpha_mip > 0).permute(0, 3, 1, 2)
    return map_Kd_mip, map_mask_mip


def pull_push_fill(map_Kd:torch.Tensor, map_mask:torch.Tensor, map_Kd_mip:torch.Tensor, map_mask_mip:torch.Tensor, kernel_2x2:torch.Tensor):
    '''
    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    map_Kd_mip: [N, C, H//2, W//2], float32, black background
    map_mask_mip: [N, 1, H//2, W//2], bool
    kernel_2x2: [4, Hk, Wk]
    '''
    B, C, H, W = map_Kd.shape
    B, C, H_mip, W_mip = map_Kd_mip.shape
    G, Hk, Wk = kernel_2x2.shape
    assert H_mip == H // 2 and W_mip == W // 2 and G == 4
    kernel_2x2 = kernel_2x2.unsqueeze(1)  # [4, 1, Hk, Wk]
    map_alpha_mip = map_mask_mip.to(dtype=torch.float32)
    map_alpha_mip_pad = torch.nn.functional.pad(map_alpha_mip, [1, 1, 1, 1], mode='replicate')
    map_Kd_mip_pad = torch.nn.functional.pad(map_Kd_mip, [1, 1, 1, 1], mode='replicate')
    map_alpha_mip_conv = torch.nn.functional.conv2d(map_alpha_mip_pad, kernel_2x2, None, 1, 0, 1, 1)  # [N, 1*4, H//2+1, W//2+1]
    map_Kd_mip_conv = torch.nn.functional.conv2d(map_Kd_mip_pad, kernel_2x2.repeat((C, 1, 1, 1)), None, 1, 0, 1, C)  # [N, C*4, H//2+1, W//2+1]
    map_alpha_mip_conv = map_alpha_mip_conv.reshape(B, 1, 2, 2, H_mip+1, W_mip+1).permute(0, 1, 4, 2, 5, 3).reshape(B, 1, (H_mip+1)*2, (W_mip+1)*2)[:, :, 1:-1, 1:-1]
    map_Kd_mip_conv = map_Kd_mip_conv.reshape(B, C, 2, 2, H_mip+1, W_mip+1).permute(0, 1, 4, 2, 5, 3).reshape(B, C, (H_mip+1)*2, (W_mip+1)*2)[:, :, 1:-1, 1:-1]
    map_Kd = torch.where(map_mask, map_Kd, map_Kd_mip_conv)
    return map_Kd, map_mask


def pull_push(map_Kd:torch.Tensor, map_mask:torch.Tensor):
    '''
    map_Kd: [N, C, H, W], float32
    map_mask: [N, 1, H, W], bool
    '''
    B, C, H, W = map_Kd.shape
    assert map_mask.dtype == torch.bool
    n_mip_level = max(min(int(math.log2(H)), int(math.log2(W))) - 2, 0)
    if n_mip_level == 0:
        return map_Kd, map_mask
    map_Kd = torch.where(map_mask, map_Kd, 0.0)

    # pull phase create the mipmap
    map_Kd_mip_list = []
    map_mask_mip_list = []
    map_Kd_mip, map_mask_mip = map_Kd, map_mask
    for mip_level in range(n_mip_level):
        map_Kd_mip, map_mask_mip = pull_push_mip(map_Kd_mip, map_mask_mip)
        map_Kd_mip_list += [map_Kd_mip]
        map_mask_mip_list += [map_mask_mip]
    
    # push phase: refill
    kernel_2x2 = torch.as_tensor([
        [
            [0.5625, 0.1875],
            [0.1875, 0.0625],
        ],
        [
            [0.1875, 0.5625],
            [0.0625, 0.1875],
        ],
        [
            [0.1875, 0.0625],
            [0.5625, 0.1875],
        ],
        [
            [0.0625, 0.1875],
            [0.1875, 0.5625],
        ],
    ], dtype=map_Kd.dtype, device=map_Kd.device)
    map_Kd_mip, map_mask_mip = map_Kd_mip_list[-1], map_mask_mip_list[-1]
    for mip_level in range(n_mip_level-1, 0, -1):
        map_Kd_up, map_mask_up = map_Kd_mip_list[mip_level-1], map_mask_mip_list[mip_level-1]
        map_Kd_mip, map_mask_mip = pull_push_fill(map_Kd_up, map_mask_up, map_Kd_mip, map_mask_mip, kernel_2x2)
    map_Kd, map_mask = pull_push_fill(map_Kd, map_mask, map_Kd_mip, map_mask_mip, kernel_2x2)
    return map_Kd, map_mask

