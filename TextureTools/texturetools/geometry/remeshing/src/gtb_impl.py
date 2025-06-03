'''
gpytoolbox implementation
https://github.com/sgsellan/gpytoolbox
'''
from typing import Tuple
import numpy as np
import torch
import gpytoolbox as gtb

from .utils import compute_edge_length

def remesh(
    v_pos:torch.Tensor,
    t_pos_idx:torch.Tensor,
    scale=1.0,
    n_steps=1,
    project=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    e_dis = compute_edge_length(v_pos, t_pos_idx)
    e_dis_mean = e_dis.mean().item()
    v_pos, t_pos_idx = gtb.remesh_botsch(
        V=v_pos.detach().cpu().numpy().astype(np.float64),
        F=t_pos_idx.detach().cpu().numpy().astype(np.int32),
        i=n_steps,
        h=e_dis_mean * scale,
        project=project,
    )
    v_pos = torch.as_tensor(v_pos, dtype=v_pos.dtype, device=v_pos.device)
    t_pos_idx = torch.as_tensor(t_pos_idx, dtype=t_pos_idx.dtype, device=t_pos_idx.device)
    return v_pos, t_pos_idx

