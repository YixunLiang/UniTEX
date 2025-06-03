import torch

def compute_edge_length(v_pos:torch.Tensor, t_pos_idx:torch.Tensor) -> torch.Tensor:
    e_pos_idx_full = torch.cat([t_pos_idx[:, [0, 1]], t_pos_idx[:, [1, 2]], t_pos_idx[:, [2, 0]]], dim=0)
    e_pos_idx_sorted = torch.sort(e_pos_idx_full, dim=-1).values
    e_pos_idx = torch.unique(e_pos_idx_sorted, dim=0, sorted=False, return_inverse=False, return_counts=False)
    e_pos = v_pos[e_pos_idx]
    e_dis = torch.norm(e_pos[:, 1, :] - e_pos[:, 0, :], dim=-1)
    return e_dis
