import torch


def dilate_vertex(f_v_idx:torch.Tensor, v_mask:torch.Tensor):
    ...  # TODO


def dilate_edge(f_v_idx:torch.Tensor, e_mask:torch.Tensor):
    ...  # TODO


def dilate_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    if depth <= 0:
        return f_mask
    v_value = torch.zeros((V,), dtype=torch.int64, device=f_v_idx.device)
    f_ones = torch.ones((f_v_idx.shape[0],), dtype=torch.int64, device=f_v_idx.device)
    f_mask_v_idx = torch.masked_select(f_v_idx, f_mask.unsqueeze(-1)).reshape(-1, 3)
    v_value = v_value.scatter_add(0, f_mask_v_idx[:, 0], f_ones).scatter_add(0, f_mask_v_idx[:, 1], f_ones).scatter_add(0, f_mask_v_idx[:, 2], f_ones)
    f_v_value = torch.gather(v_value.unsqueeze(-1).tile(1, 3), 0, f_v_idx)
    f_mask = (f_v_value.sum(dim=-1) > 0)
    return dilate_face(f_v_idx, f_mask, V, depth=depth-1)


def erode_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    return ~dilate_face(f_v_idx, ~f_mask, V, depth=depth)


def dilate_erode_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    return dilate_face(f_v_idx, erode_face(f_v_idx, dilate_face(f_v_idx, f_mask, V, depth=depth), V, depth=2*depth), V, depth=depth)


def get_boundary(f_v_idx:torch.Tensor):
    e_v_idx_full = torch.cat([f_v_idx[:, [0, 1]], f_v_idx[:, [1, 2]], f_v_idx[:, [2, 0]]], dim=0)
    e_v_idx_sorted, _ = torch.sort(e_v_idx_full, dim=-1)
    e_v_idx, e_count = torch.unique(e_v_idx_sorted, dim=0, sorted=False, return_inverse=False, return_counts=True)
    v_idx_boundary = torch.unique(e_v_idx[e_count == 1].flatten(), dim=0, sorted=False, return_inverse=False, return_counts=False)
    return v_idx_boundary


def get_boundary_tex(f_v_idx_pos:torch.Tensor, f_v_idx_tex:torch.Tensor, paired=False):
    '''
    v_idx_sel_pos, v_idx_sel_tex: [V_s,]
    f_v_idx_sel_pos, f_v_idx_tex_sub: [F, 3]
    e_v_idx_ccw_sel_pos, e_v_idx_ccw_sel_tex: [E_s, G, 3], 
        G = 2, ccw = 2 edge vertex + 1 dual vertex
    '''
    e_v_idx_ccw_pos = torch.cat([f_v_idx_pos[:, [0, 1, 2]], f_v_idx_pos[:, [1, 2, 0]], f_v_idx_pos[:, [2, 0, 1]]], dim=0)  # [3*F, 3]
    e_v_idx_ccw_tex = torch.cat([f_v_idx_tex[:, [0, 1, 2]], f_v_idx_tex[:, [1, 2, 0]], f_v_idx_tex[:, [2, 0, 1]]], dim=0)
    e_v_idx_full_pos = e_v_idx_ccw_pos[:, :2]  # [3*F, 2]
    e_v_idx_full_tex = e_v_idx_ccw_tex[:, :2]
    e_v_idx_sorted_pos, e_v_idx_sorted_idx_pos = torch.sort(e_v_idx_full_pos, dim=-1)  # [3*F, 2], [3*F, 2]
    e_v_idx_sorted_tex, e_v_idx_sorted_idx_tex = torch.sort(e_v_idx_full_tex, dim=-1)
    e_v_idx_pos, f_e_idx_pos, e_count_pos = torch.unique(e_v_idx_sorted_pos, dim=0, sorted=False, return_inverse=True, return_counts=True)  # [E, 2], [3*F,], [E,]
    e_v_idx_tex, f_e_idx_tex, e_count_tex = torch.unique(e_v_idx_sorted_tex, dim=0, sorted=False, return_inverse=True, return_counts=True)
    e_mask_boundary_pos = (e_count_pos == 1)  # [E,]
    e_mask_boundary_tex = (e_count_tex == 1)
    # v_idx_boundary_pos = torch.unique(e_v_idx_pos[e_mask_boundary_pos].flatten(), dim=0, sorted=False, return_inverse=False, return_counts=False)  # [V_b,]
    # v_idx_boundary_tex = torch.unique(e_v_idx_tex[e_mask_boundary_tex].flatten(), dim=0, sorted=False, return_inverse=False, return_counts=False)
    f_e_mask_boundary_pos = e_mask_boundary_pos[f_e_idx_pos]  # [3*F,]
    f_e_mask_boundary_tex = e_mask_boundary_tex[f_e_idx_tex]
    f_e_mask_sel = torch.logical_and(f_e_mask_boundary_tex, torch.logical_not(f_e_mask_boundary_pos))

    e_v_idx_ccw_sel_pos = torch.masked_select(e_v_idx_ccw_pos, f_e_mask_sel.unsqueeze(-1)).reshape(-1, 3)  # [E_s, 3]
    e_v_idx_ccw_sel_tex = torch.masked_select(e_v_idx_ccw_tex, f_e_mask_sel.unsqueeze(-1)).reshape(-1, 3)
    e_v_idx_full_sel_pos = e_v_idx_ccw_sel_pos[:, :2]  # [E_s, 2]
    e_v_idx_full_sel_tex = e_v_idx_ccw_sel_tex[:, :2]
    v_idx_sel_pos = torch.unique(e_v_idx_full_sel_pos.flatten(), dim=0, sorted=False, return_inverse=False, return_counts=False)  # [V_s,]
    v_idx_sel_tex = torch.unique(e_v_idx_full_sel_tex.flatten(), dim=0, sorted=False, return_inverse=False, return_counts=False)

    if not paired:
        f_mask_sel_bi = torch.isin(f_v_idx_pos, v_idx_sel_pos).any(dim=-1)  # [F,]
        f_v_idx_sel_pos = f_v_idx_pos[f_mask_sel_bi]  # [F, 3]
        f_v_idx_tex_sub = f_v_idx_tex[f_mask_sel_bi]  # NOTE: bidirectional
        return v_idx_sel_pos, v_idx_sel_tex, f_v_idx_sel_pos, f_v_idx_tex_sub

    e_mask_type_1_pos = (e_v_idx_sorted_idx_pos[:, 0] == 0)  # [3*F,]
    e_mask_type_2_pos = (e_v_idx_sorted_idx_pos[:, 0] == 1)
    e_mask_type_1_tex = (e_v_idx_sorted_idx_tex[:, 0] == 0)
    e_mask_type_2_tex = (e_v_idx_sorted_idx_tex[:, 0] == 1)
    f_idx = torch.arange(f_v_idx_pos.shape[0], dtype=f_v_idx_pos.dtype, device=f_v_idx_pos.device).repeat((3,))  # [3*F,]
    e_f_idx_pos = torch.zeros((e_v_idx_pos.shape[0], 2), dtype=f_v_idx_pos.dtype, device=f_v_idx_pos.device)  # [E, 2]
    e_f_idx_pos[:, 0].scatter_(dim=0, index=f_e_idx_pos[e_mask_type_1_pos], src=f_idx[e_mask_type_1_pos])
    e_f_idx_pos[:, 1].scatter_(dim=0, index=f_e_idx_pos[e_mask_type_2_pos], src=f_idx[e_mask_type_2_pos])
    e_f_idx_tex = torch.zeros((e_v_idx_tex.shape[0], 2), dtype=f_v_idx_pos.dtype, device=f_v_idx_pos.device)  # [E, 2]
    e_f_idx_tex[:, 0].scatter_(dim=0, index=f_e_idx_pos[e_mask_type_1_tex], src=f_idx[e_mask_type_1_tex])
    e_f_idx_tex[:, 1].scatter_(dim=0, index=f_e_idx_pos[e_mask_type_2_tex], src=f_idx[e_mask_type_2_tex])
    e_idx_sel_pos = f_e_idx_pos[f_e_mask_sel]  # [E_s,]
    e_idx_sel_tex = f_e_idx_tex[f_e_mask_sel]
    e_f_idx_sel_pos = e_f_idx_pos[e_idx_sel_pos]  # [E_s, 2]
    e_f_idx_sel_tex = e_f_idx_tex[e_idx_sel_tex]
    e_v_idx_ccw_sel_pos = e_v_idx_ccw_pos[e_f_idx_sel_pos]  # [E_s, 2, 3]
    e_v_idx_ccw_sel_tex = e_v_idx_ccw_tex[e_f_idx_sel_pos]  # NOTE: bidirectional
    return v_idx_sel_pos, v_idx_sel_tex, e_v_idx_ccw_sel_pos, e_v_idx_ccw_sel_tex

def reverse_triangle_2d(v1:torch.Tensor, v2:torch.Tensor, v3:torch.Tensor, u1:torch.Tensor, u2:torch.Tensor):
    return u1 + torch.norm(u2 - u1, dim=-1, keepdim=True) / torch.norm(v2 - v1, dim=-1, keepdim=True) * (v3 - v1)

def reverse_triangle_group_2d(e_v_ccw_sel_tex:torch.Tensor):
    '''
    e_v_ccw_sel_tex: [E_s, G, 3, 2], 
        G = 2, ccw = 2 edge vertex + 1 dual vertex
    '''
    v1, v2, v3, u1, u2, u3 = e_v_ccw_sel_tex.reshape(-1, 2*3, 2).unbind(dim=1)
    u3_reverse = reverse_triangle_2d(v1, v2, v3, u1, u2)
    v3_reverse = reverse_triangle_2d(u1, u2, u3, v1, v2)
    e_v_ccw_sel_tex_reverse = torch.stack([v1, v2, v3_reverse, u1, u2, u3_reverse], dim=1).reshape(-1, 2, 3, 2)
    return e_v_ccw_sel_tex_reverse

