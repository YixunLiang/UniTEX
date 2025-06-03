'''
reference: https://solarianprogrammer.com/2013/05/22/opengl-101-matrices-projection-view-model/
'''
from typing import List, Tuple
import torch


def intr_to_proj(intr_mtx:torch.Tensor, near=0.01, far=1000.0, perspective=True):
    proj_mtx = torch.zeros((*intr_mtx.shape[:-2], 4, 4), dtype=intr_mtx.dtype, device=intr_mtx.device)
    intr_mtx = intr_mtx.clone()
    if perspective:
        proj_mtx[..., 0, 0] = 2 * intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = 2 * intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -(far + near) / (far - near)
        proj_mtx[..., 0, 2] = 2 * intr_mtx[..., 0, 2] - 1
        proj_mtx[..., 1, 2] = 2 * intr_mtx[..., 1, 2] - 1
        proj_mtx[..., 3, 2] = -1.0
        proj_mtx[..., 2, 3] = -2.0 * far * near / (far - near)
    else:
        proj_mtx[..., 0, 0] = intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -2.0 / (far - near)
        proj_mtx[..., 3, 3] = 1.0
        proj_mtx[..., 0, 3] = -(2 * intr_mtx[..., 0, 2] - 1)
        proj_mtx[..., 1, 3] = -(2 * intr_mtx[..., 1, 2] - 1)
        proj_mtx[..., 2, 3] = - (far + near) / (far - near)
    proj_mtx[..., 1, :] = -proj_mtx[..., 1, :]  # for nvdiffrast
    return proj_mtx


def proj_to_intr(proj_mtx:torch.Tensor, perspective=True):
    intr_mtx = torch.zeros((*proj_mtx.shape[:-2], 3, 3), dtype=proj_mtx.dtype, device=proj_mtx.device)
    proj_mtx = proj_mtx.clone()
    proj_mtx[..., 1, :] = -proj_mtx[..., 1, :]  # for nvdiffrast
    if perspective:
        intr_mtx[..., 0, 0] = proj_mtx[..., 0, 0] / 2.0
        intr_mtx[..., 1, 1] = proj_mtx[..., 1, 1] / 2.0
        intr_mtx[..., 0, 2] = 0.5 * proj_mtx[..., 0, 2] + 0.5
        intr_mtx[..., 1, 2] = 0.5 * proj_mtx[..., 1, 2] + 0.5
        intr_mtx[..., 2, 2] = 1.0
    else:
        intr_mtx[..., 0, 0] = proj_mtx[..., 0, 0]
        intr_mtx[..., 1, 1] = proj_mtx[..., 1, 1]
        intr_mtx[..., 0, 2] = 0.5 * (-proj_mtx[..., 0, 3]) + 0.5
        intr_mtx[..., 1, 2] = 0.5 * (-proj_mtx[..., 1, 3]) + 0.5
        intr_mtx[..., 2, 2] = 1.0
    return intr_mtx


def c2w_to_w2c(c2w:torch.Tensor):
    # y = Rx + t, x = R_inv(y - t)
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    c2w = c2w.clone()
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c


def get_mvp_mtx(c2ws:torch.Tensor, intrinsics:torch.Tensor, perspective=True):
    w2cs = c2w_to_w2c(c2ws)
    projections = intr_to_proj(intrinsics, perspective=perspective)
    mvps = torch.matmul(projections, w2cs)
    return mvps


class Transforms:
    identity = torch.eye(4, dtype=torch.float32)
    flip_x = torch.as_tensor([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    flip_y = torch.as_tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    flip_z = torch.as_tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    swap_xy = torch.as_tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    swap_yz = torch.as_tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    swap_zx = torch.as_tensor([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)

def compose_transforms_by_names(names:List[str]):
    '''
    transforms rule: [f1, f2, f3] means f3(f2(f1(x))) or x.f1().f2().f3()
    * identity: identity
    * flip: flip_x, flip_y, flip_z
    * swap: swap_xy, swap_yz, swap_zx
    '''
    if len(names) == 0:
        return None
    elif len(names) == 1:
        return getattr(Transforms, names[0]).clone()
    else:
        return torch.linalg.multi_dot([getattr(Transforms, k) for k in reversed(names)])


def apply_transform(v_pos_homo:torch.Tensor, transforms:List[torch.Tensor]):
    if len(transforms) == 0:
        return v_pos_homo
    elif len(transforms) == 1:
        transform = transforms[0].clone()
    else:
        transform = torch.linalg.multi_dot(transforms)
    v_pos_homo = torch.matmul(v_pos_homo, transform.transpose(-2, -1).to(v_pos_homo))
    return v_pos_homo

def inverse_transform(v_pos_homo:torch.Tensor, transforms:List[torch.Tensor]):
    if len(transforms) == 0:
        return v_pos_homo
    elif len(transforms) == 1:
        transform = transforms[0].clone()
    else:
        transform = torch.linalg.multi_dot(transforms)
    transform_inverse = torch.linalg.inv(transform)
    v_pos_homo = torch.matmul(v_pos_homo, transform_inverse.transpose(-2, -1).to(v_pos_homo))
    return v_pos_homo


def project(v_pos_homo:torch.Tensor, intrinsics:torch.Tensor, perspective=True):
    projections = intr_to_proj(intrinsics, perspective=perspective)   
    v_pos_clip = torch.matmul(v_pos_homo, projections.transpose(-2, -1).to(v_pos_homo))
    v_depth = v_pos_clip[..., [3]]
    v_pos_ndc = v_pos_clip[..., :2] / v_depth
    return v_pos_ndc, v_depth

def unproject(v_pos_ndc:torch.Tensor, v_depth:torch.Tensor, intrinsics:torch.Tensor, perspective=True):
    projections = intr_to_proj(intrinsics, perspective=perspective)
    shape = torch.broadcast_shapes(v_pos_ndc.shape[:-1], v_depth.shape[:-1])
    v_pos_ndc = v_pos_ndc.expand(*shape, 2)
    v_depth = v_depth.expand(*shape, 1)
    if perspective:
        v_pos_homo = torch.cat([v_pos_ndc * v_depth, torch.zeros_like(v_depth), v_depth], dim=-1)
        v_pos_homo = torch.matmul(v_pos_homo, projections.inverse().transpose(-2, -1))
        v_pos_homo[..., -1] = 1.0
    else:
        v_pos_homo = torch.cat([v_pos_ndc, torch.zeros_like(v_depth), torch.ones_like(v_depth)], dim=-1)
        v_pos_homo = torch.matmul(v_pos_homo, projections.inverse().transpose(-2, -1))
        v_pos_homo[..., -2] = v_depth[..., 0]
        v_pos_homo[..., -1] = 1.0
    return v_pos_homo


def discretize(v_pos_ndc:torch.Tensor, H:int, W:int, ndc=True, align_corner=False, to_int=False):
    uf, vf = v_pos_ndc.unbind(-1)
    if ndc:
        uf = uf * 0.5 + 0.5
        vf = vf * 0.5 + 0.5
    if not align_corner:
        ui = uf * W
        vi = vf * H
    else:
        ui = uf * (W - 1) + 0.5
        vi = vf * (H - 1) + 0.5
    v_pos_pix = torch.stack([ui, vi], dim=-1)
    if to_int:
        v_pos_pix = torch.floor(v_pos_pix).to(dtype=torch.int64)
    return v_pos_pix

def undiscretize(v_pos_pix:torch.Tensor, H:int, W:int, ndc=True, align_corner=False, from_int=False):
    if from_int:
        v_pos_pix = v_pos_pix.to(dtype=torch.float32)
    ui, vi = v_pos_pix.unbind(-1)
    if not align_corner:
        uf = (ui + 0.5) / W
        vf = (vi + 0.5) / H
    else:
        uf = ui / (W - 1)
        vf = vi / (H - 1)
    if ndc:
        uf = uf * 2.0 - 1.0
        vf = vf * 2.0 - 1.0
    v_pos_ndc = torch.stack([uf, vf], dim=-1)
    return v_pos_ndc


def rays_to_c2ws(rays_o:torch.Tensor, rays_d:torch.Tensor) -> torch.Tensor:
    '''
    rays_o, rays_d: [..., 3], camera locations looking at origin
    c2ws: [..., 4, 4]
        * world: x forward, y right, z up, need to transform xyz to zxy
        * camera: z forward, x right, y up
    '''
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=rays_o.dtype, device=rays_o.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=rays_o.dtype, device=rays_o.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=rays_o.dtype, device=rays_o.device)
    rays_o, rays_d = torch.broadcast_tensors(rays_o, rays_d)
    batch_shape = rays_o.shape[:-1]
    # NOTE: camera locations are opposite from ray directions
    z_axis = torch.nn.functional.normalize(rays_d.neg(), dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis_mask = (x_axis == 0).all(dim=-1, keepdim=True)
    if x_axis_mask.sum() > 0:
        # NOTE: top and down is not well defined, hard code here
        x_axis = torch.where(x_axis_mask, e2, x_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    # NOTE: world f/r/u is x/y/z, camera f/r/u is z/x/y, so we need to transform x/y/z to z/x/y for c2ws
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)[..., [1, 2, 0], :]
    trans = rays_o[..., [1, 2, 0]]
    c2ws = torch.cat([
        torch.cat([rots, trans.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    return c2ws

def c2ws_to_rays(c2ws:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    c2ws: [..., 4, 4]
    rays_o, rays_d: [..., 3]
    '''
    rays_o = c2ws[..., [2, 0, 1], 3]
    rays_d = c2ws[..., [2, 0, 1], 2].neg()
    return rays_o, rays_d

def c2ws_to_ray_matrices(c2ws:torch.Tensor, intrinsics:torch.Tensor, H:int, W:int, perspective=True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    c2ws: [..., 4, 4]
    rays_om: [..., H, W, 3]
    rays_dm: [..., H, W, 3], without normalized
    '''
    c2ws = c2ws.unsqueeze(-3)
    intrinsics = intrinsics.unsqueeze(-3)
    rays_o = c2ws[..., :3, 3].unsqueeze(-2)
    rays_d = c2ws[..., :3, 2].neg().unsqueeze(-2)
    ys = torch.arange(H, dtype=c2ws.dtype, device=c2ws.device)
    xs = torch.arange(W, dtype=c2ws.dtype, device=c2ws.device)
    grid_i = torch.cartesian_prod(ys, xs).reshape(H, W, 2).flip(-1)
    grid_f = undiscretize(grid_i, H=H, W=W)
    grid_w = apply_transform(unproject(grid_f, torch.ones_like(grid_f[..., [0]]), intrinsics, perspective=perspective), [c2ws])
    if perspective:
        rays_dm = grid_w[..., :3] - rays_o
        rays_om = rays_o.expand_as(rays_dm)
    else:
        rays_om = grid_w[..., :3]
        rays_dm = rays_d.expand_as(rays_om)
    return rays_om, rays_dm


