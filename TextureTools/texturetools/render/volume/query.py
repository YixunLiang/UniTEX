import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch


def cube_to_dir(cube:torch.Tensor):
    '''
    cube: [..., 6, 2]
    dir: [..., 6, 3]
        x: origin to front, 
        y: origin to top, 
        z: origin to right,
    '''
    x, y = cube.unbind(-1)
    ones = torch.ones_like(x[..., 0])
    return torch.stack([
        ones, -y[..., 0], -x[..., 0],  # front
        -ones, -y[..., 1], x[..., 1],  # back
        x[..., 2], ones, y[..., 2],    # top
        x[..., 3], -ones, -y[..., 3],  # down
        x[..., 4], -y[..., 4], ones,   # right
        -x[..., 5], -y[..., 5], -ones, # left
    ], dim=-1).reshape(*x.shape, 6, 3)


def dir_to_cube(dir:torch.Tensor):
    '''
    dir: [..., 6, 3]
    cube: [..., 6, 2]
        x: origin to front, 
        y: origin to top, 
        z: origin to right,
    '''
    x, y, z = dir.unbind(-1)
    return torch.stack([
        -z[..., 0], -y[..., 0],  # front
        z[..., 1], -y[..., 1],   # back
        x[..., 2], z[..., 2],    # top
        x[..., 3], -z[..., 3],   # down
        x[..., 4], -y[..., 4],   # right
        -x[..., 5], -y[..., 5],  # left
    ], dim=-1).reshape(*x.shape, 6, 2)


def create_circle(n: int, radius = 1.0, t_0 = 0.0, dtype=torch.float32, device='cpu', analytic=True):
    t = torch.linspace(0, 1, n+1, dtype=dtype, device=device)
    d_t = torch.gradient(t)[0]
    gamma = torch.complex(
        radius * torch.cos(2 * torch.pi * (t - t_0)), 
        radius * torch.sin(2 * torch.pi * (t - t_0)),
    )
    if analytic:  # analytic gradient
        d_gamma = torch.complex(
            - radius * 2 * torch.pi * torch.sin(2 * torch.pi * (t - t_0)), 
            radius * 2 * torch.pi * torch.cos(2 * torch.pi * (t - t_0)),
        ) * d_t
    else:  # numerical gradient
        d_gamma = torch.gradient(gamma)[0]
    return t, d_t, gamma, d_gamma


def create_sphere(n:int, r_u = 1.0, r_v = 1.0, u_0 = 0.0, v_0 = 0.0, dtype=torch.float32, device='cpu'):
    u = torch.linspace(0, 1, n+1, dtype=dtype, device=device)
    v = torch.linspace(0, 1, n+1, dtype=dtype, device=device)
    d_u = torch.gradient(u)[0]
    d_v = torch.gradient(v)[0]
    alpha = torch.stack([
        r_v * torch.sin(torch.pi * (v - v_0)) * r_u * torch.cos(2 * torch.pi * (u - u_0)),
        r_v * torch.sin(torch.pi * (v - v_0)) * r_u * torch.sin(2 * torch.pi * (u - u_0)),
        r_v * torch.cos(torch.pi * (v - v_0)),
    ], dim=-1)
    return alpha


def cauchy_integrate(z:torch.Tensor, gamma:torch.Tensor, d_gamma:torch.Tensor, f_gamma:torch.Tensor):
    '''
    z: [...]
    gamma: [N,]
    d_gamma: [N,]
    f_gamma: [..., N, C]
    c_gamma_acc: [..., C]
    '''
    z = z.unsqueeze(-1).unsqueeze(-1)
    gamma = gamma.unsqueeze(-1)
    d_gamma = d_gamma.unsqueeze(-1)

    c_gamma = f_gamma * d_gamma / (gamma - z)
    c_gamma_acc = c_gamma.sum(dim=-2) / (2j * torch.pi)
    return c_gamma_acc


def query_triplane(positions: torch.Tensor, triplanes: torch.Tensor) -> torch.Tensor:
    '''
    positions: [B, N, 3]
    triplanes: [B, Np, Cp, Hp, Wp]
    features: [B, N, Np * Cp]
    '''
    B, N, _ = positions.shape
    B, Np, Cp, Hp, Wp = triplanes.shape
    assert Np == 3
    coordinates = torch.stack((
        positions[..., [0, 1]],
        positions[..., [0, 2]],
        positions[..., [1, 2]],
    ), dim=-3).reshape(B * Np, 1, N, 2)
    triplanes = triplanes.reshape(B * Np, Cp, Hp, Wp)
    features = torch.nn.functional.grid_sample(triplanes, coordinates, align_corners=False, mode='bilinear')
    features = features.reshape(B, Np * Cp, N).permute(0, 2, 1)
    return features


def query_box(positions: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    '''
    positions: [B, N, 3]
    boxes: [B, Nb, Cb, Hb, Wb]
    features: [B, N, Nb * Cb]
    '''
    B, N, _ = positions.shape
    B, Nb, Cb, Hb, Wb = boxes.shape
    assert Nb == 6
    coordinates = torch.stack((
        positions[..., [0, 1]],
        positions[..., [0, 1]],
        positions[..., [0, 2]],
        positions[..., [0, 2]],
        positions[..., [1, 2]],
        positions[..., [1, 2]],
    ), dim=-3).reshape(B * Nb, 1, N, 2)
    boxes = boxes.reshape(B * Nb, Cb, Hb, Wb)
    features = torch.nn.functional.grid_sample(boxes, coordinates, align_corners=False, mode='bilinear')
    features = features.reshape(B, Nb * Cb, N).permute(0, 2, 1)
    return features


def discretize_2d(v_pos_ndc:torch.Tensor, H:int, W:int, ndc=True, align_corner=False):
    uf, vf = v_pos_ndc.unbind(-1)
    if ndc:
        uf = uf * 0.5 + 0.5
        vf = vf * 0.5 + 0.5
    if not align_corner:
        ui = torch.floor(uf * W).to(dtype=torch.int64)
        vi = torch.floor(vf * H).to(dtype=torch.int64)
    else:
        ui = torch.floor(uf * (W - 1) + 0.5).to(dtype=torch.int64)
        vi = torch.floor(vf * (H - 1) + 0.5).to(dtype=torch.int64)
    v_pos_pix = torch.stack([ui, vi], dim=-1)
    return v_pos_pix


def undiscretize_2d(v_pos_pix:torch.Tensor, H:int, W:int, ndc=True, align_corner=False):
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


def undiscretize_3d(v_pos_pix:torch.Tensor, D:int, H:int, W:int, ndc=True, align_corner=False):
    ui, vi, wi = v_pos_pix.unbind(-1)
    if not align_corner:
        uf = (ui + 0.5) / W
        vf = (vi + 0.5) / H
        wf = (wi + 0.5) / D
    else:
        uf = ui / (W - 1)
        vf = vi / (H - 1)
        wf = wi / (H - 1)
    if ndc:
        uf = uf * 2.0 - 1.0
        vf = vf * 2.0 - 1.0
        wf = wf * 2.0 - 1.0
    v_pos_ndc = torch.stack([uf, vf, wf], dim=-1)
    return v_pos_ndc


def make_grid_2d(H, W, dtype=torch.float32, device='cpu'):
    ys = torch.arange(H, dtype=dtype, device=device)
    xs = torch.arange(W, dtype=dtype, device=device)
    grid_i = torch.cartesian_prod(ys, xs).reshape(H, W, 2).flip(-1)
    grid_f = undiscretize_2d(grid_i, H=H, W=W)
    return grid_f


def make_grid_3d(D, H, W, dtype=torch.float32, device='cpu'):
    zs = torch.arange(D, dtype=dtype, device=device)
    ys = torch.arange(H, dtype=dtype, device=device)
    xs = torch.arange(W, dtype=dtype, device=device)
    grid_i = torch.cartesian_prod(zs, ys, xs).reshape(D, H, W, 3).flip(-1)
    grid_f = undiscretize_3d(grid_i, H=H, W=W)
    return grid_f


def query_circle(positions: torch.Tensor, circles: torch.Tensor) -> torch.Tensor:
    '''
    positions: [B, N, 2]
    circles: [B, Cc, Wc]
    features: [B, N, Cc]
    '''
    B, N, _ = positions.shape
    B, Cc, Wc = circles.shape
    circles = torch.cat([circles, circles[..., [0]]], dim=-1)
    t, d_t, gamma, d_gamma = create_circle(Wc, dtype=positions.dtype, device=positions.device)
    z = torch.view_as_complex(positions)
    f_gamma = circles.permute(0, 2, 1).unsqueeze(-3)
    c_gamma_acc = cauchy_integrate(z, gamma, d_gamma, f_gamma).real
    return c_gamma_acc


def query_cylinder(positions: torch.Tensor, cylinders: torch.Tensor) -> torch.Tensor:
    '''
    positions: [B, N, 3]
    cylinders: [B, Cc, Dc, Wc]
    features: [B, N, Cc]
    '''
    B, N, _ = positions.shape
    B, Cc, Dc, Wc = cylinders.shape
    H, W = 128, 128
    grid = make_grid_2d(H, W, dtype=positions.dtype, device=positions.device)
    t, d_t, gamma, d_gamma = create_circle(Wc-1, dtype=positions.dtype, device=positions.device)
    z = torch.view_as_complex(grid).unsqueeze(-3)
    f_gamma = cylinders.permute(0, 2, 3, 1).unsqueeze(-4).unsqueeze(-4)
    c_gamma_acc = cauchy_integrate(z, gamma, d_gamma, f_gamma).real.permute(0, 4, 3, 1, 2)  # [B, C, D, H, W]
    # FIXME


def uv_to_normal(uv:torch.Tensor):
    # NOTE: \theta \in [0, 2 \pi], \psi \in [0, \pi]
    u, v = uv.unbind(-1)
    theta = u * torch.pi + torch.pi
    psi = v * (torch.pi / 2) + (torch.pi / 2)
    n_x = torch.sin(psi) * torch.cos(theta)
    n_y = torch.sin(psi) * torch.sin(theta)
    n_z = torch.cos(psi)
    normal = torch.stack([n_x, n_y, n_z], dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    return normal


def normal_to_uv(normal:torch.Tensor):
    # NOTE: \theta \in [0, 2 \pi], \psi \in [0, \pi]
    normal = torch.nn.functional.normalize(normal, dim=-1)
    n_x, n_y, n_z = normal.unbind(-1)
    sin_psi = torch.sqrt(1 - n_z.square())
    psi = torch.arccos(n_z)
    theta = torch.arctan2(n_y / sin_psi, n_x / sin_psi).nan_to_num()
    theta = torch.where(theta > 0, theta, theta + 2 * torch.pi)
    u = (theta - torch.pi) / torch.pi
    v = (psi - (torch.pi / 2)) / (torch.pi / 2)
    uv = torch.stack([u, v], dim=-1)
    return uv


## cell test ##


def test_query_circle():
    t, _, gamma, _ = create_circle(2048+1)
    xy = torch.view_as_real(gamma)
    x, y = xy.unbind(-1)
    circles = torch.stack([
        torch.sin(x),
        torch.cos(x+y),
        torch.sin(y),
    ], dim=-1)
    grid = make_grid_2d(512, 512)
    features = query_circle(grid, circles.unsqueeze(0).permute(0, 2, 1))

    # visualize features on circles
    vis = np.zeros((512, 512, 3), dtype=np.uint8)
    xy_i = discretize_2d(xy, 512, 512)
    for xy, v in zip(xy_i.clamp(0, 511).detach().cpu().numpy(), circles.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)):
        vis = cv2.circle(vis, xy, 1, v.tolist(), 2)
    cv2.imwrite('vis1.png', vis)

    # visualize features on plane
    vis = features.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('vis2.png', vis)


def test_uv_normal():
    grid = make_grid_2d(512, 512)
    normal = uv_to_normal(grid)
    uv = normal_to_uv(normal)
    cv2.imwrite('vis3.png', normal.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
    cv2.imwrite('vis4.png', torch.cat([grid, torch.full_like(grid[..., [0]], -1)], dim=-1).mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))
    cv2.imwrite('vis5.png', torch.cat([uv, torch.full_like(uv[..., [0]], -1)], dim=-1).mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255).detach().cpu().numpy().astype(np.uint8))


def test_fitting():
    im = cv2.resize(cv2.imread("refined_im.png", -1), (256, 256))
    features_gt = torch.as_tensor(im, dtype=torch.float32).div(255).mul(2.0).sub(1.0)

    t, _, gamma, _ = create_circle(2048+1)
    xy = torch.view_as_real(gamma)
    x, y = xy.unbind(-1)
    circles = torch.nn.Parameter(torch.zeros((2048+1, 3), dtype=torch.float32))
    grid = make_grid_2d(256, 256)

    circles.data = circles.data.cuda()
    grid = grid.cuda()
    features_gt = features_gt.cuda()
    opt = torch.optim.Adam([circles], lr=0.1)
    for i in range(100):
        features = query_circle(grid, circles.unsqueeze(0).permute(0, 2, 1))
        loss = torch.nn.MSELoss()(features, features_gt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(i, loss.item())

    circles = circles.data
    features = query_circle(grid, circles.unsqueeze(0).permute(0, 2, 1))

    # visualize features on circles
    vis = np.zeros((512, 512, 3), dtype=np.uint8)
    xy_i = discretize_2d(xy, 512, 512)
    for xy, v in zip(xy_i.clamp(0, 511).detach().cpu().numpy(), circles.mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)):
        vis = cv2.circle(vis, xy, 1, v.tolist(), 2)
    cv2.imwrite('vis6.png', vis)

    # visualize features on plane
    vis = torch.cat([features, features_gt], dim=1).mul(0.5).add(0.5).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('vis7.png', vis)


if __name__ == '__main__':
    # test_query_circle()
    # test_uv_normal()
    test_fitting()

