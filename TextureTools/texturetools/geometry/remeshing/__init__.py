import os
from typing import Tuple
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from .normal_renderer import (
    NormalRenderer,
)
from .large_step_remeshing import (
    AdamUniform, 
    compute_matrix, 
    from_differential, 
    get_differential_solver, 
    to_differential,
)
from .continuous_remeshing import (
    MeshOptimizer
)


def large_step_remeshing(
    vertices:torch.Tensor, faces:torch.Tensor, 
    c2ws:torch.Tensor, projections:torch.Tensor, gt_images:torch.Tensor, 
    n_steps=1000,
    visualize=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    vertices: [V, 3]
    faces: [F, 3]
    c2ws: [N, 3, 3]
    projections: [N, 4, 4] or [4, 4]
    gt_images: [N, H, W, 3]
    n_steps: Number of optimization steps
    '''
    if visualize:
        os.makedirs(f'./.cache', exist_ok=True)
    v, f = vertices, faces
    normal_renderer = NormalRenderer()
    _, H, W, _ = gt_images.shape
    mvp = torch.matmul(projections, torch.inverse(c2ws))
    M = compute_matrix(v, f, lambda_=19.0)
    solver = get_differential_solver(M, 'Cholesky')
    u = to_differential(M, v)
    u = torch.nn.Parameter(u)
    opt = AdamUniform([u], lr=3e-2)
    for iter in tqdm(range(n_steps)):
        v = from_differential(solver, u)
        render_images = normal_renderer(v, f, mvp, H, W)
        loss = (render_images - gt_images).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if visualize:
            save_image(torch.cat([render_images, gt_images], dim=0).mul(0.5).add(0.5).clamp(0.0, 1.0).permute(0, 3, 1, 2), f'./.cache/{iter:04d}.png')
    vertices = v.detach()
    faces = f
    return vertices, faces


def continuous_remeshing(
    vertices:torch.Tensor, faces:torch.Tensor, 
    c2ws:torch.Tensor, projections:torch.Tensor, gt_images:torch.Tensor, 
    n_steps=1000,
    visualize=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    vertices: [V, 3]
    faces: [F, 3]
    c2ws: [N, 3, 3]
    projections: [N, 4, 4] or [4, 4]
    gt_images: [N, H, W, 3]
    n_steps: Number of optimization steps
    '''
    if visualize:
        os.makedirs(f'./.cache', exist_ok=True)
    v, f = vertices, faces
    normal_renderer = NormalRenderer()
    _, H, W, _ = gt_images.shape
    mvp = torch.matmul(projections, torch.inverse(c2ws))

    opt = MeshOptimizer(v, f, lr=3e-2)
    for iter in tqdm(range(n_steps)):
        render_images = normal_renderer(v, f, mvp, H, W)
        loss = (render_images - gt_images).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        opt.remesh()
        if visualize:
            save_image(torch.cat([render_images, gt_images], dim=0).mul(0.5).add(0.5).clamp(0.0, 1.0).permute(0, 3, 1, 2), f'./.cache/{iter:04d}.png')
    vertices = opt.vertices.detach()
    faces = opt.faces.detach()
    return vertices, faces


# TODO
def masked_large_step_remeshing(
    vertices:torch.Tensor, faces:torch.Tensor, 
    c2ws:torch.Tensor, projections:torch.Tensor, gt_images:torch.Tensor, weight_images:torch.Tensor,
    n_steps=1000,
    visualize=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    vertices: [V, 3]
    faces: [F, 3]
    c2ws: [N, 3, 3]
    projections: [N, 4, 4] or [4, 4]
    gt_images: [N, H, W, 3]
    weight_images: [N, H, W, 1]

    n_steps: Number of optimization steps
    step_size: Step size
    lambda_: Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)
    '''
    if visualize:
        os.makedirs(f'./.cache', exist_ok=True)
    v, f = vertices, faces
    normal_renderer = NormalRenderer()
    _, H, W, _ = gt_images.shape
    mvp = torch.matmul(projections, torch.inverse(c2ws))
    M = compute_matrix(v, f, lambda_=19.0)
    solver = get_differential_solver(M, 'Cholesky')
    u = to_differential(M, v)
    u = torch.nn.Parameter(u)
    opt = AdamUniform([u], step_size=3e-2)
    for iter in tqdm(range(n_steps)):
        v = from_differential(solver, u)
        render_images = normal_renderer(v, f, mvp, H, W)
        loss = (render_images - gt_images).mul(weight_images).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if visualize:
            save_image(torch.cat([render_images, gt_images], dim=0).mul(0.5).add(0.5).clamp(0.0, 1.0).permute(0, 3, 1, 2), f'./.cache/{iter:04d}.png')
    vertices = v.detach()
    faces = f
    return vertices, faces


