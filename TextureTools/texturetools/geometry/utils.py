from typing import Union
import numpy as np
import torch


def to_tensor_f(vertices:Union[torch.Tensor, np.ndarray], device='cuda'):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.to(dtype=torch.float32, device=device)
    else:
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    return vertices

def to_tensor_i(faces:Union[torch.Tensor, np.ndarray], device='cuda'):
    if isinstance(faces, torch.Tensor):
        faces = faces.to(dtype=torch.int64, device=device)
    else:
        faces = torch.as_tensor(faces, dtype=torch.int64, device=device)
    return faces

def to_array_f(vertices:Union[torch.Tensor, np.ndarray]):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy().astype(np.float32)
    elif isinstance(vertices, np.ndarray):
        vertices = vertices.astype(np.float32)
    else:
        vertices = np.asarray(vertices, dtype=np.float32)
    return vertices

def to_array_i(faces:Union[torch.Tensor, np.ndarray]):
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy().astype(np.int64)
    elif isinstance(faces, np.ndarray):
        faces = faces.astype(np.int64)
    else:
        faces = np.asarray(faces, dtype=np.int64)
    return faces

