import math
from typing import Optional, Tuple
import torch


def select_sharp_edges(
    vertices:torch.Tensor, 
    faces:torch.Tensor,
    normals:Optional[torch.Tensor],
    angle_threhold_deg:float=15.0,
):
    '''
    find sharp edges in triangle mesh

    vertices: [V, 3], float32
    faces: [F, 3], int64
    normals: [F, 3], float32, normalized
    adjacency: [E_adj, 2], wrt faces
    edges_unique: [E, 2], int64
    edges_mask_nonmanifold: [E,], bool
    edges_mask_sharp: [E,], bool
    '''
    assert 0.0 <= angle_threhold_deg <= 180.0, f'angle_threhold_deg should be in [0.0, 180.0], but {angle_threhold_deg}'
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges.sort(dim=1, stable=True).values
    edges_face = torch.arange(faces.shape[0], dtype=faces.dtype, device=faces.device).repeat_interleave(3, dim=0)
    edges_unique, edges_inverse, edges_counts = torch.unique(edges, dim=0, return_inverse=True, return_counts=True)
    # edges_index = edges_inverse.argsort(stable=True)[torch.cat([edges_counts.new_zeros(1), edges_counts.cumsum(dim=0)])[:-1]]
    edges_mask = (edges_counts == 2)
    edges_mask_inverse = edges_mask[edges_inverse]
    edges__value, edges__index = edges_inverse[edges_mask_inverse].sort(dim=0, stable=True)
    edges__value = edges__value.reshape(-1, 2)[:, 0]
    adjacency = edges_face[edges_mask_inverse][edges__index].reshape(-1, 2)
    adjacency = adjacency.sort(dim=1, stable=True).values

    if normals is None:
        areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
        normals = torch.nn.functional.normalize(areas, dim=-1)
    adjacency_normal = normals[adjacency, :]
    adjacency_cos = torch.nn.functional.cosine_similarity(adjacency_normal[:, 0, :], adjacency_normal[:, 1, :], dim=-1)
    adjacency_mask = (adjacency_cos > math.cos(math.radians(angle_threhold_deg)))

    edges_mask_nonmanifold = torch.logical_not(edges_mask)
    edges_mask_sharp = edges_mask.clone()
    edges_mask_sharp.scatter_(0, edges__value[adjacency_mask], False)
    return edges_unique, edges_mask_nonmanifold, edges_mask_sharp


def sample_on_edges_v1(
    vertices:torch.Tensor, 
    edges:torch.Tensor, 
    edges_mask:Optional[torch.Tensor], 
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    uniform sampling on (selected) edges by probability

    vertices: [V, 3], float32
    edges: [E, 2], int64
    edges_mask: [E,], bool

    samples: [N, 3], float32
    edge_index: [N,], int64
    edge_t: [N, 1], float32
    '''
    generator = None if seed is None else torch.Generator(device='cuda').manual_seed(seed)
    if edges_mask is not None:
        edges = torch.masked_select(edges, edges_mask.unsqueeze(-1)).reshape(-1, 2)
        assert edges.shape[0] > 0, f'edges selected by edges_mask are empty'
        edge_index = torch.where(edges_mask)[0]
    else:
        edge_index = torch.arange(edges.shape[0], dtype=torch.int64, device='cuda')
    edges_vertex = vertices[edges, :]
    edges_length = torch.norm(edges_vertex[:, 1, :] - edges_vertex[:, 0, :], dim=-1)
    edge_t = torch.rand((N,), dtype=torch.float32, device='cuda', generator=generator).unsqueeze(-1)
    samples_idx = torch.multinomial(edges_length, N, replacement=True, generator=generator)
    samples_edge_vertex = edges_vertex[samples_idx, :, :]
    samples = edge_t * samples_edge_vertex[:, 0, :] + (1.0 - edge_t) * samples_edge_vertex[:, 1, :]
    edge_index = edge_index[samples_idx]
    return samples, edge_index, edge_t


def sample_on_edges_v2(
    vertices:torch.Tensor, 
    edges:torch.Tensor, 
    edges_mask:Optional[torch.Tensor], 
    N=10_000_000,
    seed:Optional[int]=666,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    uniform sampling on (selected) edges with equal steps

    vertices: [V, 3], float32
    edges: [E, 2], int64
    edges_mask: [E,], bool

    samples: [N, 3], float32
    edge_index: [N,], int64
    edge_t: [N, 1], float32
    '''
    if edges_mask is not None:
        edges = torch.masked_select(edges, edges_mask.unsqueeze(-1)).reshape(-1, 2)
        assert edges.shape[0] > 0, f'edges selected by edges_mask are empty'
        edge_index = torch.where(edges_mask)[0]
    else:
        edge_index = torch.arange(edges.shape[0], dtype=torch.int64, device='cuda')
    edges_vertex = vertices[edges, :]
    edges_length = torch.norm(edges_vertex[:, 1, :] - edges_vertex[:, 0, :], dim=-1)
    edges_start = torch.roll(torch.cumsum(edges_length, dim=0), shifts=1, dims=0)
    edges_start[0] = 0.0
    edges_t = torch.linspace(0.0, edges_length.sum().item(), N, dtype=torch.float32, device='cuda')
    edges_idx = torch.searchsorted(edges_start[1:], edges_t)
    edge_t = ((edges_t - edges_start[edges_idx]) / edges_length[edges_idx]).unsqueeze(-1)
    edge_t = torch.nan_to_num(edge_t, nan=0.5, posinf=0.5, neginf=0.5)
    samples_edge_vertex = edges_vertex[edges_idx, :, :]
    samples = edge_t * samples_edge_vertex[:, 0, :] + (1.0 - edge_t) * samples_edge_vertex[:, 1, :]
    edge_index = edge_index[edges_idx]
    return samples, edge_index, edge_t


def select_and_sample_on_edges(
    vertices:torch.Tensor,
    faces:torch.Tensor,
    normals:Optional[torch.Tensor]=None,
    method='equal_steps',
    angle_threhold_deg:float=15.0,
    N=10_000_000,
    seed=666,
):
    '''
    uniform sampling on (selected) edges with equal steps

    vertices: [V, 3], float32
    faces: [F, 3], int64
    normals: [F, 3], float32, optional
    method: probability, equal_steps

    samples: [N, 3], float32
    edge_index: [N,], int64
    edge_t: [N, 1], float32
    '''
    assert method in ['probability', 'equal_steps']
    edges, edges_mask_nonmanifold, edges_mask_sharp = select_sharp_edges(vertices, faces, normals=normals, angle_threhold_deg=angle_threhold_deg)
    edges_mask = torch.logical_or(edges_mask_nonmanifold, edges_mask_sharp)
    if edges_mask.sum() == 0:
        print('no sharp edges or nonmanifold edges on mesh')
        samples = None
        edge_index = None
        edge_t = None
    else:
        if method == 'probability':
            samples, edge_index, edge_t = sample_on_edges_v1(vertices, edges, edges_mask=edges_mask, N=N, seed=seed)
        elif method == 'equal_steps':
            samples, edge_index, edge_t = sample_on_edges_v2(vertices, edges, edges_mask=edges_mask, N=N, seed=seed)
        else:
            raise NotImplementedError(f'method {method} is not supported')
    return samples, edge_index, edge_t

