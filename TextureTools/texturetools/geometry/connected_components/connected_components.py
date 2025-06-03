from typing import List, Tuple, Union
import numpy as np
from scipy.sparse import coo_matrix, csgraph
import torch

# https://docs.cupy.dev/en/stable/install.html
# pip install cupy-cuda12x
import cupy as cp
# https://docs.rapids.ai/api/cugraph/stable/basics
# pip install cugraph-cu12
import cugraph


def connected_components(edges:torch.Tensor, min_len=1):
    '''
    find groups of connected nodes from an edge list.

    edges: (E, 2), int64
    min_len: int, minimum length of a component group to return
    index: (V1 + V2 + ... + VN,), int64
    labels_counts: (N,), int64, i.e. [V1, V2, ..., VN]

    usage:
    ```python
    components_index = torch.split(index, labels_counts, dim=0)
    components_attr = torch.split(nodes_atr.index_select(0, index), labels_counts, dim=0)
    ```
    '''
    def _cc(_edges:torch.Tensor, _count=None) -> Tuple[int, torch.Tensor]:
        '''
        _edges: [E, 2]
        _labels: [V,]
        '''
        _count = _count or _edges.max().item() + 1
        if _edges.device.type == 'cpu':
            data_array = np.ones(_edges.shape[0], dtype=bool)
            row_col_array = _edges.cpu().numpy().T
            graph = coo_matrix(
                (data_array, row_col_array), 
                dtype=bool, 
                shape=(_count, _count),
            )
            labels_num, labels_array = csgraph.connected_components(
                graph,
                directed=False,
                connection='weak',
                return_labels=True,
            )
        elif _edges.device.type == 'cuda':
            # NOTE: RuntimeError: Internal error: got invalid data type enum value from Numpy: bool
            data_array = cp.ones(_edges.shape[0], dtype=cp.float32)
            row_col_array = cp.asarray(_edges, dtype=cp.int64).T
            graph = cp.sparse.coo_matrix(
                (data_array, row_col_array), 
                dtype=cp.float32, 
                shape=(_count, _count),
            )
            labels_num, labels_array = cugraph.connected_components(
                graph,
                directed=False,
                connection='weak',
                return_labels=True,
            )
        else:
            raise NotImplementedError(f'device {_edges.device.type} is not supported')
        _labels = torch.as_tensor(labels_array, dtype=_edges.dtype, device=_edges.device)
        return labels_num, _labels

    edges_unique = torch.unique(edges)
    if edges_unique.shape[0] == 0:
        return []
    if edges.shape[0] == 0:
        if min_len <= 1:
            return edges_unique.reshape(-1, 1).unbind(0)
        else:
            return []
    counts = [0]
    if len(edges) > 0:
        counts.append(edges.max().item())
    if len(edges_unique) > 0:
        counts.append(edges_unique.max().item())
    count = max(counts) + 1

    mask = torch.zeros((count,), dtype=torch.bool, device=edges.device)
    mask.scatter_(0, edges_unique, True)
    index = torch.where(mask)[0]
    edges = edges[mask[edges].all(dim=1)]

    # print(f"Computing connected components, V={count}, E={edges.shape[0]}, may take a while ...")
    # t = perf_counter()
    _, labels = _cc(edges, count)
    # print(f"Computing connected components {perf_counter() - t} sec")

    # NOTE: loop in python is too slow
    # components = [index[labels == label] for label in torch.unique(labels)]
    # labels_counts = torch.as_tensor([c.shape[0] for c in components], dtype=torch.int64, device=edges.device)
    # index = torch.cat(components, dim=0)

    labels_sorted, labels_index = torch.sort(labels, stable=True)
    index = index[labels_index]
    # NOTE: unique oom
    # labels_counts = torch.unique_consecutive(labels_index, return_counts=True)
    prepend = torch.full((1,), -1, dtype=torch.int64, device=edges.device)
    append = torch.full((1,), index.shape[0], dtype=torch.int64, device=edges.device)
    labels_counts = torch.diff(torch.nonzero(torch.diff(labels_sorted, prepend=prepend)).squeeze(-1), append=append)
    return index, labels_counts


def face_adjacency(faces:torch.Tensor) -> torch.Tensor:
    '''
    find adjacent face indices in triangle mesh

    faces: [F, 3]
    adjacency: [E_adj, 2], wrt faces
    '''
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges.sort(dim=1, stable=True).values
    edges_face = torch.arange(faces.shape[0], dtype=faces.dtype, device=faces.device).repeat_interleave(3, dim=0)
    edges_unique, edges_inverse, edges_counts = torch.unique(edges, dim=0, return_inverse=True, return_counts=True)
    # edges_index = edges_inverse.argsort(stable=True)[torch.cat([edges_counts.new_zeros(1), edges_counts.cumsum(dim=0)])[:-1]]
    edges_mask = (edges_counts == 2)[edges_inverse]
    edges__index = edges_inverse[edges_mask].sort(dim=0, stable=True).indices
    adjacency = edges_face[edges_mask][edges__index].reshape(-1, 2)
    adjacency = adjacency.sort(dim=1, stable=True).values
    return adjacency


def split_faces(faces:torch.Tensor, only_watertight=False, return_index=False) \
    -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    '''
    split triangle faces into connected components

    faces: [F, 3], int64
    only_watertight: whether return watertight components only
    return_index: whether return index of components
    components: [[F1, 3], [F2, 3], ...], int64
    components_index: [[F1,], [F2,], ...], int64, w.r.t. faces
    '''
    adjacency = face_adjacency(faces)
    if only_watertight:
        min_len = 4
    else:
        min_len = 1
    index, labels_counts = connected_components(edges=adjacency, min_len=min_len)
    labels_counts = labels_counts.tolist()
    components = torch.split(faces.index_select(0, index), labels_counts, dim=0)
    if return_index:
        components_index = torch.split(index, labels_counts, dim=0)
        return components, components_index
    else:
        return components


