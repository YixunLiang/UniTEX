import torch


def remove_unreferenced_vertices(vertices:torch.Tensor, faces:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    '''
    mask = torch.zeros((vertices.shape[0],), dtype=torch.bool, device=faces.device)
    mask.scatter_(0, faces.reshape(-1), True)
    V = mask.sum().item()
    if V == vertices.shape[0]:
        return vertices, faces
    index = torch.arange(V, dtype=torch.int64, device=faces.device)
    inverse = torch.zeros((vertices.shape[0],), dtype=torch.int64, device=faces.device)
    inverse.masked_scatter_(mask, index)
    faces = inverse[faces]
    vertices = vertices[mask, :]
    return vertices, faces


