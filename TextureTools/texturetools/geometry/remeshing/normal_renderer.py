import torch
import nvdiffrast.torch as dr

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    v1, v2, v3 = verts[faces].unbind(dim=1)
    fn = torch.linalg.cross(v2 - v1, v3 - v1, dim=-1)
    return fn

def compute_face_vertex_angles(verts, faces):
    """
    Compute per-face angles.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    v1, v2, v3 = verts[faces].unbind(dim=1)
    e1, e2, e3 = v2 - v1, v3 - v2, v1 - v3
    a1 = torch.arccos(torch.nn.functional.cosine_similarity(e1, -e3, dim=-1).clamp(-1.0, 1.0))
    a2 = torch.arccos(torch.nn.functional.cosine_similarity(e2, -e1, dim=-1).clamp(-1.0, 1.0))
    a3 = (torch.pi - a1 - a2).clamp(0.0, torch.pi)
    fva = torch.stack([a1, a2, a3], dim=1)
    return fva

def compute_vertex_normals(verts, faces, mode='area'):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    mode: area, angle, average
    """
    V = verts.shape[0]
    F = faces.shape[0]
    fn = compute_face_normals(verts, faces)
    if mode == 'area':
        fvn = fn[:, None, :].expand(-1, 3, 3)
    elif mode == 'angle':
        fn = torch.nn.functional.normalize(fn, dim=-1)
        fva = compute_face_vertex_angles(verts, faces)
        fvn = fn[:, None, :].expand(-1, 3, 3) * fva[:, :, None].expand(-1, 3, 3)
    elif mode == 'average':
        fn = torch.nn.functional.normalize(fn, dim=-1)
        fvn = fn[:, None, :].expand(-1, 3, 3)
    else:
        raise NotImplementedError(f'invalid mode {mode}')
    # NOTE: if include_self=False, view input as zeros
    vvn = torch.zeros((V, 3, 3), dtype=verts.dtype, device=verts.device)
    vvn = torch.scatter_reduce(vvn, dim=0, index=faces[:, :, None].expand(-1, 3, 3), src=fvn, reduce='sum', include_self=True)
    vn = vvn.sum(dim=1)
    return vn

class NormalRenderer:
    def __init__(self):
        self.glctx = dr.RasterizeGLContext()

    def __call__(self, v, f, mvp, H, W):
        vn = compute_vertex_normals(v, f)
        vn = torch.nn.functional.normalize(vn, dim=-1)
        v_hom = torch.cat([v, torch.ones_like(v[..., [0]])], dim=-1)
        v_ndc = torch.matmul(v_hom, mvp.transpose(-1, -2))
        f_int32 = f.to(dtype=torch.int32)
        rast = dr.rasterize(self.glctx, v_ndc, f_int32, (H, W))[0]
        attr = dr.interpolate(vn, rast, f_int32)[0]
        attr_aa = dr.antialias(attr, rast, v_ndc, f_int32, pos_gradient_boost=1.0)
        return attr_aa

