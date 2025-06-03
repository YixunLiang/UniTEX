import torch

from ...raytracing import RayTracing
from .connected_components import split_faces
from ...raytracing.check_visibility import sphere_rt, sphere_rt_nvdiffrast


def visiable_connected_components(
    vertices:torch.Tensor, faces:torch.Tensor, 
    n_rays_or_cameras=1_000, 
    sample_offset=0.0, 
    method='raytracing',
) -> torch.Tensor:
    if method == 'raytracing':
        bvh = RayTracing(vertices, faces, backend='optix')
        rays_tid = sphere_rt(bvh, n_rays=n_rays_or_cameras, sample_offset=sample_offset)
    elif method == 'nvdiffrast':
        bvh = RayTracing(vertices, faces, backend='nvdiffrast', H=1024, W=1024)
        rays_tid = sphere_rt_nvdiffrast(bvh, n_cameras=n_rays_or_cameras, sample_offset=sample_offset)
    else:
        raise NotImplementedError(f'method {method} is not supported')
    faces_list, faces_index_list = split_faces(faces, return_index=True)
    faces = [f for (f, idx) in zip(faces_list, faces_index_list) \
        if torch.isin(rays_tid, idx).sum().item() > 0]
    assert len(faces) > 0, f'reconstructed mesh is empty'
    faces = torch.cat(faces, dim=0)
    return faces

