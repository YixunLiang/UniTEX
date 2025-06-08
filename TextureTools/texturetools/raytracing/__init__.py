import os
from typing import Tuple
import torch


if len(os.environ.get('OptiX_INSTALL_DIR', '')) > 0:
    DEFAULT_RAY_TRACING_BACKEND = 'optix'
else:
    DEFAULT_RAY_TRACING_BACKEND = 'aprmis'


class RayTracing:
    def __init__(self, vertices:torch.Tensor, faces:torch.Tensor, backend=DEFAULT_RAY_TRACING_BACKEND, **kwargs):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        backend: optix, aprmis, nvdiffrast
        '''
        V, _ = vertices.shape
        F, _ = faces.shape
        if backend in ['optix', 'triro']:
            # https://github.com/lcp29/trimesh-ray-optix
            # https://developer.nvidia.com/designworks/optix/downloads/legacy
            # export OptiX_INSTALL_DIR=${HOME}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
            from triro.ray.ray_optix import RayMeshIntersector
            self.ray_tracing = RayMeshIntersector(vertices=vertices, faces=faces)
        elif backend in ['aprmis', 'slang']:
            # https://github.com/brabbitdousha/A-Python-Ray-Mesh-Intersector-in-Slangpy.git
            from .rt_aprmis import APRMISRayTracing
            self.ray_tracing = APRMISRayTracing(vertices=vertices, faces=faces)
        elif backend in ['nvdiffrast', 'nvdiff']:
            # https://github.com/NVlabs/nvdiffrast
            from .rt_nvdiffrast import NVDiffrastRayTracing
            self.ray_tracing = NVDiffrastRayTracing(
                vertices=vertices, faces=faces,
                cuda_or_gl=kwargs.get('cuda_or_gl', True), 
                H=kwargs.get('H', 512), 
                W=kwargs.get('W', 512), 
                perspective=kwargs.get('perspective', True), 
                fov=kwargs.get('fov', 49.1), 
                near=kwargs.get('near', 0.01), 
                far=kwargs.get('far', 1000.0), 
            )
        else:
            raise NotImplementedError(f'backend {backend} is not supported')
        self.backend = backend
        self.V = V
        self.F = F
    
    def update_raw(self, vertices:torch.Tensor, faces:torch.Tensor):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        '''
        return self.ray_tracing.update_raw(vertices=vertices, faces=faces)

    def intersects_closest(self, rays_o:torch.Tensor, rays_d:torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        rays_o: [N, 3], float32
        rays_d: [N, 3], float32

        hit: [N,] or [N, H, W], bool
            A boolean tensor indicating if each ray intersects with the mesh.
        front: [N,] or [N, H, W], bool
            A boolean tensor indicating if the intersection is from the front face of the mesh.
        tri_idx: [N,] or [N, H, W], int64
            The index of the triangle that was intersected by each ray.
        loc: [N, 3] or [N, H, W, 3], float32
            The 3D coordinates of the intersection point for each ray.
        uv: [N, 2] or [N, H, W, 2], float32
            The UV coordinates of the intersection point for each ray.
        '''
        if self.backend in ['optix', 'triro']:
            hit, front, tri_idx, loc, uv = self.ray_tracing.intersects_closest(origins=rays_o, directions=rays_d, stream_compaction=False)
        elif self.backend in ['nvdiffrast', 'nvdiff']:
            hit, front, tri_idx, loc, uv = self.ray_tracing.intersects_closest(rays_o=rays_o, rays_d=rays_d)
        elif self.backend in ['aprmis']:
            hit, front, tri_idx, loc, uv = self.ray_tracing.intersects_closest(rays_o=rays_o, rays_d=rays_d)
        else:
            raise NotImplementedError(f'backend {self.backend} is not supported')
        return hit, front, tri_idx, loc, uv


def ray_tracing(vertices:torch.Tensor, faces:torch.Tensor, rays_o:torch.Tensor, rays_d:torch.Tensor, backend='optix') \
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    vertices: [V, 3], float32
    faces: [F, 3], int64
    rays_o: [N, 3], float32
    rays_d: [N, 3], float32
    backend: optix, aprmis, nvdiffrast

    hit: [N,] or [N, H, W], bool
    front: [N,] or [N, H, W], bool
    tri_idx: [N,] or [N, H, W], int64
    loc: [N, 3] or [N, H, W, 3], float32
    uv: [N, 2] or [N, H, W, 2], float32
    '''
    return RayTracing(vertices, faces, backend=backend).intersects_closest(rays_o, rays_d)

