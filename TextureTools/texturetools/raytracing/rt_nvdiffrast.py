import math
from typing import Tuple
import numpy as np
import trimesh
from PIL import Image
import torch

# https://github.com/NVlabs/nvdiffrast.git
import nvdiffrast.torch as dr
try:
    cuda_ctx = dr.RasterizeCudaContext(device='cuda')
except:
    cuda_ctx = None
try:
    gl_ctx = dr.RasterizeGLContext(device='cuda')
except:
    gl_ctx = None


class NVDiffrastRayTracing:
    def __init__(
        self, vertices:torch.Tensor, faces:torch.Tensor, 
        cuda_or_gl=True, H=512, W=512, perspective=True, fov=49.1, near=0.01, far=1000.0,
    ):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        '''
        if cuda_or_gl:
            self.ctx = cuda_ctx
        else:
            self.ctx = gl_ctx
        assert self.ctx is not None, f'initialize nvdiffrc failed'
        self.vertices = vertices.float().contiguous().cuda()
        self.faces = faces.int().contiguous().cuda()
        if perspective:
            self.proj = torch.as_tensor([
                [1 / (2 * math.tan(math.radians(fov) / 2)), 0.0, 0.0, 0.0], 
                [0.0, -1 / (2 * math.tan(math.radians(fov) / 2)), 0.0, 0.0], 
                [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
                [0.0, 0.0, -1.0, 0.0],
            ], dtype=torch.float32, device='cuda')
        else:
            raise NotImplementedError('not supported now')
        self.e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda')
        self.e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
        self.zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
        self.H = H
        self.W = W
    
    def update_raw(self, vertices:torch.Tensor, faces:torch.Tensor):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        '''
        self.vertices = vertices.float().contiguous().cuda()
        self.faces = faces.int().contiguous().cuda()
    
    def _rays_to_c2ws(self, rays_o:torch.Tensor, rays_d:torch.Tensor) -> torch.Tensor:
        '''
        rays_o, rays_d: [..., 3], camera locations looking at origin
        c2ws: [..., 4, 4]
            * world: x forward, y right, z up, need to transform xyz to zxy
            * camera: z forward, x right, y up
        '''
        rays_o, rays_d = torch.broadcast_tensors(rays_o, rays_d)
        batch_shape = rays_o.shape[:-1]
        # NOTE: camera locations are opposite from ray directions
        z_axis = torch.nn.functional.normalize(rays_d.neg(), dim=-1)
        x_axis = torch.linalg.cross(self.e3, z_axis, dim=-1)
        x_axis_mask = (x_axis == 0).all(dim=-1, keepdim=True)
        if x_axis_mask.sum() > 0:
            # NOTE: top and down is not well defined, hard code here
            x_axis = torch.where(x_axis_mask, self.e2, x_axis)
        y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
        rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)[..., [1, 2, 0], :]
        trans = rays_o[..., [1, 2, 0]]
        c2ws = torch.cat([
            torch.cat([rots, trans.unsqueeze(-1)], dim=-1),
            self.zzzo.expand(batch_shape + (1, -1)),
        ], dim=1)
        return c2ws

    def intersects_closest(self, rays_o:torch.Tensor, rays_d:torch.Tensor, stream_compaction=False) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        rays_o: [N, 3], float32
        rays_d: [N, 3], float32

        hit: [N, H, W], bool
            A boolean tensor indicating if each ray intersects with the mesh.
        front: [N, H, W], bool
            A boolean tensor indicating if the intersection is from the front face of the mesh.
        tri_idx: [N, H, W], int64
            The index of the triangle that was intersected by each ray.
        loc: [N, H, W, 3], float32
            The 3D coordinates of the intersection point for each ray.
        uv: [N, H, W, 2], float32
            The UV coordinates of the intersection point for each ray.
        '''
        c2ws = self._rays_to_c2ws(rays_o, rays_d)
        batch_shape = c2ws.shape[:-2]
        c2ws = c2ws.reshape(-1, 4, 4)
        v_pos_homo = torch.cat([self.vertices, torch.ones_like(self.vertices[:, :1])], dim=-1)
        v_pos_clip = torch.matmul(v_pos_homo, torch.matmul(self.proj, torch.linalg.inv(c2ws)).permute(0, 2, 1))
        with torch.no_grad():
            out, _ = dr.rasterize(self.ctx, v_pos_clip, self.faces, (self.H, self.W))
        # NOTE: face indices start from 1 in opengl
        hit = (out[:, :, :, -1] > 0).reshape(*batch_shape, self.H, self.W)
        front = None
        tri_idx = out[:, :, :, -1].to(dtype=torch.int64).sub(1).reshape(*batch_shape, self.H, self.W)
        loc = None
        uv = out[:, :, :, :2].reshape(*batch_shape, self.H, self.W, 2)
        return hit, front, tri_idx, loc, uv


if __name__ == '__main__':
    mesh = trimesh.load('gradio_examples_mesh/cute_wolf/textured_mesh.glb', process=False, force='mesh')
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device='cuda')

    nvdiffrt = NVDiffrastRayTracing(vertices, faces, height=1024, width=1024)
    rays_o = torch.as_tensor([
        [2.8, 0.0, 0.0],
        [0.0, 2.8, 0.0],
        [-2.8, 0.0, 0.0],
        [0.0, -2.8, 0.0],
        [0.0, 0.0, 2.8],
        [0.0, 0.0, -2.8],
    ], dtype=torch.float32, device='cuda')
    rays_d = torch.as_tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    hit, _, _, _, _ = nvdiffrt.intersects_closest(rays_o, rays_d)
    hit_im = Image.fromarray(hit.reshape(2, 3, 1024, 1024).permute(0, 2, 1, 3).reshape(2*1024, 3*1024).mul(255.0).detach().cpu().numpy().astype(np.uint8))
    hit_im.save('debug.png')

