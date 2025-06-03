from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import trimesh
import open3d as o3d
from ...camera.generator import generate_box_views_c2ws, generate_intrinsics
from ...camera.conversion import c2w_to_w2c, intr_to_proj, c2ws_to_rays, inverse_transform, project, rays_to_c2ws, c2ws_to_ray_matrices
from ...io.mesh_loader import load_whole_mesh
from ...geometry.uv.uv_atlas import preprocess_blank_mesh_trimesh as preprocess_blank_mesh
from ...utils.parse_color import parse_color
from ...utils.timer import CPUTimer

# https://github.com/NVlabs/nvdiffrec
import nvdiffrast.torch as dr

# https://github.com/lcp29/trimesh-ray-optix
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# export OptiX_INSTALL_DIR=${HOME}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
from triro.ray.ray_optix import RayMeshIntersector


class GeometryRenderer:
    def __init__(self, backend='nvdiffrast', inverse_backend='open3d'):
        assert backend in ['nvdiffrast', 'nvdiffrast_cuda', 'nvdiffrast_opengl', 'optix']
        assert inverse_backend in ['open3d', 'optix']
        self.backend = backend
        self.inverse_backend = inverse_backend
        self.n_views = None
        self.vertices = None
        self.faces = None
        self.areas = None
        self.face_normals = None
        self.vertex_normals = None
        self.vertices_2d = None
        self.faces_2d = None
        self.c2ws = None
        self.intrinsics = None

    def add_cameras(self, n_views=6, fov_deg=49.1, scale=0.85, perspective=False):
        assert n_views in [1, 2, 3, 4, 6]
        self.n_views = n_views
        if self.n_views == 1:
            self.n_rows, self.n_cols = 1, 1
        elif self.n_views == 2:
            self.n_rows, self.n_cols = 1, 2
        elif self.n_views == 3:
            self.n_rows, self.n_cols = 1, 3
        elif self.n_views == 4:
            self.n_rows, self.n_cols = 2, 2
        elif self.n_views == 6:
            self.n_rows, self.n_cols = 2, 3
        else:
            raise NotImplementedError(f'n_views {self.n_views} is not supported')
        self.perspective = perspective
        if self.n_views == 1:
            c2ws = generate_box_views_c2ws()[[0], :, :]
        elif self.n_views == 2:
            c2ws = generate_box_views_c2ws()[[0, 2], :, :]
        elif self.n_views == 3:  # NOTE: front, left, back
            c2ws = generate_box_views_c2ws()[[0, 3, 2], :, :]
        elif self.n_views == 4:
            c2ws = generate_box_views_c2ws()[:4, :, :]
        elif self.n_views == 6:
            c2ws = generate_box_views_c2ws()
        self.c2ws = c2ws.to(dtype=torch.float32, device='cuda', memory_format=torch.contiguous_format)
        if self.perspective:
            intrinsics = generate_intrinsics(fov_deg, fov_deg, fov=True, degree=True)
        else:
            intrinsics = generate_intrinsics(scale, scale, fov=False, degree=False)
        self.intrinsics = intrinsics.to(dtype=torch.float32, device='cuda', memory_format=torch.contiguous_format)
        return self

    def add_mesh(self, vertices:torch.Tensor, faces:torch.Tensor, vertices_2d:torch.Tensor, faces_2d:torch.Tensor):
        self.vertices = vertices.to(dtype=torch.float32, device='cuda', memory_format=torch.contiguous_format)
        self.faces = faces.to(dtype=torch.int64, device='cuda', memory_format=torch.contiguous_format)
        self.areas = torch.linalg.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], vertices[faces[:, 2], :] - vertices[faces[:, 0], :], dim=-1)
        self.face_normals = torch.nn.functional.normalize(self.areas, dim=-1)
        vertex_normals = torch.zeros((vertices.shape[0], 3, 3),dtype=vertices.dtype, device=vertices.device)
        vertex_normals.scatter_add_(0, faces.unsqueeze(-1).expand(self.faces.shape[0], 3, 3), self.face_normals.unsqueeze(1).expand(self.faces.shape[0], 3, 3))
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals.sum(dim=1), dim=-1)
        self.vertices_2d = vertices_2d.to(dtype=torch.float32, device='cuda', memory_format=torch.contiguous_format)
        self.faces_2d = faces_2d.to(dtype=torch.int64, device='cuda', memory_format=torch.contiguous_format)
        return self

    @CPUTimer('render')
    def render(self, H:int, W:int, background='white') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        H, W: int
        background: str, float, tuple[float], list[float]
        alpha, ccm, normal: [N, H, W, C]
        '''
        if self.backend in ['nvdiffrast', 'nvdiffrast_cuda', 'nvdiffrast_opengl']:
            if self.backend in ['nvdiffrast', 'nvdiffrast_cuda']:
                ctx = dr.RasterizeCudaContext(device='cuda')
            elif self.backend == 'nvdiffrast_opengl':
                ctx = dr.RasterizeGLContext(device='cuda')
            else:
                ctx = None
            vertices_homo = torch.cat([self.vertices, torch.ones_like(self.vertices[:, :1])], dim=-1)
            vertices_clip = torch.matmul(vertices_homo, torch.matmul(intr_to_proj(self.intrinsics, perspective=self.perspective), c2w_to_w2c(self.c2ws)).permute(0, 2, 1))
            faces = self.faces.to(dtype=torch.int32)
            # NOTE: if H != W, render image will be resized directly
            rast_out, _ = dr.rasterize(ctx, vertices_clip, faces, (H, W))
            alpha = (rast_out[..., [3]] > 0).to(dtype=torch.float32)
            ccm, _ = dr.interpolate(self.vertices.contiguous(), rast_out, faces)
            normal, _ = dr.interpolate(self.vertex_normals.contiguous(), rast_out, faces)
        elif self.backend == 'optix':
            bvh_optix = RayMeshIntersector(vertices=self.vertices, faces=self.faces)
            rays_o, rays_d = c2ws_to_ray_matrices(self.c2ws, self.intrinsics, H, W, perspective=self.perspective)
            rays_mask, _, rays_tid, _, rays_uv = bvh_optix.intersects_closest(rays_o, rays_d, stream_compaction=False)
            rays_uv = torch.cat([rays_uv, 1.0 - rays_uv.sum(-1, keepdim=True)], dim=-1)
            rays_faces = self.faces[rays_tid, :]
            alpha = rays_mask.unsqueeze(-1).to(dtype=torch.float32)
            ccm = torch.sum(self.vertices[rays_faces, :] * rays_uv.unsqueeze(-1), dim=-2)
            normal = torch.sum(self.vertex_normals[rays_faces, :] * rays_uv.unsqueeze(-1), dim=-2)
        else:
            raise NotImplementedError(f'backend {self.backend} is not supported')
        background = parse_color(background)
        if background is not None:
            background = background.to(dtype=torch.float32, device='cuda')
            ccm = (ccm * 0.5 + 0.5) * alpha + background * (1.0 - alpha)
            normal = (normal * 0.5 + 0.5) * alpha + background * (1.0 - alpha)
        else:
            ccm = ccm * 0.5 + 0.5
            normal = normal * 0.5 + 0.5
        return alpha, ccm, normal

    def render_image(self, H:int, W:int, background='white') -> Tuple[Image.Image, Image.Image, Image.Image]:
        '''
        H, W: int
        background: str, float, tuple[float], list[float]
        alpha, ccm, normal: Image
        '''
        alpha, ccm, normal = self.render(H=H, W=W, background=background)
        alpha_im = alpha.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        alpha_im = Image.fromarray(alpha_im.reshape(self.n_rows, self.n_cols, H, W).transpose(0, 2, 1, 3).reshape(self.n_rows * H, self.n_cols * W), mode='L')
        ccm_im = ccm.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        ccm_im = Image.fromarray(ccm_im.reshape(self.n_rows, self.n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(self.n_rows * H, self.n_cols * W, 3), mode='RGB')
        normal_im = normal.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        normal_im = Image.fromarray(normal_im.reshape(self.n_rows, self.n_cols, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(self.n_rows * H, self.n_cols * W, 3), mode='RGB')
        return alpha_im, ccm_im, normal_im

    def inverse_render_image(self, H:int, W:int, images:Image.Image):
        H_im, W_im = images.size[1] // self.n_rows, images.size[0] // self.n_cols
        rgb_im = np.array(images.convert('RGB')).reshape(self.n_rows, H_im, self.n_cols, W_im, 3).transpose(0, 2, 1, 3, 4).reshape(self.n_rows * self.n_cols, H_im, W_im, 3)
        rgb = torch.as_tensor(rgb_im, dtype=torch.float32, device='cuda').div(255.0).clamp(0.0, 1.0)
        rgb_map = self.inverse_render(H=H, W=W, images=rgb)
        rgb_map_im = rgb_map.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        rgb_map_im = Image.fromarray(rgb_map_im.reshape(H, W, 4), mode='RGBA')
        return rgb_map_im

    @CPUTimer('inverse_render')
    def inverse_render(self, H:int, W:int, images:torch.Tensor) -> torch.Tensor:
        '''
        Reference:
        * https://github.com/nihalsid/image-to-atlas/blob/main/utils/renderer.py,
            backproject_to_atlas
        * https://github.com/AiuniAI/Unique3D/blob/main/scripts/project_mesh.py,
            multiview_color_projection
        * https://github.com/3DTopia/MVPaint
        * https://github.com/isl-org/Open3D/blob/main/cpp/open3d/t/geometry/TriangleMesh.cpp,
            TriangleMesh::ProjectImagesToAlbedo
        * https://github.com/Tencent/Hunyuan3D-2/blob/main/hy3dgen/texgen/differentiable_renderer/mesh_render.py,
            MeshRender.bake_texture
        '''
        N_im, H_im, W_im, C_im = images.shape
        if self.inverse_backend == 'open3d':
            device_o3d = o3d.core.Device('CPU:0')
            dtype_f = o3d.core.float32
            dtype_ui = o3d.core.uint8
            mesh_o3d = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy()), 
                triangles=o3d.utility.Vector3iVector(self.faces.cpu().numpy()),
            )
            mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals.detach().cpu().numpy())
            mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector(self.vertices_2d.mul(0.5).add(0.5)[self.faces_2d, :].reshape(-1, 2).detach().cpu().numpy())
            mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d, device=device_o3d)
            images = [o3d.t.geometry.Image(o3d.core.Tensor(
                im.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8),
                dtype=dtype_ui,
                device=device_o3d,
            )) for im in images]
            # NOTE: rt in open3d is not w2c
            extrinsics_rt = c2w_to_w2c(self.c2ws).detach().cpu().numpy()
            extrinsic_matrices = [o3d.core.Tensor(rt, dtype=dtype_f, device=device_o3d) for rt in extrinsics_rt]
            intrinsics_abs = np.asarray([W_im, H_im, 1.0], dtype=np.float32)[:, None] * self.intrinsics.detach().cpu().numpy()
            intrinsic_matrices = [o3d.core.Tensor(intrinsics_abs, dtype=dtype_f, device=device_o3d)] * len(self.c2ws)
            o3d.visualization.draw(
                [{
                    "name": 'geometry',
                    "geometry": mesh_o3d,
                }] + [{
                    "name": f"camera-{i:02}",
                    "geometry":
                        o3d.geometry.LineSet.create_camera_visualization(W_im, H_im, K.numpy(), Rt.numpy(), 1.0)
                } for i, (K, Rt) in enumerate(zip(intrinsic_matrices, extrinsic_matrices))],
                show_ui=True,
            )
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
                image_map = mesh_o3d.project_images_to_albedo(
                    images=images,
                    intrinsic_matrices=intrinsic_matrices,
                    extrinsic_matrices=extrinsic_matrices,
                    tex_size=max(H, W),
                    update_material=True,
                )
            image_map = image_map.as_tensor().numpy()
            image_map = torch.as_tensor(image_map, dtype=torch.float32, device='cuda').div(255.0).clamp(0.0, 1.0)
            image_map = torch.cat([image_map, torch.ones_like(image_map[..., [0]])], dim=-1)
        elif self.inverse_backend == 'optix':
            vertices = torch.cat([self.vertices_2d, torch.zeros_like(self.vertices_2d[..., [0]])], dim=-1)
            bvh_optix = RayMeshIntersector(vertices=vertices, faces=self.faces_2d)
            us = torch.linspace(-1.0, 1.0, W+1, dtype=torch.float32, device='cuda')[:W].add(1.0 / W).unsqueeze(-2)
            vs = torch.linspace(-1.0, 1.0, H+1, dtype=torch.float32, device='cuda')[:H].add(1.0 / H).unsqueeze(-1)
            rays_o = torch.stack([
                us * torch.ones_like(vs),
                torch.ones_like(us) * vs,
                torch.full_like(us, fill_value=-1.0) * torch.ones_like(vs),
            ], dim=-1)
            rays_d = torch.zeros_like(rays_o)
            rays_d[..., -1].fill_(1.0)
            rays_mask, _, rays_tid, _, rays_uv = bvh_optix.intersects_closest(rays_o, rays_d, stream_compaction=False)
            rays_uv = torch.cat([rays_uv, 1.0 - rays_uv.sum(-1, keepdim=True)], dim=-1)
            rays_faces = self.faces[rays_tid, :]
            alpha_map = rays_mask.unsqueeze(-1).to(dtype=torch.float32)
            ccm_map = torch.sum(self.vertices[rays_faces, :] * rays_uv.unsqueeze(-1), dim=-2)
            normal_map = torch.sum(self.vertex_normals[rays_faces, :] * rays_uv.unsqueeze(-1), dim=-2)
            e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
            cos_maps = torch.nn.functional.cosine_similarity(normal_map, -e3, dim=-1).unsqueeze(-1)
            xy_maps, _ = project(inverse_transform(torch.cat([ccm_map, torch.ones_like(ccm_map[..., [0]])], dim=-1), [self.c2ws.unsqueeze(-3)]), self.intrinsics, perspective=self.perspective)
            rgb_maps = torch.nn.functional.grid_sample(images.permute(0, 3, 1, 2), xy_maps).permute(0, 2, 3, 1)
            bvh_optix = RayMeshIntersector(vertices=self.vertices, faces=self.faces)
            o_maps = self.c2ws[..., :3, 3].unsqueeze(-2).unsqueeze(-2)
            d_maps = torch.nn.functional.normalize(ccm_map - o_maps, dim=-1)
            o_maps, d_maps = torch.broadcast_tensors(o_maps, d_maps)
            mask_maps, _, tid_maps, _, _ = bvh_optix.intersects_closest(o_maps, d_maps, stream_compaction=False)
            vmask_maps = torch.logical_and(mask_maps, tid_maps == rays_tid).unsqueeze(-1)
            weight_maps = torch.where(vmask_maps, 1.0, 0.0)
            rgb_map = torch.sum(weight_maps * rgb_maps, dim=0) / torch.sum(weight_maps, dim=0).clamp_min(1e-8)
            image_map = torch.cat([rgb_map, torch.ones_like(rgb_map[..., [0]])], dim=-1)
        elif self.inverse_backend == 'nvdiffrast':
            ...
        else:
            raise NotImplementedError(f'inverse_backend {self.inverse_backend} is not supported')
        return image_map


if __name__ == '__main__':
    gr = GeometryRenderer(backend='optix', inverse_backend='optix')
    gr.add_cameras(6, perspective=True)
    mesh_path = 'gradio_examples_mesh/horse_in_house/blank_mesh.glb'
    mesh_3d = load_whole_mesh(mesh_path)
    vertices, faces, uvs_2d, faces_2d = preprocess_blank_mesh(mesh_3d)
    gr.add_mesh(vertices, faces, uvs_2d, faces_2d)
    alpha, ccm, normal = gr.render_image(H=2048, W=2048, background='white')
    alpha.save('test_gr_alpha.png')
    ccm.save('test_gr_ccm.png')
    normal.save('test_gr_normal.png')
    normal_map = gr.inverse_render_image(2048, 2048, normal)
    normal_map.save('test_gr_normal_map_optix.png')
    gr.inverse_backend = 'open3d'
    normal_map = gr.inverse_render_image(2048, 2048, normal)
    normal_map.save('test_gr_normal_map_open3d.png')

