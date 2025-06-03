import os
import torch
import slangtorch
from .bvhhelpers import get_bvh_m, get_bvh

m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box = get_bvh_m()
m_intersect_test = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/intersect_test2.slang'))


class APRMISRayTracing:
    def __init__(self, vertices:torch.Tensor, faces:torch.Tensor):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        '''
        self.vertices = vertices.float().contiguous().cuda()
        self.faces = faces.int().contiguous().cuda()
        self.LBVHNode_info, self.LBVHNode_aabb = get_bvh(
            self.vertices, self.faces, 
            m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box
        )

    def update_raw(self, vertices:torch.Tensor, faces:torch.Tensor):
        '''
        vertices: [V, 3], float32
        faces: [F, 3], int64
        '''
        self.vertices = vertices.float().contiguous().cuda()
        self.faces = faces.int().contiguous().cuda()
        self.LBVHNode_info, self.LBVHNode_aabb = get_bvh(
            self.vertices, self.faces, 
            m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box
        )

    def intersects_closest(self, rays_o:torch.Tensor, rays_d:torch.Tensor):
        '''
        rays_o: [N, 3], float32
        rays_d: [N, 3], float32

        hit: [N,], bool
            A boolean tensor indicating if each ray intersects with the mesh.
        front: [N,], bool
            A boolean tensor indicating if the intersection is from the front face of the mesh.
        tri_idx: [N,], int64
            The index of the triangle that was intersected by each ray.
        loc: [N, 3], float32
            The 3D coordinates of the intersection point for each ray.
        uv: [N, 2], float32
            The UV coordinates of the intersection point for each ray.
        '''
        rays_o, rays_d = torch.broadcast_tensors(rays_o, rays_d)
        batch_shape = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().cuda().reshape(-1, 3)
        rays_d = rays_d.contiguous().cuda().reshape(-1, 3)
        num_rays = rays_o.shape[0]

        hit = torch.zeros((num_rays, 1), dtype=torch.bool, device='cuda')
        hit_tid_map = torch.full((num_rays, 1), fill_value=-1, dtype=torch.int32, device='cuda')
        hit_pos_map = torch.zeros((num_rays, 3), dtype=torch.float32, device='cuda')
        hit_uv_map = torch.zeros((num_rays, 2), dtype=torch.float32, device='cuda')

        m_intersect_test.intersect(
            num_rays=int(num_rays), 
            rays_o=rays_o, 
            rays_d=rays_d,
            g_lbvh_info=self.LBVHNode_info, 
            g_lbvh_aabb=self.LBVHNode_aabb,
            vert=self.vertices, 
            v_indx=self.faces,
            hit_map=hit, 
            hit_tid_map=hit_tid_map, 
            hit_pos_map=hit_pos_map, 
            hit_uv_map=hit_uv_map
        ).launchRaw(
            blockSize=(256, 1, 1), 
            gridSize=((num_rays+255)//256, 1, 1),
        )

        hit = hit.reshape(*batch_shape)
        front = None
        tri_idx = hit_tid_map.to(dtype=torch.int64).reshape(*batch_shape)
        loc = hit_pos_map.reshape(*batch_shape, 3)
        uv = hit_uv_map.reshape(*batch_shape, 2)
        return hit, front, tri_idx, loc, uv


