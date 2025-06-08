import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import trimesh
import pyexr
import slangtorch
import time
import csv
import numpy as np
from bvhhelpers import *

#load obj
mesh = trimesh.load('./bunny.obj')

vrt = torch.from_numpy(mesh.vertices).cuda().float()
v_ind = torch.from_numpy(mesh.faces).cuda().int()
#----
#load shaders first this can be slow at the first time
m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box = get_bvh_m()

# get bvh tree
start_time = time.time()
LBVHNode_info, LBVHNode_aabb = get_bvh(vrt, v_ind, m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU bvh build finished in: {elapsed_time} s")

print("bvh build over!")

#let's have a test
m_intersect_test = slangtorch.loadModule('bvhworkers/intersect_test.slang')
# generating rays
y, x = torch.meshgrid([torch.linspace(1, -1, 800), 
                       torch.linspace(-1, 1, 800)], indexing='ij')
z = -torch.ones_like(x)
ray_directions = torch.stack([x, y, z], dim=-1).cuda()
ray_origins = torch.Tensor([0, 0.1, 0.3]).cuda().broadcast_to(ray_directions.shape)

ray_origins = ray_origins.contiguous().reshape(-1,3)
ray_directions = ray_directions.contiguous().reshape(-1,3)

num_rays=ray_origins.shape[0]

hit = torch.zeros((num_rays, 1), dtype=torch.int).cuda()
hit_pos_map = torch.zeros((num_rays, 3), dtype=torch.float).cuda()

start_time = time.time()
m_intersect_test.intersect(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
               g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
               vert=vrt, v_indx=v_ind,
               hit_map=hit, hit_pos_map=hit_pos_map)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))
end_time = time.time()

elapsed_time = end_time - start_time
print("ray query time:", elapsed_time, "s")

# drawing result
#locs = hit.repeat(1,3)
locs = hit_pos_map
pyexr.write(f'./color.exr', locs.reshape(800,800,3).cpu().numpy())