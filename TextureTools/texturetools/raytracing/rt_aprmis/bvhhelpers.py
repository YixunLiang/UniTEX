import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import trimesh
import pyexr
import slangtorch
import time
import csv
import numpy as np

def get_bvh_m():
    m_gen_ele = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/get_elements.slang'))
    m_morton_codes = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/lbvh_morton_codes.slang'))
    m_radixsort = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/lbvh_single_radixsort.slang'))
    m_hierarchy = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/lbvh_hierarchy.slang'))
    m_bounding_box = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), 'bvhworkers/lbvh_bounding_boxes.slang'))
    
    return m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box

def get_bvh(vrt, v_ind, m_gen_ele, m_morton_codes, m_radixsort, m_hierarchy, m_bounding_box):
    #first part, get element and bbox---------------
    primitive_num = v_ind.shape[0]
    ele_primitiveIdx = torch.zeros((primitive_num, 1), dtype=torch.int).cuda()
    ele_aabb = torch.zeros((primitive_num, 6), dtype=torch.float).cuda()

    # Invoke normally
    m_gen_ele.generateElements(vert=vrt, v_indx=v_ind, ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((primitive_num+255)//256, 1, 1))
    extent_min_x = ele_aabb[:,0].min()
    extent_min_y = ele_aabb[:,1].min()
    extent_min_z = ele_aabb[:,2].min()

    extent_max_x = ele_aabb[:,3].max()
    extent_max_y = ele_aabb[:,4].max()
    extent_max_z = ele_aabb[:,5].max()
    num_ELEMENTS = ele_aabb.shape[0]
    #-------------------------------------------------
    #morton codes part
    pcMortonCodes = m_morton_codes.pushConstantsMortonCodes(
        g_num_elements=num_ELEMENTS, g_min_x=extent_min_x, g_min_y=extent_min_y, g_min_z=extent_min_z,
        g_max_x=extent_max_x, g_max_y=extent_max_y, g_max_z=extent_max_z
    )
    morton_codes_ele = torch.zeros((num_ELEMENTS, 2), dtype=torch.int).cuda()

    m_morton_codes.morton_codes(pc=pcMortonCodes, ele_aabb=ele_aabb, morton_codes_ele=morton_codes_ele)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

    #--------------------------------------------------
    # radix sort part
    morton_codes_ele_pingpong = torch.zeros((num_ELEMENTS, 2), dtype=torch.int).cuda()
    m_radixsort.radix_sort(g_num_elements=int(num_ELEMENTS), g_elements_in=morton_codes_ele, g_elements_out=morton_codes_ele_pingpong)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))

    #--------------------------------------------------
    # hierarchy
    num_LBVH_ELEMENTS = num_ELEMENTS + num_ELEMENTS - 1
    LBVHNode_info = torch.zeros((num_LBVH_ELEMENTS, 3), dtype=torch.int).cuda()
    LBVHNode_aabb = torch.zeros((num_LBVH_ELEMENTS, 6), dtype=torch.float).cuda()
    LBVHConstructionInfo = torch.zeros((num_LBVH_ELEMENTS, 2), dtype=torch.int).cuda()

    m_hierarchy.hierarchy(g_num_elements=int(num_ELEMENTS), ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb,
                        g_sorted_morton_codes=morton_codes_ele, g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, g_lbvh_construction_infos=LBVHConstructionInfo)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

    #--------------------------------------------------
    # bounding_boxes
    #'''
    tree_heights = torch.zeros((num_ELEMENTS, 1), dtype=torch.int).cuda()
    m_bounding_box.get_bvh_height(g_num_elements=int(num_ELEMENTS), g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                g_lbvh_construction_infos=LBVHConstructionInfo, tree_heights=tree_heights)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

    tree_height_max = tree_heights.max()
    for i in range(tree_height_max):
        m_bounding_box.get_bbox(g_num_elements=int(num_ELEMENTS), expected_height=int(i+1),
                            g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                    g_lbvh_construction_infos=LBVHConstructionInfo)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

    m_bounding_box.set_root(
                g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb)\
        .launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1)) 
    
    return LBVHNode_info, LBVHNode_aabb