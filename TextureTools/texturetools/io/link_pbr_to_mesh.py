import os
from typing import Optional, Union
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial


def link_rgb_to_mesh(
    src_path:Union[str, trimesh.Trimesh], 
    rgb_path:Union[str, Image.Image], 
    dst_path:Optional[str]=None,
) -> trimesh.Trimesh:
    input_mesh:trimesh.Trimesh = trimesh.load(src_path, process=False, force='mesh') if isinstance(src_path, str) else input_mesh  # NOTE: not safe
    rgb = Image.open(rgb_path) if isinstance(rgb_path, str) else rgb_path  # NOTE: not safe
    input_mesh.visual.material = PBRMaterial(
        baseColorTexture=rgb.transpose(Image.FLIP_TOP_BOTTOM),
        metallicRoughnessTexture=None,
        normalTexture=None,
        baseColorFactor=None,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    input_mesh.merge_vertices()
    # input_mesh.fix_normals()
    # input_mesh.face_normals
    # input_mesh.vertex_normals
    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        input_mesh.export(dst_path)
    return input_mesh


def link_pbr_to_mesh(
    src_path:Union[str, trimesh.Trimesh], 
    albedo_path:Union[str, Image.Image], 
    metallic_roughness_path:Union[str, Image.Image], 
    bump_path:Union[str, Image.Image], 
    dst_path:Optional[str]=None,
) -> trimesh.Trimesh:
    input_mesh:trimesh.Trimesh = trimesh.load(src_path, process=False, force='mesh') if isinstance(src_path, str) else input_mesh  # NOTE: not safe
    albedo = Image.open(albedo_path) if isinstance(albedo_path, str) else albedo_path  # NOTE: not safe
    metallic_roughness = Image.open(metallic_roughness_path) if isinstance(metallic_roughness_path, str) else metallic_roughness_path  # NOTE: not safe
    bump = Image.open(bump_path) if isinstance(bump_path, str) else bump_path  # NOTE: not safe
    input_mesh.visual.material = PBRMaterial(
        baseColorTexture=albedo.transpose(Image.FLIP_TOP_BOTTOM),
        metallicRoughnessTexture=metallic_roughness.transpose(Image.FLIP_TOP_BOTTOM),
        normalTexture=bump.transpose(Image.FLIP_TOP_BOTTOM),
        baseColorFactor=None,
        metallicFactor=None,
        roughnessFactor=None,
    )
    input_mesh.merge_vertices()
    input_mesh.fix_normals()
    input_mesh.face_normals
    input_mesh.vertex_normals
    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        input_mesh.export(dst_path)
    return input_mesh



