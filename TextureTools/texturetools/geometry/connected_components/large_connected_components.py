import numpy as np
import torch

from .connected_components import split_faces


def large_connnected_components(faces:torch.Tensor):
    faces_list = split_faces(faces)
    # NOTE: face areas from mc/dmc/mt/dmt are approximately uniform
    faces = faces_list[np.argmax([f.shape[0] for f in faces_list])]
    return faces

