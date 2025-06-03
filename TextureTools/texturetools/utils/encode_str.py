import numpy as np
import torch

def encode_str_to_array(text:str, width=64, fillchar='\x00'):
    return np.frombuffer(text.ljust(width, fillchar).encode('utf-8'), dtype=np.uint8)

def decode_array_to_str(array:np.ndarray, fillchar='\x00'):
    return array.tobytes().decode('utf-8').rstrip(fillchar)

def encode_str_to_tensor(text:str, width=64, fillchar='\x00'):
    return torch.from_numpy(np.frombuffer(text.ljust(width, fillchar).encode('utf-8'), dtype=np.uint8))

def decode_tensor_to_str(tensor:torch.Tensor, fillchar='\x00'):
    return tensor.data.cpu().numpy().tobytes().decode('utf-8').rstrip(fillchar)
