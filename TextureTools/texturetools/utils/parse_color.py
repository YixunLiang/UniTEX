from typing import List, Optional, Tuple, Union
import torch
from PIL.ImageColor import colormap

# NOTE: example: #f0f8ff
COLOR_DICT = {k: torch.tensor([int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)], dtype=torch.float32).div(255.0) for k, v in colormap.items()}
def parse_color(color:Optional[Union[str, float, Tuple[float], List[float]]]=None) -> Optional[torch.Tensor]:
    if color is None:
        return None
    if isinstance(color, str) and color in COLOR_DICT.keys():
        color = COLOR_DICT[color]
    elif isinstance(color, float):
        color = torch.tensor([color], dtype=torch.float32)  # NOTE: allow broadcast
    elif isinstance(color, (Tuple, List)) and len(color) == 3 and all(isinstance(c, float) for c in color):
        color = torch.tensor(color, dtype=torch.float32)
    else:
        raise NotImplementedError
    return color

