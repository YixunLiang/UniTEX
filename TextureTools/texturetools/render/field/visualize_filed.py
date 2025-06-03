from typing import Callable
import numpy as np

# pip install vedo
from vedo import Volume
from vedo.applications import IsosurfaceBrowser


def create_grid(field:Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], D:int, H:int, W:int) -> np.ndarray:
    '''
    vs: [D, H, W], float32
    '''
    xs = (np.linspace(-1.0, 1.0, W+1, dtype=np.float32)[:W] + (1.0 / W))
    ys = (np.linspace(-1.0, 1.0, H+1, dtype=np.float32)[:H] + (1.0 / H))[:, None]
    zs = (np.linspace(-1.0, 1.0, D+1, dtype=np.float32)[:D] + (1.0 / D))[:, None, None]
    xs, ys, zs = np.broadcast_arrays(xs, ys, zs)
    vs = field(xs, ys, zs)
    return vs

def example_field(xs:np.ndarray, ys:np.ndarray, zs:np.ndarray) -> np.ndarray:
    '''
    xs, ys, zs: [D, H, W], float32
    vs: [D, H, W], float32
    '''
    vs = xs ** 2 + ys ** 2 + zs ** 2
    return vs

def show_isosurface(data:np.ndarray):
    '''
    data: [D, H, W], float32
    '''
    assert data.ndim == 3
    vol = Volume(data)
    plt = IsosurfaceBrowser(vol, use_gpu=True)
    plt.show(axes=7, mode=0, title='show_grid').close()


if __name__ == '__main__':
    show_isosurface(create_grid(example_field, 512, 512, 512))

