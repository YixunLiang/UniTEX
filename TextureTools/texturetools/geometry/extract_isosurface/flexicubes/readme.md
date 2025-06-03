# Submodule of FlexiCubes

https://github.com/nv-tlabs/FlexiCubes/commit/241bf4927e53f59e55da5e5818dbb9359010b8cb

usage:
``` python
from flexicubes import FlexiCubes

fc = FlexiCubes()
depth = 8
x_nx3, cube_fx8 = fc.construct_voxel_grid(2 ** depth)
x_nx3 = x_nx3 * 2.0
sdf = torch.rand_like(x_nx3[:,0]) - 0.1
vertices, faces, _ = fc(x_nx3, sdf, cube_fx8, 2 ** depth)
```
