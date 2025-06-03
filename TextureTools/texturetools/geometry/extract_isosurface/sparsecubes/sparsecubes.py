import torch

# https://github.com/nv-tlabs/FlexiCubes.git
from ..flexicubes.flexicubes import FlexiCubes


class SparseCubes(FlexiCubes):
    '''
    NOTE: FlexiCubes is SparseCubes, just rewrite _get_case_id and __call__
    '''
    @torch.no_grad()
    def _get_case_id(
        self, 
        occ_fx8, 
        surf_cubes, 
        res,
    ):
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)
        return case_ids

    def __call__(
        self, 
        x_nx3, 
        s_n, 
        cube_fx8, 
        res=None, 
        beta_fx12=None, 
        alpha_fx8=None,
        gamma_f=None, 
        training=False, 
        output_tetmesh=False, 
        grad_func=None,
    ):
        return super().__call__(
            x_nx3, s_n, cube_fx8, res,
            beta_fx12=beta_fx12,
            alpha_fx8=alpha_fx8,
            gamma_f=gamma_f,
            training=training,
            output_tetmesh=output_tetmesh,
            grad_func=grad_func,
        )

