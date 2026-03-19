from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import trimesh
import torch.nn as nn
from einops import repeat, rearrange
from tqdm import trange
from itertools import product
import itertools

from LTM.craftsman.models.transformers.perceiver_1d import PerceiverWithFluxDec
from LTM.craftsman.utils.checkpoint import checkpoint
from LTM.craftsman.utils.base import BaseModule
from LTM.craftsman.utils.typing import *
###################### Utils

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False
#### code borrow from https://github.com/NVIDIA/Cosmos-Tokenizer/blob/main/cosmos_tokenizer/modules/patching.py

class Patcher(torch.nn.Module):
    """A module to convert image tensors into patches using torch operations.

    The main difference from `class Patching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Patching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()
        return x


class UnPatcher(torch.nn.Module):
    """A module to convert patches into image tensorsusing torch operations.

    The main difference from `class Unpatching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Unpatching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yl += torch.nn.functional.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh = torch.nn.functional.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh += torch.nn.functional.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        y = torch.nn.functional.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )
        y += torch.nn.functional.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x

# https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L162-L168
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


VALID_EMBED_TYPES = ["identity", "fourier", "learned_fourier", "siren"]

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class LearnedFourierEmbedder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        per_channel_dim = half_dim // input_dim
        self.weights = nn.Parameter(torch.randn(per_channel_dim))

        self.out_dim = self.get_dims(input_dim)

    def forward(self, x):
        # [b, t, c, 1] * [1, d] = [b, t, c, d] -> [b, t, c * d]
        freqs = (x[..., None] * self.weights[None] * 2 * np.pi).view(*x.shape[:-1], -1)
        fouriered = torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)
        return fouriered
    
    def get_dims(self, input_dim):
        return input_dim * (self.weights.shape[0] * 2 + 1)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Siren(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)
    
    def init_(self, weight, bias, c, w0):
        dim = self.in_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out
    
def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    elif embed_type == "learned_fourier":
        embedder_obj = LearnedFourierEmbedder(in_channels=input_dim, dim=num_freqs)
    
    elif embed_type == "siren":
        embedder_obj = Siren(in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim)

    else:
        raise ValueError(f"{embed_type} is not valid. Currently only supprts {VALID_EMBED_TYPES}")
    return embedder_obj


###################### AutoEncoder
class AutoEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        num_latents: int = 256
        embed_dim: int = 64
        width: int = 768
        tsdf_threshold: float = 0.
    cfg: Config

    def configure(self) -> None:
        super().configure()

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
 
    def forward(self,
                surface: torch.FloatTensor,
                queries: torch.FloatTensor,
                sample_posterior: bool = True,
                sharp_surface: torch.FloatTensor = None):
        shape_latents, kl_embed, posterior = self.encode(surface, sample_posterior=sample_posterior, sharp_surface=sharp_surface)

        latents = self.decode(kl_embed) # [B, num_latents, width]

        logits = self.query(queries, latents) # [B,]

        return shape_latents, latents, posterior, logits
    
    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

class PlaneDecoder(nn.Module):
    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 width: int,
                 layer: int,
                 grid_features: int,
                 embedder: FourierEmbedder,
                 plane_dim: int = 40,
                 plane_num: int = 3,
                 plane_res: int = 32*4,
                 use_plane_field: bool = False,
                 dual_latents: bool = False,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()
        self.plane_res = plane_res
        self.plane_dim = plane_dim
        self.use_checkpoint = use_checkpoint
        self.embedder = embedder
        # if plane_num == 3:
        self.dual_latents = dual_latents
        # if dual_latents:
        #     self.mlp_decoder = OSGDecoder(
        #             n_features=plane_dim*plane_num*2,
        #             out_dim=out_dim,
        #             hidden_dim=width,
        #             num_layers=layer,
        #             plane_num=plane_num,
        #             use_plane_field=use_plane_field,
        #             embedder = self.embedder,
        #         )
        # else:
        self.mlp_decoder = OSGDecoder(
                n_features=plane_dim,
                out_dim=out_dim,
                grid_features=grid_features,
                hidden_dim=width,
                num_layers=layer,
                plane_num=plane_num,
                use_plane_field=use_plane_field,
                embedder = self.embedder,
            )
        # else:
        # self.mlp_decoder = OSGDecoder(
        #     n_features=plane_dim*plane_num,
        #     out_dim=out_dim,
        #     hidden_dim=width,
        #     num_layers=layer,
        #     use_plane_field=use_plane_field,
        # )
        # elif plane_num == 1:
        #     self.mlp_decoder = OSGDecoder(
        #         n_features=(plane_dim+17),
        #         out_dim=out_dim,
        #         hidden_dim=width,
        #         num_layers=layer,
        #     )            
        self.proj_planes_sign = torch.tensor([[[1., -1.]],
                                        [[-1., -1.]],
                                        [[1., 1.]]])
        #
        # self.proj_grid_sign = torch.tensor([[[1., -1., -1.]],])


    def query_plane(self,queries: torch.FloatTensor, plane: torch.FloatTensor, plane_num: int):
        # indices2D = torch.stack(
        #     (-queries[..., [0, 1]], -queries[..., [2, 1]], queries[..., [0, 2]]),
        #     dim=-3,
        # ).to(plane.dtype)
        # breakpoint()
        if plane_num == 3:
            indices2D = torch.stack(
                (queries[..., [0, 1]], queries[..., [2, 1]], queries[..., [0, 2]]),
                dim=-3,
            ).to(plane.dtype)
            proj_sign = self.proj_planes_sign.unsqueeze(0).to(indices2D)
        elif plane_num == 2:
            indices2D = torch.stack(
                (queries[..., [0, 1]], queries[..., [2, 1]]),
                dim=-3,
            ).to(plane.dtype)
            proj_sign = self.proj_planes_sign[:plane_num].unsqueeze(0).to(indices2D)     
        elif plane_num == 1:  
            indices2D = torch.stack(
                (queries[..., [0, 1]],),
                dim=-3,
            ).to(plane.dtype)
            proj_sign = self.proj_planes_sign[:plane_num].unsqueeze(0).to(indices2D) 

        indices2D = indices2D * proj_sign
        #from torchvision.utils import save_image
        out = F.grid_sample(
            rearrange(plane, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np = plane_num).float(), # cat the same position plane to save times
            rearrange(indices2D, "B Nc N Nd -> (B Nc) () N Nd", Nc = plane_num).float(),#【W H】
            align_corners=True,
            mode="bilinear",
            padding_mode="border"
        ).to(plane.dtype)
        out = rearrange(out, "(B Np) Cp () N -> B N Np Cp", Np=plane_num).contiguous()
        return out


    def query_grid(self,queries: torch.FloatTensor, latents: torch.FloatTensor):
        batch_size, n_coords, n_dims = queries.shape
        sampled_features = torch.nn.functional.grid_sample(latents.expand(batch_size, -1, -1, -1, -1).float(),
                                                       queries.reshape(batch_size, 1, 1, -1, n_dims).float(), #【W H D】
                                                       mode='bilinear', padding_mode='border', align_corners=True).to(latents.dtype)
        N, C, H, W, D = sampled_features.shape
        sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
        return sampled_features

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        # transform plane token to plane
        # breakpoint()
        if isinstance(latents,list):
            latents, grid_latents = latents[0], latents[1]
        else:
            grid_latents = None
        plane_num = latents.shape[-1]//latents.shape[-2]
        qual_planes = rearrange(latents, "B Ci Hp (Np Wp) -> B Np Ci Hp Wp", Np=plane_num).contiguous()
        queried_features = self.query_plane(queries, qual_planes, plane_num = plane_num)


        if grid_latents is not None:
            grid_features = self.query_grid(queries,grid_latents)
        # if plane_num == 1:
        #     queried_features = torch.cat([queried_features,self.embedder(queries[...,-1:])],dim=-1)
        # elif plane_num == 2:
        #     queried_features = torch.cat([queried_features],dim=-1)
        # elif plane_num == 3:
        # queried_features = torch.cat([queried_features],dim=-1)
        x = self.mlp_decoder(queried_features,grid_features)
        return x

    # def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
    #     return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class OSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int, use_plane_field: bool, embedder: FourierEmbedder,plane_num: int, grid_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, out_dim: int = 1, skips=[4], activation: nn.Module = nn.ReLU, geometry_init: bool=False):
        super().__init__()
        self.skips = skips
        self.plane_num = plane_num
        self.pts_linears = nn.ModuleList(
            [nn.Linear(n_features*(plane_num)+grid_features, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim)
                if i not in self.skips else nn.Linear(hidden_dim + n_features*(plane_num)+grid_features, hidden_dim) for i in range(num_layers-1)])
        # self.feat_in_layers = nn.Linear(n_features-(17*3), hidden_dim)
        self.output_layers = nn.Linear(hidden_dim, out_dim)
        self.ln_post = nn.LayerNorm(n_features*(plane_num)+grid_features)
        self.out_dim = out_dim
        self.activation = nn.Softplus(beta=100)
        self.use_plane_field = use_plane_field
        self.n_features = n_features


        if self.use_plane_field:
            self.embedder = embedder
            self.plane_field_in = nn.Linear((n_features//plane_num)+17, n_features//plane_num) # hard code for embedding
            torch.nn.init.constant_(self.plane_field_in.bias, 0.0)
            torch.nn.init.constant_(self.plane_field_in.weight, 0.0)
            # torch.nn.init.normal_(self.plane_field_in.weight[:, :1], 0.0, np.sqrt(2) / np.sqrt((n_features+self.embedder.out_dim)//3))
        bias = 0.5
        if geometry_init:
            for l in range(len(self.pts_linears)):
                if l == len(self.pts_linears) - 1:
                    torch.nn.init.normal_(self.pts_linears[l].weight, mean=np.sqrt(np.pi) / np.sqrt(n_features), std=0.0001)
                    torch.nn.init.constant_(self.pts_linears[l].bias, -bias)
                if l == 0:
                    torch.nn.init.constant_(self.pts_linears[l].bias, 0.0)
                    torch.nn.init.constant_(self.pts_linears[l].weight[:, 3:], 0.0)
                    torch.nn.init.normal_(self.pts_linears[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(n_features))
                else:
                    torch.nn.init.constant_(self.pts_linears[l].bias, 0.0)
                    torch.nn.init.normal_(self.pts_linears[l].weight, 0.0, np.sqrt(2) / np.sqrt(n_features))
            torch.nn.init.constant_(self.output_layers.bias, 0.0)
            torch.nn.init.normal_(self.output_layers.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))            
        # if out_dim == 4:
        #     self.output_act = F.relu() # nn.Identity() 

    def forward(self, x, x_grid, query = None):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
         #= nn.LayerNorm(hidden_dim)
        
        if self.use_plane_field:
            # query[0,1,2] = x y z
            #(queries[..., [0, 1]], queries[..., [2, 1]], queries[..., [0, 2]]),
            if self.plane_num ==3:
                x_extra = torch.cat([self.embedder(query[...,(2,0,1)].unsqueeze(-1)),x],dim=-1)
            elif self.plane_num ==2:
                x_extra = torch.cat([self.embedder(query[...,(2,0)].unsqueeze(-1)),x],dim=-1)
            x = self.plane_field_in(x_extra) + x
            x = rearrange(x, "B N Np Cp -> B N (Np Cp)").contiguous()
        else:
            x = rearrange(x, "B N Np Cp -> B N (Np Cp)").contiguous()
        x = torch.cat([x,x_grid],dim=-1)
        x = self.ln_post(x)
        h = x
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = self.activation(x)
            if i in self.skips:
                x = torch.cat([x, h], -1)
        x = self.output_layers(x)
        rgb = x #*(1+2*0.001)-0.001
        # if self.out_dim == 3:
        #rgb = torch.sigmoid(x)*(1+2*0.001)-0.001
        return rgb

class FluxAutoencoderUnet(AutoEncoder):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        n_samples: int = 4096
        plane_res: int = 32
        plane_dim_high: int = 40
        plane_num: int = 3
        use_downsample: bool = False
        downsample_ratio: float = 0.0625
        num_latents: int = 256
        point_feats: int = 0
        embed_point_feats: bool = False
        out_dim: int = 1
        embed_dim: int = 64
        sample_type: str = "sample_posterior"
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        width: int = 768
        width_querier: int = 64
        layer_querier: int = 8
        heads: int = 12
        num_encoder_layers: int = 8
        num_decoder_layers: int = 16
        init_scale: float = 0.25
        qkv_bias: bool = True
        use_ln_post: bool = False
        use_flash: bool = False
        use_checkpoint: bool = True
        use_multi_reso: Optional[bool] = False
        resolutions: Optional[List[int]] = None
        sampling_prob: Optional[List[float]] = None
        with_sharp_data: Optional[bool] = False
        use_flux_latent_as_query: bool = False
        use_lora: bool = True

        # additional configs for flux_dec
        pretrain_flux_name_or_path: str = ""
        act_fn: str = "silu"
        block_out_channels: Tuple[int] = field(default_factory=lambda:[256, 512, 512]) #,512 #128,
        down_block_types: Tuple[str] = field(default_factory=lambda:[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ])
        force_upcast: bool = True
        in_channels: int = 3
        latent_channels: int =16
        latents_mean: Optional[Tuple[float]] = None
        latents_std: Optional[Tuple[float]] = None
        layers_per_block: int = 2
        mid_block_add_attention: bool = True
        norm_num_groups: int = 32
        out_channels: int = 3
        sample_size: int = 1024
        scaling_factor: float = 0.3611
        shift_factor: float = 0.1159
        up_block_types: Tuple[str] = field(default_factory=lambda:[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ])
        tsdf_threshold: float = 0.0
        enc_residual: bool = True
        # mask ratio
        use_mask: bool=False
        use_mask_ratio: float = 0.
        # mask_ratio: float = 0.
        mask_ratios: Optional[List[float]] = None
        mask_ratio_sampling_prob: Optional[List[float]] = None
        #\
        direct_query: bool = False
        use_baselatent: bool = False
        use_plane_field: bool = False
        use_mask_for_embedding: bool=False
        post_query: bool = True
        compress: bool = False
        dual_latents: bool = False
        random_mask_kl_ratio: float = 0.5
        only_flux_latent_ratio: float = 0.15
        plane_num_input: int = 2
        add_plane_num: int = 1


    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.embedder = get_embedder(embed_type=self.cfg.embed_type, num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)

        # encoder
        self.cfg.init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        if self.cfg.embed_dim > 0:
            # VAE embed
            if self.cfg.sample_type != "sample_posterior":
                self.pre_kl = nn.Identity() #nn.Conv2d(self.cfg.width, self.cfg.embed_dim*2,1,1,0) #nn.Linear(self.cfg.width, self.cfg.embed_dim*2)
                self.post_kl = nn.Identity()#nn.Linear(self.cfg.embed_dim, self.cfg.width//2)
                self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
            else:
                self.pre_kl = nn.Identity() #nn.Conv2d(self.cfg.width, self.cfg.embed_dim*2,1,1,0)
                self.post_kl = nn.Identity()#nn.Linear(self.cfg.embed_dim, self.cfg.width//2)
                self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        else:
            self.latent_shape = (self.cfg.num_latents, self.cfg.width)

        self.decoder = PerceiverWithFluxDec(
            n_ctx=self.cfg.num_latents,
            embed_dim= self.cfg.embed_dim,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            embedder=self.embedder,
            direct_query=self.cfg.direct_query,
            plane_res = self.cfg.plane_res,
            plane_dim = self.cfg.plane_dim_high,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
            in_channels=self.cfg.latent_channels,
            out_channels=self.cfg.out_channels,
            up_block_types=self.cfg.up_block_types,
            block_out_channels=self.cfg.block_out_channels,
            layers_per_block=self.cfg.layers_per_block,
            norm_num_groups=self.cfg.norm_num_groups,
            act_fn=self.cfg.act_fn,
            mid_block_add_attention=self.cfg.mid_block_add_attention,
            post_query = self.cfg.post_query,
            plane_num = self.cfg.plane_num,
            compress = self.cfg.compress,
            add_plane_num = self.cfg.add_plane_num
        )

        # decoder
        self.querier = PlaneDecoder(
            out_dim=self.cfg.out_dim,
            embedder=self.embedder,
            grid_features=self.cfg.width//8,
            num_latents=self.cfg.num_latents,
            plane_dim=self.cfg.plane_dim_high,
            plane_num=self.cfg.plane_num,
            dual_latents=self.cfg.dual_latents,
            plane_res = self.cfg.plane_res*8, # high res for 4 * plane
            width=self.cfg.width_querier,
            layer=self.cfg.layer_querier,
            use_plane_field=self.cfg.use_plane_field,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )
        #
        # augmentations
        self.resolutions = self.cfg.resolutions
        self.sampling_prob = self.cfg.sampling_prob
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        # breakpoint       
        if self.cfg.use_plane_field:
                        #self.plane_field_in = nn.Linear((n_features+self.embedder.out_dim)//3, n_features//3) # hard code for embedding
            torch.nn.init.constant_(self.querier.mlp_decoder.plane_field_in.bias, 0.0)
            torch.nn.init.constant_(self.querier.mlp_decoder.plane_field_in.weight, 0.0)

        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained VAE model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            else:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt.items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt 

            self.load_state_dict(pretrained_ckpt, strict=False)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def decode(self, 
               latents: torch.FloatTensor,
               sharp_surface: torch.FloatTensor,
               surface: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(latents) # [B, num_latents, embed_dim] -> [B, num_latents, width]
        latents = self.decoder(latents, surface, sharp_surface)
        return latents

    def query(self, 
              queries: torch.FloatTensor, 
              latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3] 
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], occupancy logits
        """

        logits = self.querier(queries, latents) #.squeeze(-1)

        return logits

    def forward(self,
                surface: torch.FloatTensor,
                queries: torch.FloatTensor,
                flux_latents: torch.FloatTensor,
                sample_posterior: bool = True,
                sharp_surface: torch.FloatTensor = None):

        #shape_latents = self.encode(surface, flux_latents = flux_latents, sample_posterior=sample_posterior, sharp_surface=sharp_surface)
        shape_latents = 0.
        latents = self.decode(flux_latents, surface, sharp_surface) # [B, num_latents, width]
        logits = self.query(queries, latents) # [B,]

        return shape_latents, latents, logits