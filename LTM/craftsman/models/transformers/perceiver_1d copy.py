import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LTM_old.craftsman.utils.typing import *
from LTM_old.craftsman.utils.checkpoint import checkpoint
from LTM_old.craftsman.models.transformers.attention import ResidualCrossAttentionBlock, ResidualCrossAttentionLoRABlock
import collections
import itertools

from .utils import init_linear
from .attention import  ResidualAttentionBlock
from einops import rearrange,repeat

from diffusers.loaders import PeftAdapterMixin
from diffusers.models.autoencoders.vae import UNetMidBlock2D as _UNetMidBlock2D
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D as _UpDecoderBlock2D
###################### Utils

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable): #works on pytorch 1.9.1
            return x
        return tuple(itertools.repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class UNetMidBlock2D(_UNetMidBlock2D,PeftAdapterMixin):
    ...

class UpDecoderBlock2D(_UpDecoderBlock2D,PeftAdapterMixin):
    ...

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


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    use_flash=use_flash,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class PixelShuffleUpsampleNetwork(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        out_channels: int,
        scale_factor: int = 8,
        conv_layers: int = 4,
        conv_kernel_size: int = 3,
        plane_res: int = 32,
        use_checkpoint: bool = False
    ):
        super().__init__()
        layers = []
        output_channels = out_channels * scale_factor**2
        self.scale = scale_factor
        self.plane_res = plane_res
        self.use_checkpoint = use_checkpoint
        in_channels = width
        self.conv_skip = nn.Conv2d(in_channels,out_channels,3,1,1)
        for i in range(conv_layers):
            cur_out_channels = (
                in_channels if i != conv_layers - 1 else output_channels
            )
            layers.append(
                nn.Conv2d(
                    in_channels,
                    cur_out_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                )
            )
            if i != conv_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.PixelShuffle(scale_factor))

        self.upsample = nn.Sequential(*layers)
    
    def _forward(self, x: Float[Tensor, "B Ci Hp Wp"]): # B,N,C
        # transform plane token to plane
        #x = rearrange(plane_tokens, "B (Np Hp Wp) Ci -> (B Np) Ci Hp Wp", Np=3, Hp=self.plane_res).contiguous()
        return self.upsample(x) + self.conv_skip(F.interpolate(x,scale_factor=self.scale))

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


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


class PerceiverWithFluxDec(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        embed_dim: int,
        embedder: FourierEmbedder,
        direct_query: bool,
        init_scale: float = 0.25,
        plane_dim: int = 40,
        compress: bool = False,
        plane_num: int = 3,
        plane_res: int = 32,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False,
        post_query: bool = False,
        # cofigs for vae decoders
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        add_plane_num: int = 1
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.plane_res = plane_res
        self.direct_query = direct_query
        self.embedder = embedder
        self.Patch_embedding = PatchEmbed(img_size=128,patch_size=4 ,in_chans=16*7,embed_dim=width)
        if not direct_query:
            self.pos_emb = nn.Parameter(torch.randn((width,plane_res//2,plane_res*6//2)) * (1. /(10*width)) ** 0.5)
            self.grid_emb = nn.Parameter(torch.randn((16**3,width)) * (1. /(10*width))** 0.5)
            self.ln_post = nn.LayerNorm(width)
            self.kv_project = nn.Linear(embedder.out_dim+3,width)
            self.kv_add_project = nn.Conv2d(width,width,1,1,0)
            self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
            )
            self.self_attn = Perceiver(
            n_ctx=3072,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
            )
            self.UpsampleNetwork = PixelShuffleUpsampleNetwork(
            width=width,
            out_channels=plane_dim//2,
            plane_res = plane_res,
            use_checkpoint=use_checkpoint,
            scale_factor=8,
            )

        self.add_plane_num = add_plane_num
        self.downsample = Patcher(patch_size = 4)#nn.Conv2d(width,width,2,2,0,bias=False)
        self.upsample = UnPatcher(patch_size = 4)
        self.post_query = post_query
        self.compress = compress
        self.VoxelShuffle = PixelShuffle3d(scale=2)
        self.mask_token = nn.Parameter(torch.randn((3,)) * 0.002)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor,surface: torch.Tensor, sharp_surface: torch.Tensor):
        input_plane = x
        pc,feats = surface[..., :3], surface[..., 3:]
        bs, N, D = pc.shape
        feats_mask = feats.sum(dim=-1)
        mask_tokens = self.mask_token.unsqueeze(0)
        feats[feats_mask ==6.] = mask_tokens
        data = self.embedder(pc)
        data = torch.cat([data, feats], dim=-1)
        # data = self.input_proj(data)
        bs = x.shape[0]
        Np = x.shape[-1]//x.shape[-2]
        x_down = self.downsample(x)
        x = self.Patch_embedding(x_down)
        pos_embs = repeat(self.pos_emb,'c h w-> b c h w',b=bs) #*10.
        pos_embs = self.kv_add_project(x+pos_embs)
        h_size,w_size = pos_embs.shape[-2],pos_embs.shape[-1]
        grid_querys = repeat(self.grid_emb,'n c -> b n c',b=bs) #*10.
        x = rearrange(pos_embs,'B C H W -> B (H W) C').contiguous()
        plane_querys_tok_length = x.shape[1]
        x = torch.cat([x,grid_querys],dim=1)
        # mask_tokens = self.mask_token.unsqueeze(0)
        x_data = self.kv_project(data)
        # x_data[feats_mask ==6.] = mask_tokens
        x_data_length = x_data.shape[1]
        x = self.cross_attn(x, x_data)
        x = self.self_attn(x)
        x_plane, x_grid = x[:,:plane_querys_tok_length], x[:,plane_querys_tok_length:]
        x_plane = rearrange(x_plane,'B (H W) C -> B C H W',H=h_size,W=w_size).contiguous()
        x_grid = rearrange(x_grid,'B (D H W) C -> B C D H W',H=16,W=16,D=16).contiguous()
        x_grid = self.VoxelShuffle(x_grid)
        # 
        #
        breakpoint()
        x_plane = self.UpsampleNetwork(x_plane)
        x_plane = rearrange(x_plane,'B C H (Np Ns W) -> B (Ns C) H (Np W)',Np = Np//2,Ns=2).contiguous()
        return [x_plane, x_grid]

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.double_conv(x)


class MLP(nn.Module):
    def __init__(self,width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class PatchEmbed(nn.Module):
    """ Temporal Images to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=4, embed_dim=1024):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.process = (DoubleConv(embed_dim, embed_dim))
        # nn.init.constant_(self.mlp.c_proj.weight, 0)
        # nn.init.constant_(self.mlp.c_proj.bias, 0)

    def forward(self, x):
        #HF map
        x = self.process(self.proj(x))
        # x = x.flatten(2).transpose(1, 2)
        # x = self.mlp(x)
        return x