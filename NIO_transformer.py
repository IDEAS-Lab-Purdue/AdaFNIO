import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
#from main_afnonet import get_args
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias
class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x
class AFNONet(nn.Module):
    def __init__(
            self,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=3,
            out_chans=3,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        #self.params = params
        self.img_size = img_size
        self.patch_size = (patch_size[0], patch_size[1])
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class NIO_transformer(nn.Module):
    def __init__(self, img_size=(100,100), patch_size=(2,2), in_chans=3, stack_factor=2, num_classes=0,embed_dim=768, depth=8,
                 mlp_ratio=4., width=100, height=100, mixing_type='sa'):
        super().__init__()
        #self.interpolation_l = nn.Conv2d(3, 3, kernel_size=1)
        #self.interpolation_r = nn.Conv2d(3,3,kernel_size=1)
        self.embed_dim = embed_dim
        #self.AF_transformer_red = AFNONet(img_size=img_size, patch_size=patch_size, in_chans=2, out_chans=1, depth=1)
        #self.AF_transformer_green = AFNONet(img_size=img_size, patch_size=patch_size, in_chans=2, out_chans=1, depth=1)
        #self.AF_transformer_blue = AFNONet(img_size=img_size, patch_size=patch_size, in_chans=2, out_chans=1, depth=1)
        self.AF_transformer = AFNONet(img_size=img_size, patch_size=patch_size, in_chans=6, out_chans=3, depth=depth)
        self.height = height
        self.width = width
        #self.UNO_block_red = UNO_vanilla(2, 1, pad=0, factor=16/4)
        #self.UNO_block_green = UNO_vanilla(2, 1, pad=0, factor=16/4)
        #self.UNO_block_blue = UNO_vanilla(2, 1, pad=0, factor=16/4)
        #self.UNO_block_merge = UNO(3, 3, pad=0, factor=16/4)
        
        
        """
        self.output_dims = in_chans*width*height
        #self.unflatten = nn.Unflatten(1, (3,16,16))
        self.deconv1 = nn.ConvTranspose2d(3, 3, 3,stride=1, padding=1)
        #self.alpha_drop1 = nn.AlphaDropout(p=0.42)
        #self.batch_norm1 = nn.BatchNorm2d(3)
        self.deconv2 = nn.ConvTranspose2d(3, 3, 4,stride=2, padding=1)
        #self.alpha_drop2 = nn.AlphaDropout(p=0.42)
        #self.batch_norm2 = nn.BatchNorm2d(3)
        self.deconv3 = nn.ConvTranspose2d(3, 3, 2,stride=2, padding=7)
        #self.alpha_drop3 = nn.AlphaDropout(p=0.42)
        #self.batch_norm3 = nn.BatchNorm2d(3)
        self.deconv4 = nn.ConvTranspose2d(3, 3, 2,stride=2, padding=0)
        #self.alpha_drop4 = nn.AlphaDropout(p=0.42)
        #self.batch_norm4 = nn.BatchNorm2d(3)
        self.pointwise1 = pointwise_op(3, 3, self.height, self.width)
        self.pointwise2 = pointwise_op(3, 3, self.height, self.width)
        self.pointwise3 = pointwise_op(3,3, self.height, self.width)
        self.pointwise4 = pointwise_op(3, 3, self.height, self.width)
        """
        
        
    def forward(self, x):
        self.height = x.shape[2]
        self.width = x.shape[3]
        #self.output_dims = self.height*self.width*x.shape[4]
        x_l = x[0]
        x_r = x[1]
        
        x_l_colors = x_l.permute(1,0,2,3)
        x_r_colors = x_r.permute(1,0,2,3)
        
        #print(x_l_colors.shape)
        #x_l_red = x_l_colors[0]
        #x_l_green = x_l_colors[1]
        #x_l_blue = x_l_colors[2]
        
        #x_r_red = x_r_colors[0]
        #x_r_green = x_r_colors[1]
        #x_r_blue = x_r_colors[2]
        
        #x_red = torch.stack((x_l_red, x_r_red)).permute(1,0,2,3)
        #x_green = torch.stack((x_l_green, x_r_green)).permute(1,0,2,3)
        #x_blue = torch.stack((x_l_blue, x_r_blue)).permute(1,0,2,3)
        
        #print(x_red.shape)
        #print(x_green.shape)
        #print(x_blue.shape)

        #x = torch.cat((x_l, x_r), dim=1)
        #x = self.AF_transformer(x)
        #r = self.UNO_block_red(x_red)
        #g = self.UNO_block_green(x_green)
        #b = self.UNO_block_blue(x_blue)
        #uno_x = torch.cat((r,g,b), dim=1)
        
        
        x = torch.cat((x_l, x_r), dim=1)
        transfo_x = self.AF_transformer(x)
        #x_concat = 0.0001*uno_x + transfo_x
        #x_out = self.UNO_block_merge(x_concat)
        #x_concat = transfo_x
        
        """
        x = self.deconv1(x)
        x = self.pointwise1(x, self.height//8, self.width//8)
        #x = torch.nn.functional.interpolate(x, size = (self.output_dims//8, self.output_dims//8),mode = 'bicubic',align_corners=True)
        #x = self.alpha_drop1(x)
        #x = self.batch_norm1(x)
        x = self.deconv2(x)
        x = self.pointwise2(x, self.height//4, self.width//4)
        #x = torch.nn.functional.interpolate(x, size=(self.output_dims//4, self.output_dims//4), mode = 'bicubic',align_corners=True)
        #x = self.alpha_drop2(x)
        #x = self.batch_norm2(x)
        x = self.deconv3(x)
        x = self.pointwise3(x, self.height//2, self.width//2)
        #x = torch.nn.functional.interpolate(x, size=(self.output_dims//2, self.output_dims//2), mode = 'bicubic',align_corners=True)
        #x = self.alpha_drop3(x)
        #x = self.batch_norm3(x)
        x = self.deconv4(x)
        x = self.pointwise4(x, self.height, self.width)
        #x = torch.nn.functional.interpolate(x, size=(self.output_dims, self.output_dims), mode = 'bicubic',align_corners=True)
        #x = self.alpha_drop4(x)
        #x = self.batch_norm4(x)
        #x = self.pointwise(x, self.height, self.width)
        """
        #print(x.shape)
        return transfo_x
    def sub_mean(self, x):
        mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x -= mean
        return x, mean
