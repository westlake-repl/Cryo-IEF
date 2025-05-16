# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

# from transformers import ViTConfig, ViTModel,ViTForImageClassification
from timm.models.vision_transformer import VisionTransformer, _cfg,Block
from timm.models.swin_transformer import swin_s3_tiny_224, _cfg
from timm.layers.helpers import to_2tuple
from timm.layers import PatchEmbed,format

# from blitz.modules import BayesianLinear
# from blitz.utils import variational_estimator
import torch.nn.functional as F

__all__ = [
    'vit_small',
    'swin_vit_tiny',
    'vit_base',
    'vit_huge',
    'vit_giant',
    'vit_conv_small',
    'vit_conv_base',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=True,use_bn=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()



        # weight initialization
        for name, m in self.named_modules():
            if use_bn and isinstance(m, Block):
                # module.norm1 = BN_bnc(module.norm1.normalized_shape)
                m.norm2 = BN_bnc(m.norm2.normalized_shape)

            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h,indexing='ij')
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            x = my_checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x: torch.Tensor) :
        feature = self.forward_features(x)
        y = self.forward_head(feature)
        return feature[:, 0],y

# @variational_estimator
# class BayesianClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         # self.linear = nn.Linear(input_dim, output_dim)
#         self.blinear1 = BayesianLinear(input_dim, 512)
#         self.blinear2 = BayesianLinear(512, output_dim)
#
#     def forward(self, x):
#         x_ = self.blinear1(x)
#         x_ = F.relu(x_)
#         x_=self.blinear2(x_)
#         return x_
class Classifier_new(nn.Module):
    def __init__(self, input_dim, output_dim,num_linear_layers=1,mlp_dim=1024,add_vit_blocks_num=0):
        super().__init__()
        self.add_vit_blocks_num=add_vit_blocks_num
        if add_vit_blocks_num>0:

            self.vit_blocks=nn.Sequential(*[
            Block(
                dim=input_dim,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_norm=False,
                init_values=None,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                # act_layer=nn.GELU,
                # mlp_layer=Mlp,
            )
            for i in range(add_vit_blocks_num)])

        mlp = []
        for l in range(num_linear_layers):
            if l==0:
                dim1=input_dim
            else:
                dim1 =  mlp_dim
                mlp_dim=mlp_dim//2
            dim2 = output_dim if l == num_linear_layers - 1 else mlp_dim

            # mlp.append(nn.Linear(dim1, dim2, bias=False))
            mlp.append(nn.Linear(dim1, dim2))

            if l < num_linear_layers - 1:
                mlp.append(nn.SyncBatchNorm(dim2))
                mlp.append(nn.ReLU(inplace=True))
        self.mlp_classifier= nn.Sequential(*mlp)
        self.norm=nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        if self.add_vit_blocks_num >0:
            x_=self.vit_blocks(x)
        else:
            x_=x
        x_=self.norm(x_)
        # x_ = x_[:, 0]
        x_ = self.mlp_classifier(x_)
        # x_=self.bn1(x_)
        # x_ = F.relu(x_)
        # x_=self.linear2(x_)
        # x_=self.bn2(x_)
        # x_ = F.relu(x_)
        # x_=self.linear3(x_)
        return x_

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)

        # self.linear1 = nn.Linear(input_dim, 1024)
        # self.bn1=nn.SyncBatchNorm(1024)
        # self.linear2 = nn.Linear(1024, 256)
        # self.bn2=nn.SyncBatchNorm(256)
        # self.linear3 = nn.Linear(256, output_dim)

        # self.linear1 = nn.Linear(input_dim, 512)
        # # self.bn1=nn.SyncBatchNorm(1024)
        # self.linear2 = nn.Linear(512, 2)

        self.linear1= nn.Linear(input_dim, 2)

    def forward(self, x):
        x_ = self.linear1(x)
        # x_=self.bn1(x_)
        # x_ = F.relu(x_)
        # x_=self.linear2(x_)
        # x_=self.bn2(x_)
        # x_ = F.relu(x_)
        # x_=self.linear3(x_)
        return x_

class Classifier_2linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)

        # self.linear1 = nn.Linear(input_dim, 1024)
        # self.bn1=nn.SyncBatchNorm(1024)
        # self.linear2 = nn.Linear(1024, 256)
        # self.bn2=nn.SyncBatchNorm(256)
        # self.linear3 = nn.Linear(256, output_dim)

        self.linear1 = nn.Linear(input_dim, 512)
        # self.bn1=nn.SyncBatchNorm(1024)
        self.linear2 = nn.Linear(512, 2)

        # self.linear1= nn.Linear(input_dim, 2)

    def forward(self, x):
        x_ = self.linear1(x)
        # x_=self.bn1(x_)
        x_ = F.relu(x_)
        x_=self.linear2(x_)
        # x_=self.bn2(x_)
        # x_ = F.relu(x_)
        # x_=self.linear3(x_)
        return x_



class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=False,**kwargs):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        if in_chans==3:
            embed_dim=embed_dim//2
            self.proj2 = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))
            nn.init.uniform_(self.proj2.weight, -val, val)
            nn.init.zeros_(self.proj2.bias)


        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 1, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if C==3:
            x_f = x[:, 1:, :, :]
            x=x[:,0:1,:,:]

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x=format.nchw_to(x, 'NHWC')
        x = self.norm(x)
        if C==3:
            x_f = self.proj2(x_f)
            if self.flatten:
                x_f = x_f.flatten(2).transpose(1, 2)  # BCHW -> BNC
            else:
                x_f = format.nchw_to(x_f, 'NHWC')
            x_f = self.norm(x_f)
            x=torch.cat([x,x_f],dim=-1)


        return x


class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm,flatten=False, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                          ])


    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x=format.nchw_to(x, 'NHWC')
        return x
def vit_small(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    model = VisionTransformerMoCo(
        in_chans=in_chans,
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,stop_grad_conv1=stop_grad_conv1,**kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_base(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,stop_grad_conv1=stop_grad_conv1,**kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model



def vit_large(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),use_bn=use_bn,stop_grad_conv1=stop_grad_conv1, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model

def vit_huge(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),use_bn=use_bn,stop_grad_conv1=stop_grad_conv1, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model

def vit_giant(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),use_bn=use_bn,stop_grad_conv1=stop_grad_conv1, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_conv_small(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans,patch_size=patch_size,use_bn=use_bn,stop_grad_conv1=stop_grad_conv1,
         embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model

def vit_conv_base(patch_size=14,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans,patch_size=patch_size,use_bn=use_bn,stop_grad_conv1=stop_grad_conv1,
         embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model
def vit_hMLP_base(patch_size=16,in_chans=1,gradient_checkpointing=False,use_bn=True,stop_grad_conv1=False,**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans,patch_size=patch_size,use_bn=use_bn,stop_grad_conv1=stop_grad_conv1,
         embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=hMLP_stem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model



def swin_vit_tiny(**kwargs):
    # minus one ViT block
    model = swin_s3_tiny_224(**kwargs)
    model.default_cfg = _cfg()
    return model



class BN_bnc(nn.SyncBatchNorm):
    """
    BN_bnc: BatchNorm1d on hidden feature with (B,N,C) dimension
    """

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N, C)  # (B,N,C) -> (B*N,C)
        x = super().forward(x)   # apply batch normalization
        x = x.reshape(B, N, C)   # (B*N,C) -> (B,N,C)
        return x


# class BN_MLP(timm.layers.Mlp):
#     """
#     BN_MLP: add BN_bnc in-between 2 linear layers in MLP module
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.norm = BN_bnc(kwargs['hidden_features'])
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.norm(x)  # apply batch normalization before activation
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x



from itertools import chain
from torch.utils.checkpoint import checkpoint
def my_checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential Cryo_IEF.

    Sequential Cryo_IEF execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state,use_reentrant=False)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


