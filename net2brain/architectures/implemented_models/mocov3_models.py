# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This file contains code adapted and restructured from Meta's MoCo v3 repository:
# https://github.com/facebookresearch/moco-v3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications made by Net2Brain:
# - adapt to Timm version 0.4.12

import math
import warnings
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

from ..netsetbase import CACHE_DIR
from ..shared_functions import download_to_path


###### COPIED FROM MOCOV3 OFFICIAL REPO###
__all__ = [
    'vit_small',
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
    # Net2Brain entry points referenced by configs/mocov3.json
    'vit_small_300ep',
    'vit_base_300ep',
    'resnet50_100ep',
    'resnet50_300ep',
    'resnet50_1000ep',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
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
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
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

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
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
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def load_state_dict(file_url, linear_keyword):
    # Cache the checkpoint alongside the other Net2Brain downloads
    checkpoint_dir = CACHE_DIR / "mocov3_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / file_url.rsplit("/", 1)[-1]

    if not checkpoint_path.exists():
        print(f"~ Downloading weights to {checkpoint_path}")
        download_to_path(file_url, checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        # A cached file that cannot be read is unusable, e.g. left over from an
        # interrupted download. Discard it and fetch the checkpoint once more.
        warnings.warn(f"Cached checkpoint {checkpoint_path} is unreadable, re-downloading.")
        checkpoint_path.unlink(missing_ok=True)
        download_to_path(file_url, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    return state_dict


def load_pretrained_weights(model, file_url, linear_keyword):
    """Load MoCo v3 weights into `model` and verify that they actually arrived.

    Only the (unused) linear classification head is expected to stay randomly
    initialized. Anything else missing means the checkpoint layout changed and the
    features would silently be extracted from an untrained network.
    """
    state_dict = load_state_dict(file_url, linear_keyword)
    msg = model.load_state_dict(state_dict, strict=False)

    expected_missing = {'%s.weight' % linear_keyword, '%s.bias' % linear_keyword}
    unexpected_missing = set(msg.missing_keys) - expected_missing
    if unexpected_missing or msg.unexpected_keys:
        raise RuntimeError(
            f"MoCo v3 checkpoint {file_url} does not match the model definition. "
            f"Missing keys: {sorted(unexpected_missing)}. "
            f"Unexpected keys: {sorted(msg.unexpected_keys)}."
        )
    return model

def vit_base_300ep(pretrained: bool=True, **kwargs):

    model = vit_base()
    file_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    if pretrained:
        load_pretrained_weights(model, file_url, linear_keyword="head")

    model.eval()
    return model

def vit_small_300ep(pretrained: bool=True, **kwargs):

    model = vit_small()
    file_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
    if pretrained:
        load_pretrained_weights(model, file_url, linear_keyword="head")

    model.eval()
    return model

def resnet50_100ep(pretrained: bool=True, **kwargs):

    model = torchvision_models.__dict__['resnet50']()
    file_url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar"
    if pretrained:
        load_pretrained_weights(model, file_url, linear_keyword="fc")

    model.eval()
    return model

def resnet50_300ep(pretrained: bool=True, **kwargs):

    model = torchvision_models.__dict__['resnet50']()
    file_url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar"
    if pretrained:
        load_pretrained_weights(model, file_url, linear_keyword="fc")

    model.eval()
    return model

def resnet50_1000ep(pretrained: bool=True, **kwargs):

    model = torchvision_models.__dict__['resnet50']()
    file_url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
    if pretrained:
        load_pretrained_weights(model, file_url, linear_keyword="fc")

    model.eval()
    return model