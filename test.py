import requests
import supervision as sv
from PIL import Image
import numpy as np
from collections import defaultdict
from transformers.utils.backbone_utils import BackboneMixin, BackboneConfigMixin
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

import torch
from torch import nn
from torch import Tensor
from typing import List, Literal, Optional, Union, Tuple, Callable, Set, Any
from pydantic import BaseModel, field_validator
import os
import torchvision.transforms.functional as vF
import torch.nn.functional as F
from tqdm import tqdm
import math
import copy
import argparse
import json
import collections.abc
from tinygrad.dtype import dtypes
import pickle

from tinygrad import Tensor as tinyTensor, nn as tinynn

def to_tiny(x):
    if type(x) in [tuple, list]:
        ret = []
        for i in range(len(x)): ret.append(to_tiny(x[i]))
        return tuple(ret) if type(x) is tuple else ret
    return tinyTensor(x.detach().numpy()) if type(x) != tinyTensor else x
def to_torch(x):
    if type(x) in [tuple, list]:
        ret = []
        for i in range(len(x)): ret.append(to_torch(x[i]))
        return tuple(ret) if type(x) == tuple else ret
    return Tensor(x.numpy()) if type(x) != Tensor else x

COCO_CLASSES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella",
31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
89: "hair drier", 90: "toothbrush",
}

DEVICE = "cpu"

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

configs = {"small": {'architectures': ['Dinov2Model'], 'attention_probs_dropout_prob': 0.0, 'drop_path_rate': 0.0, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.0, 'hidden_size': 384, 'image_size': 518, 'initializer_range': 0.02, 'layer_norm_eps': 1e-06, 'layerscale_value': 1.0, 'mlp_ratio': 4, 'model_type': 'dinov2', 'num_attention_heads': 6, 'num_channels': 3, 'num_hidden_layers': 12, 'patch_size': 14, 'qkv_bias': True, 'torch_dtype': 'float32', 'transformers_version': '4.32.0.dev0', 'use_swiglu_ffn': False}}

PLATFORM_MODELS = {
    "rf-detr-xlarge.pth": "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xlarge.pth",
    "rf-detr-xxlarge.pth": "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth",
}


OPEN_SOURCE_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    "rf-detr-base-o365.pth": "https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-seg-preview.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
    "rf-detr-large-2026.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
    "rf-detr-xlarge.pth": "https://storage.googleapis.com/rfdetr/rf-detr-xl-ft.pth",
    "rf-detr-xxlarge.pth": "https://storage.googleapis.com/rfdetr/rf-detr-2xl-ft.pth",
    "rf-detr-seg-nano.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
    "rf-detr-seg-small.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
    "rf-detr-seg-medium.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    "rf-detr-seg-large.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
    "rf-detr-seg-xlarge.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
    "rf-detr-seg-xxlarge.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
}


class tiny_seq():
    def __init__(self, size=0):
        self.modules = [None] * size

    def __setitem__(self, idx, value):
        self.modules[idx] = value
        
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __iter__(self):
        return iter(self.modules)

    def __forward__(self, x):
        for i in self.modules: x = i(x)
        return x


class WindowedDinov2WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = "dinov2_with_registers"
    def __init__(): pass

class Dinov2WithRegistersPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
    def forward(self, x):
        x = to_tiny(x)
        x = self.projection_tiny(x).flatten(2).transpose(1, 2)
        return x

class WindowedDinov2WithRegistersEmbeddings(nn.Module):

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

    def forward(self, pixel_values, bool_masked_pos: Optional[Any] = None):
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token_tiny.expand(batch_size, -1, -1)
        embeddings = to_tiny(embeddings)
        embeddings = tinyTensor.cat(cls_tokens, embeddings, dim=1)
        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings_tiny

        # reshape for windows
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        cls_token_with_pos_embed = embeddings[:, :1]
        pixel_tokens_with_pos_embed = embeddings[:, 1:]
        
        pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)
        num_w_patches_per_window = num_w_patches // self.config.num_windows
        num_h_patches_per_window = num_h_patches // self.config.num_windows
        num_windows = self.config.num_windows
        windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(batch_size * num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1)
        windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
        windowed_pixel_tokens = windowed_pixel_tokens.reshape(batch_size * num_windows ** 2, num_h_patches_per_window * num_w_patches_per_window, -1)
        windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows ** 2, 1, 1)
        embeddings = tinyTensor.cat(windowed_cls_token_with_pos_embed, windowed_pixel_tokens, dim=1)
        return embeddings

class Dinov2WithRegistersSelfAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

    def transpose_for_scores(self, x):
        x = to_tiny(x)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

class Dinov2WithRegistersSelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2WithRegistersLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None: pass

    def forward(self, x):
        x = self.dense_tiny(x)
        return to_torch(x)

class Dinov2WithRegistersAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None: pass

    def forward(
        self,
        hidden_states: Any,
        head_mask: Optional[Any] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Any, Any], Tuple[Any]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class Dinov2WithRegistersSdpaSelfAttention(Dinov2WithRegistersSelfAttention):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)

    def forward(
        self, hidden_states, head_mask: Optional[Any] = None, output_attentions: bool = False
    ) -> Union[Tuple[Any, Any], Tuple[Any]]:

        hidden_states = to_tiny(hidden_states)
        mixed_query_layer = self.query_tiny(hidden_states)


        key_layer = self.transpose_for_scores(self.key_tiny(hidden_states))
        value_layer = self.transpose_for_scores(self.value_tiny(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = to_tiny(query_layer)
        key_layer = to_tiny(key_layer)
        value_layer = to_tiny(value_layer)

        d_k = query_layer.size(-1)
        attn_scores = tinyTensor.matmul(query_layer, key_layer.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = tinyTensor.softmax(attn_scores, axis=-1)
        context_layer = tinyTensor.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class Dinov2WithRegistersSdpaAttention(Dinov2WithRegistersAttention):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention = Dinov2WithRegistersSdpaSelfAttention(config)

DINOV2_WITH_REGISTERS_ATTENTION_CLASSES = {
    "eager": Dinov2WithRegistersAttention,
    "sdpa": Dinov2WithRegistersSdpaAttention,
}

HOSTED_MODELS = {**OPEN_SOURCE_MODELS, **PLATFORM_MODELS}

class Dinov2WithRegistersMLP(nn.Module):
    def __init__(self, config) -> None: pass

    def forward(self, hidden_state):
        hidden_state = to_tiny(hidden_state)
        hidden_state = self.fc1_tiny(hidden_state)
        hidden_state = hidden_state * 0.5 * (1.0 + tinyTensor.erf(hidden_state / math.sqrt(2.0)))
        hidden_state = self.fc2_tiny(hidden_state)
        return to_torch(hidden_state)

class Dinov2WithRegistersLayerScale(nn.Module):
    def __init__(self, config) -> None: pass

    def forward(self, hidden_state):
        hidden_state = to_tiny(hidden_state)
        x = hidden_state * self.lambda1_tiny
        return x

class WindowedDinov2WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None: pass
    def forward(
        self,
        hidden_states: Any,
        head_mask: Optional[Any] = None,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ):
        hidden_states = to_tiny(hidden_states)
        shortcut = hidden_states
        if run_full_attention:
            # reshape x to remove windows
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)
        x = self.norm1_tiny(hidden_states)

        # todo
        self_attention_outputs = self.attention(
            x,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)
        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + shortcut

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2_tiny(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = layer_output + hidden_states

        layer_output = to_torch(layer_output)
        outputs = (layer_output,) + outputs
        return outputs

class WindowedDinov2WithRegistersEncoder(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None: pass
    def forward(
        self,
        hidden_states: Any,
        head_mask: Optional[Any] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            run_full_attention = i not in self.config.window_block_indexes
            layer_head_mask = None
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention)
            hidden_states = layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,)
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
class WindowedDinov2WithRegistersBackbone(PreTrainedModel, BackboneMixin):
    _supports_sdpa = True #todo, why need?

    def __init__(self, config: WindowedDinov2WithRegistersConfig): pass

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = to_tiny(hidden_state)
                hidden_state = self.layernorm_tiny(hidden_state)
                hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                # this was actually a bug in the original implementation that we copied here,
                # cause normally the order is height, width
                batch_size, _, height, width = pixel_values.shape
                patch_size = self.config.patch_size

                num_h_patches = height // patch_size
                num_w_patches = width // patch_size

                # undo windowing
                num_windows_squared = self.config.num_windows ** 2
                B, HW, C = hidden_state.shape
                num_h_patches_per_window = num_h_patches // self.config.num_windows
                num_w_patches_per_window = num_w_patches // self.config.num_windows
                hidden_state = hidden_state.reshape(B // num_windows_squared, num_windows_squared * HW, C)
                hidden_state = hidden_state.reshape((B // num_windows_squared) * self.config.num_windows, self.config.num_windows, num_h_patches_per_window, num_w_patches_per_window, C)
                hidden_state = hidden_state.permute(0, 2, 1, 3, 4)

                hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                hidden_state = to_torch(hidden_state)
                feature_maps += (hidden_state,)

        output = (feature_maps,) + outputs[2:]
        return output

class DinoV2(nn.Module):
    def __init__(self): pass

    def forward(self, x):
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        x = self.encoder(x)
        return list(x[0])

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    attention_weights = to_tiny(attention_weights)
    sampling_locations = to_tiny(sampling_locations)
    value = to_tiny(value)
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    value_l_ = value.view(B * n_heads, head_dim, int(value_spatial_shapes[0][0]), int(value_spatial_shapes[0][0]))
    sampling_grid_l_ = sampling_grids[:, :, :, 0].transpose(1, 2).flatten(0, 1)

    N, C, H, W = value_l_.shape
    _, H_out, W_out, _ = sampling_grid_l_.shape
    x = (sampling_grid_l_[..., 0] + 1) * W / 2 - 0.5
    y = (sampling_grid_l_[..., 1] + 1) * H / 2 - 0.5
    x0 = x.floor().cast(dtype=dtypes.int64)
    y0 = y.floor().cast(dtype=dtypes.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0.float()

    wy = y - y0.cast(dtype=dtypes.float)
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy


    v00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)

    v01 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)

    v10 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    v11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)

    value_flat = value_l_.view(N, C, H * W)
    def gather(xi, yi):
        idx = (yi * W + xi).view(N, 1, -1).expand(-1, C, -1)
        return tinyTensor.gather(value_flat, 2, idx).view(N, C, H_out, W_out)

    g00 = gather(x0c, y0c)
    g01 = gather(x0c, y1c)
    g10 = gather(x1c, y0c)
    g11 = gather(x1c, y1c)
    sampling_value_l_ = (g00 * (w00 * v00).unsqueeze(1) + g01 * (w01 * v01).unsqueeze(1) +
    g10 * (w10 * v10).unsqueeze(1) + g11 * (w11 * v11).unsqueeze(1))
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    output = (sampling_value_l_ * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    ret = output.transpose(1, 2).contiguous()
    return to_torch(ret)

class MultiheadAttention_tiny(): # todo
    def __init__(self, m):
        self.in_proj_weight = m.in_proj_weight
        self.in_proj_bias = m.in_proj_bias
        self.out_proj_weight = m.out_proj.weight
        self.out_proj_bias = m.out_proj.bias

class TransformerDecoderLayer_tiny():
    def __init__(self, dec):
        self.self_attn = dec.self_attn
        self.norm1_tiny = dec.norm1_tiny
        self.norm2_tiny = dec.norm2_tiny
        self.norm3_tiny = dec.norm3_tiny
        self.cross_attn = dec.cross_attn
        self.linear1_tiny = dec.linear1_tiny
        self.linear2_tiny = dec.linear2_tiny

    def __call__(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     ):
        
        tgt = to_tiny(tgt)
        query_pos = to_tiny(query_pos)
        q = k = tgt + query_pos
        v = tgt

        C = 256
        B, T, C = q.shape
        H = 8
        D = C // H
        w = to_tiny(self.self_attn.in_proj_weight)
        b = to_tiny(self.self_attn.in_proj_bias)
        bo = to_tiny(self.self_attn.out_proj_bias)
        wq, wk, wv = w.chunk(3, dim=0)
        bq, bk, bv = b.chunk(3, dim=0)

        q = q @ wq.T + bq
        k = k @ wk.T + bk
        v = v @ wv.T + bv

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        attn = tinyTensor.scaled_dot_product_attention(q,k,v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        tgt2 = attn @ to_tiny(self.self_attn.out_proj_weight).T + bo
        tgt = tgt + tgt2
        tgt = self.norm1_tiny(tgt)
        tgt2 = self.cross_attn(
            tgt+query_pos,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask
        )
        tgt = tgt + tgt2
        tgt = self.norm2_tiny(tgt)
        x = self.linear1_tiny(tgt)
        x = x.relu()
        tgt2 = self.linear2_tiny(x)
        tgt += tgt2
        tgt = self.norm3_tiny(tgt)
        return tgt


class TransformerDecoderLayer(nn.Module):
    def __init__(self): pass
    
def gen_sineembed_for_position(pos_tensor, dim=128):
    pos_tensor = to_tiny(pos_tensor)
    scale = 2 * math.pi
    dim_t = tinyTensor.arange(dim)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = tinyTensor.stack(pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos(), dim=3).flatten(2)
    pos_y = tinyTensor.stack(pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos(), dim=3).flatten(2)
    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = tinyTensor.stack(pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos(), dim=3).flatten(2)

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = tinyTensor.stack(pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos(), dim=3).flatten(2)
    pos = tinyTensor.cat(pos_y, pos_x, pos_w, pos_h, dim=2)
    return to_torch(pos)

class TransformerDecoder_tiny():
    def __init__(self, decoder):
        super().__init__()

        # copy modules / attributes
        self.layers = copy.deepcopy(decoder.layers)
        self.norm_tiny = copy.deepcopy(decoder.norm_tiny)
        self.ref_point_head = copy.deepcopy(decoder.ref_point_head)

        # copy simple attributes
        self.d_model = decoder.d_model

    def __call__(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    refpoints_unsigmoid: Optional[Tensor] = None,
                    # for memory
                    level_start_index: Optional[Tensor] = None, # num_levels
                    spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                    valid_ratios: Optional[Tensor] = None):
            output = tgt
            
            intermediate = []
            refpoints_unsigmoid = to_tiny(refpoints_unsigmoid)
            def get_reference(refpoints_unsigmoid, valid_ratios):
                valid_ratios = to_tiny(valid_ratios)
                obj_center = refpoints_unsigmoid[..., :4]
                refpoints_input = obj_center[:, :, None] * tinyTensor.cat(valid_ratios, valid_ratios, dim=-1)[:, None]
                refpoints_input = to_torch(refpoints_input)
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2) # bs, nq, 256*2
                query_pos = self.ref_point_head(query_sine_embed)
                return refpoints_input, query_pos, query_sine_embed

            for layer_id, layer in enumerate(self.layers):
                refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid, valid_ratios) #todo
                pos_transformation = 1

                query_pos = query_pos * pos_transformation
                output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                            is_first=(layer_id == 0),
                            reference_points=refpoints_input,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index)

                output = to_tiny(output)
                x = self.norm_tiny(output)
                intermediate.append(x)
            
            output = self.norm_tiny(output)
            intermediate.pop()
            intermediate.append(output)
            return [tinyTensor.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

class TransformerDecoder(nn.Module):
    def __init__(self): pass

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shape, unsigmoid=True):
    memory = to_tiny(memory)
    memory_padding_mask = to_tiny(memory_padding_mask).cast(dtype=dtypes.bool)
    H_, W_ = spatial_shape, spatial_shape
    mask = memory_padding_mask.reshape(1, H_, W_)

    valid_H = (~mask[:, :, 0]).sum(axis=1).unsqueeze(-1)
    valid_W = (~mask[:, 0, :]).sum(axis=1).unsqueeze(-1)

    x = tinyTensor.linspace(0, H_ - 1, H_)
    y = tinyTensor.linspace(0, W_ - 1, W_)
    grid_y, grid_x = tinyTensor.meshgrid(y, x)

    grid = tinyTensor.cat(grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), dim=-1)
    scale = tinyTensor.cat(valid_W, valid_H, dim=1).view(1, 1, 1, 2)
    grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) / scale

    wh = tinyTensor.ones_like(grid) * 0.05
    output_proposals = tinyTensor.cat(grid, wh, dim=-1).view(1, -1, 4)

    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))
    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals
    #return to_torch(output_memory), to_torch(output_proposals)

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module
    """
    def __init__(self): pass

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        query = to_tiny(query)
        reference_points = to_tiny(reference_points)
        input_flatten = to_tiny(input_flatten)
        input_spatial_shapes = to_tiny(input_spatial_shapes)
        input_padding_mask = to_tiny(input_padding_mask)

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        input_spatial_shapes = to_torch(input_spatial_shapes)

        value = self.value_proj_tiny(input_flatten)
        value = value.masked_fill(input_padding_mask[..., None], float(0))
        sampling_offsets = self.sampling_offsets_tiny(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights_tiny(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        attention_weights = attention_weights.softmax(-1)
        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = to_tiny(output)
        output = self.output_proj_tiny(output)
        return output

class Transformer_tiny():
    def __init__(self, decoder, enc_output, enc_out_bbox_embed, enc_out_class_embed, bbox_reparam, enc_output_norm_tiny, enc_output_tiny,
        num_queries, d_model):
        self.decoder = decoder
        self.enc_output = enc_output
        self.enc_out_bbox_embed = enc_out_bbox_embed
        self.enc_out_class_embed = enc_out_class_embed
        self.bbox_reparam = bbox_reparam
        self.enc_output_norm_tiny = enc_output_norm_tiny
        self.enc_output_tiny = enc_output_tiny
        self.num_queries = num_queries
        self.d_model = d_model

    def __call__(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):

        self.enc_out_class_embed_w = to_tiny(self.enc_out_class_embed[0].weight)
        self.enc_out_class_embed_b = to_tiny(self.enc_out_class_embed[0].bias)

        refpoint_embed = to_tiny(refpoint_embed)

        src = srcs[0] if type(srcs) == list else srcs
        pos_embed = pos_embeds[0] if type(pos_embeds) == list else pos_embeds
        bs, _, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)              # bs, hw, c
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
        mask = masks[0].flatten(1) if type(masks) == list else masks.flatten(1)
        spatial_shapes = Tensor([[h,h]]).long()
        level_start_index = Tensor([0])

        output_memory, output_proposals = gen_encoder_output_proposals(
            src, mask, h, unsigmoid=not self.bbox_reparam)
        
        output_memory_gidx = self.enc_output_norm_tiny(self.enc_output_tiny(output_memory))
        enc_outputs_class_unselected_gidx = output_memory_gidx @ self.enc_out_class_embed_w.T + self.enc_out_class_embed_b

        
        enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed(output_memory_gidx)

        enc_outputs_coord_delta_gidx = to_tiny(enc_outputs_coord_delta_gidx)

        enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
            :2] * output_proposals[..., 2:] + output_proposals[..., :2]
        enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
        enc_outputs_coord_unselected_gidx = tinyTensor.cat(enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx, dim=-1)



        topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
        x = enc_outputs_class_unselected_gidx.max(-1)
        topk_proposals_gidx = tinyTensor.topk(x, topk, dim=1)[1] # bs, nq

        boxes_ts = enc_outputs_coord_unselected_gidx.gather(dim=1, index=topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4))

        # get memory tgt
        memory_ts = output_memory_gidx.gather(dim=1, index=topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))

        # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
        # (bs, nq, d)

        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)

        ts_len = boxes_ts.shape[-2]
        refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
        refpoint_embed_subset = refpoint_embed[..., ts_len:, :]


        refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * boxes_ts[..., 2:]
        refpoint_embed_cxcy = refpoint_embed_cxcy + boxes_ts[..., :2]
        refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * boxes_ts[..., 2:]
        refpoint_embed_ts_subset = tinyTensor.cat(refpoint_embed_cxcy, refpoint_embed_wh, dim=-1)
        refpoint_embed = tinyTensor.cat(refpoint_embed_ts_subset, refpoint_embed_subset, dim=-2)

        hs, references = self.decoder(tgt, src, memory_key_padding_mask=mask,
                        pos=pos_embed, refpoints_unsigmoid=refpoint_embed,
                        level_start_index=level_start_index,
                        spatial_shapes=spatial_shapes,
                        valid_ratios=Tensor([[[1., 1.]]]))

        return hs, references, to_torch(memory_ts), boxes_ts

class Transformer(nn.Module):
    def __init__(self): pass

    def forward(self): exit()

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
    )

class BackboneBase(nn.Module):
    def __init__(self):
        super().__init__()

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu', layer_norm=False, rms_norm=False):
        super(ConvX, self).__init__()
        padding = (kernel // 2, kernel // 2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(kernel, kernel),
                              stride=stride, padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        
        self.conv_tiny = tinynn.Conv2d(in_planes, out_planes, (kernel, kernel), stride, padding, dilation, groups, False)
        self.bn = LayerNorm(out_planes)

    def forward(self, x):
        x = to_tiny(x)
        x = self.conv_tiny(x)
        x = self.bn(x)
        x = to_tiny(x)
        out = tinyTensor.silu(x)
        return to_torch(out)
    

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, number, shortcut, groups, expansion """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)  # optional act=FReLU(c2)
        self.m = Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y = to_tiny(y)
        y.extend(to_tiny(m(y[-1])) for m in self.m)
        y = tinyTensor.cat(*y, dim=1)
        y = self.cv2(y)
        y = to_torch(y)
        return y


class LayerNorm(nn.Module):
    def __init__(self): pass

    def forward(self, x):
        if type(x) != tinyTensor: x = to_tiny(x)
        x = x.permute(0, 2, 3, 1)
        x -= x.mean(axis=-1, keepdim=True)
        var = (x ** 2).mean(axis=-1, keepdim=True) + self.eps
        var = tinyTensor.sqrt(var)
        x_norm = x / var
        x_norm = x_norm * self.weight_tiny
        x_norm = x_norm + self.bias_tiny
        x = x_norm
        x = x.permute(0, 3, 1, 2)
        return to_torch(x)


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(self): pass

    def forward(self, x):
        x = to_tiny(x)
        feat_fuse = tinyTensor.cat(*x, dim=1)
        feat_fuse = to_torch(feat_fuse)
        stage_output = self.stages[0](feat_fuse)
        return [stage_output]

class NestedTensor(object):
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding

class PositionEmbeddingSine_tiny():
    def __init__(self, pos):
        self.num_pos_feats = pos.num_pos_feats
        self.temperature = pos.temperature
        self.scale = pos.scale

    def __call__(self, tensor_list: NestedTensor, align_dim_orders = True):
        mask = tensor_list.mask
        if type(mask) != tinyTensor: mask = to_tiny(mask)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = tinyTensor.arange(self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = tinyTensor.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = tinyTensor.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = tinyTensor.cat(pos_y, pos_x, dim=3).permute(0, 3, 1, 2)
        return to_torch(pos)

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None): pass

    def forward(self, tensor_list: NestedTensor, align_dim_orders = True):
        mask = tensor_list.mask
        if type(mask) != tinyTensor: mask = to_tiny(mask)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = tinyTensor.arange(self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = tinyTensor.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = tinyTensor.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = tinyTensor.cat(pos_y, pos_x, dim=3).permute(0, 3, 1, 2)
        return to_torch(pos)

class Backbone_tiny():
    """backbone."""
    def __init__(self, backbone):
        self.encoder = backbone.encoder
        self.projector = backbone.projector

    def __call__(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        m = tensor_list.mask
        m = to_tiny(m)
        mask = ~tinyTensor.interpolate(m.unsqueeze(0), size=feats[0].shape[-2:])[0]
        mask = to_torch(mask).bool()
        out.append(NestedTensor(feats[0], mask))
        return out

class Backbone(BackboneBase):
    """backbone."""
    def __init__(self): pass

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        m = tensor_list.mask
        m = to_tiny(m)
        mask = ~tinyTensor.interpolate(m.unsqueeze(0), size=feats[0].shape[-2:])[0]
        mask = to_torch(mask).bool()
        out.append(NestedTensor(feats[0], mask))
        return out

def build_backbone(
    encoder,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
    )

    model = Joiner(backbone, position_embedding)
    return model

class Joiner_tiny():
    def __init__(self, joiner):
        super().__init__()
        self.backbone = copy.deepcopy(joiner[0])
        self.position_embedding = copy.deepcopy(joiner[1])

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            return self.backbone
        elif idx == 1:
            return self.position_embedding
        else:
            raise IndexError(f"Joiner_tiny index {idx} out of range")

    def __setitem__(self, idx, value):
        if idx == 0:
            self.backbone = value
        elif idx == 1:
            self.position_embedding = value
        else:
            raise IndexError(f"Joiner_tiny index {idx} out of range")

    def __call__(self, tensor_list):
        x = self.backbone(tensor_list)
        pos = []
        for x_ in x:
            pos.append(
                self.position_embedding(
                    x_, align_dim_orders=False
                ).to(x_.tensors.dtype)
            )
        return x, pos

class Joiner(nn.Sequential):
    def __init__(self): pass
    
def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list) -> NestedTensor:
    tensor_list = to_torch(tensor_list)
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    return NestedTensor(tensor, mask)

class MLP_tiny():
    def __init__(self, mlp):
        super().__init__()
        self.num_layers = mlp.num_layers
        self.layers = copy.deepcopy(mlp.layers)
        self.layers_tiny = copy.deepcopy(mlp.layers_tiny)

    def __call__(self, x):
        for i in range(self.num_layers):
            self.layers_tiny[i].weight = to_tiny(self.layers[i].weight)
            self.layers_tiny[i].bias = to_tiny(self.layers[i].bias)

        x = to_tiny(x)
        for i, layer in enumerate(self.layers_tiny):
            x = tinyTensor.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return to_torch(x)

class MLP(nn.Module):
    def __init__(self): pass
    def __call__(self, x):
        for i in range(self.num_layers):
            self.layers_tiny[i].weight = to_tiny(self.layers[i].weight)
            self.layers_tiny[i].bias = to_tiny(self.layers[i].bias)

        x = to_tiny(x)
        for i, layer in enumerate(self.layers_tiny):
            x = tinyTensor.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return to_torch(x)


class LWDETR_tiny():
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self, model):
        self.backbone = model.backbone
        self.transformer = model.transformer
        self.refpoint_embed = model.refpoint_embed
        self.query_feat = model.query_feat
        self.bbox_embed = model.bbox_embed
        self.class_embed = model.class_embed
        self.num_queries = model.num_queries

    def __call__(self, samples: NestedTensor, targets=None):
        samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        src, mask = features[0].tensors, features[0].mask
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(src, mask, poss, refpoint_embed_weight, query_feat_weight)
        outputs_coord_delta = self.bbox_embed(hs)

        outputs_coord_delta = to_tiny(outputs_coord_delta)
        ref_unsigmoid = to_tiny(ref_unsigmoid)

        outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
        outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
        outputs_coord = tinyTensor.cat(outputs_coord_cxcy, outputs_coord_wh, dim=-1)

        outputs_coord = to_torch(outputs_coord)
        hs = to_torch(hs)
        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        hs_enc_list = hs_enc.chunk(1, dim=1)
        cls_enc = []
        cls_enc_gidx = self.transformer.enc_out_class_embed[0](to_tiny(hs_enc_list[0]))
        cls_enc.append(cls_enc_gidx)
        out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
        return out

class LWDETR(nn.Module):
    def __init__(self): pass

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = [t.squeeze(-1) for t in x.split(1, dim=-1)]

    w_pos = w.clip(0.0, float("inf"))
    h_pos = h.clip(0.0, float("inf"))

    b = [
        x_c - 0.5 * w_pos,
        y_c - 0.5 * h_pos,
        x_c + 0.5 * w_pos,
        y_c + 0.5 * h_pos,
    ]
    return tinyTensor.stack(b, dim=-1)

class PostProcess():
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    def __call__(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_logits = to_tiny(out_logits)
        out_bbox = to_tiny(out_bbox)
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = tinyTensor.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = tinyTensor.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_fct = tinyTensor.stack(img_w, img_h, img_w, img_h, dim=1)
        boxes = boxes * scale_fct[:, None, :]
        topk_values = to_torch(topk_values)
        labels = to_torch(labels).int()
        boxes = to_torch(boxes)
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values, labels, boxes)]

def to_tiny_seq(x):
    ret = tiny_seq(len(x))
    for i in range(len(x)):
        ret[i] = x[i]
    return ret

class Model:
    def __init__(self, **kwargs):
        args = argparse.Namespace(
            num_select=kwargs.get('num_select', 100),
            **{k: v for k, v in kwargs.items() if k != 'num_select'}
        )
        self.args = args
        self.resolution = args.resolution
        with open(f'tiny_{args.pretrain_weights}2.pkl', 'rb') as f: self.model_tiny = pickle.load(f)

        
        SKIP_KEYS = {
            "_parameters", "_buffers", "_modules",
            "_backward_hooks", "_forward_hooks",
            "_forward_pre_hooks", "_state_dict_hooks",
            "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
            "_non_persistent_buffers_set",
        }


        def print_obj(obj, path="self.model", seen=None):
            if seen is None:
                seen = set()
            
            oid = id(obj)
            if oid in seen:
                print(f"{path}: <recursion>")
                return
            seen.add(oid)
            
            # Basic types
            if isinstance(obj, (int, float, str, bool, type(None))):
                print(f"{path}: {obj}")
                return
            
            if isinstance(obj, torch.Tensor):
                print(f"{path}: Tensor{tuple(obj.shape)}")
                return
            
            # Collections
            if isinstance(obj, (list, tuple)):
                print(f"{path}: {obj.__class__.__name__}[{len(obj)}]")
                for i, v in enumerate(obj):
                    print_obj(v, f"{path}[{i}]", seen)
                return
            
            if isinstance(obj, dict):
                print(f"{path}: dict[{len(obj)}]")
                for k, v in obj.items():
                    print_obj(v, f"{path}.{k}", seen)
                return
            
            # Objects - just print class name and recurse
            print(f"{path}: {type(obj)}")
            
            if not hasattr(obj, "__dict__"):
                return
            
            for k, v in obj.__dict__.items():
                print_obj(v, f"{path}.{k}", seen)
                
                # TransformerDecoder - 
                # self.layers: ModuleList
                # self.norm_tiny: tinynn.layernorm
                # self.ref_point_head: MLP
        print_obj(self.model_tiny)
        #with open(f'tiny_{args.pretrain_weights}2.pkl', 'wb') as f: pickle.dump(self.model_tiny, f)

        self.postprocess = PostProcess(num_select=args.num_select)
        self.stop_early = False

class ModelConfig(BaseModel):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    license: str = "Apache-2.0"


class RFDETRBaseConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560

class RFDETRNanoConfig(RFDETRBaseConfig):
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

class ModelConfig(BaseModel):
    pretrain_weights: Optional[str] = None
    resolution: int

    @field_validator("pretrain_weights", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))

class RFDETR:
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.model = self.get_model(self.model_config)

    def get_model(self, config: ModelConfig): return Model(**config.dict())

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, Any, List[Union[str, np.ndarray, Image.Image, Any]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img)

            img = vF.to_tensor(img)

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is "
                    "normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(
                    f"Invalid image shape. Expected 3 channels (RGB), but got "
                    f"{img.shape[0]} channels."
                )
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = vF.normalize(img_tensor, self.means, self.stds)
            img_tensor = vF.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        processed_images = to_tiny(processed_images)
        batch_tensor = tinyTensor.stack(*processed_images)
        predictions = self.model.model_tiny(batch_tensor)
        target_sizes = tinyTensor(orig_sizes)
        results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            detections = sv.Detections(
                xyxy=boxes.float().cpu().numpy(),
                confidence=scores.float().cpu().numpy(),
                class_id=labels.cpu().numpy(),
            )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

class RFDETRNano(RFDETR):
    def get_model_config(self, **kwargs): return RFDETRNanoConfig(**kwargs)

class RFDETRSmallConfig(RFDETRBaseConfig):
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"

class RFDETRSmall(RFDETR):
    def get_model_config(self, **kwargs): return RFDETRSmallConfig(**kwargs)
    
class RFDETRMediumConfig(RFDETRBaseConfig):
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"

class RFDETRMedium(RFDETR):
    def get_model_config(self, **kwargs): return RFDETRMediumConfig(**kwargs)

class RFDETRLarge(RFDETR):
    def get_model_config(self, **kwargs): return RFDETRLargeConfig(**kwargs)

#res 704, ps 16, 2 windows, 4 dec layers, 300 queries, ViT-S basis
class RFDETRLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_small"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_windows: int = 2
    patch_size: int = 16
    projector_scale: List[Literal["P4",]] = ["P4"]
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_classes: int = 90
    positional_encoding_size: int = 704 // 16
    pretrain_weights: Optional[str] = "rf-detr-large-2026.pth"
    resolution: int = 704

excepted_xyxys = [[[61.86511,247.66309,652.2484,930.8369,],
[1.3346028,361.53326,648.76166,1264.4553, ],
[622.7612,720.39746,698.42926,787.9133, ]],

[[68.586105,247.72815,620.7118,930.0703, ],
[0.51882505,661.1699,443.48486,1268.1208, ],
[0.45747757,354.613,641.79736,1264.0822, ],
[623.15704,715.6762,701.5984,787.10944, ],],

[[68.82359,247.85782,621.86975,926.5808, ],
[626.3732,731.4297,696.52435,787.97186, ],
[0.800972,354.816,647.8018,1265.4277, ]],

[[68.12298,249.11542,634.0267,927.67834, ],
[-0.49580097,660.9515,439.59613,1272.4813, ],
[625.1135,730.70593,695.80963,787.00354, ],
[2.5050545,357.18567,593.51825,1266.9233, ]],
]

models = [RFDETRNano(), RFDETRSmall(), RFDETRMedium(), RFDETRLarge()]
for i, model in enumerate(models):
  #image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
  image = Image.open('dog.jpg')
  detections = model.predict(image, threshold=0.5)
  labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
  annotated_image = sv.BoxAnnotator().annotate(image, detections)
  annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
  np.testing.assert_allclose(detections.xyxy, excepted_xyxys[i], atol=0.5)
  annotated_image.save("annotated_image.jpg")

print("PASSED")
