import requests
import supervision as sv
from PIL import Image
import numpy as np
from collections import defaultdict
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput

from typing import List, Literal, Optional, Union, Tuple, Callable, Set, Any
from pydantic import BaseModel
import os
from tqdm import tqdm
import math
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load

from tinygrad import Tensor as tinyTensor, nn as tinynn
import cv2

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

class Dinov2WithRegistersPatchEmbeddings_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.projection_tiny = d.projection_tiny

    def __call__(self, x):
        x = self.projection_tiny(x).flatten(2).transpose(1, 2)
        return x

class WindowedDinov2WithRegistersEmbeddings_tiny():
    def __init__(self, w=None):
        if w is None: return
        self.patch_embeddings = w.patch_embeddings
        self.position_embeddings_tiny = w.position_embeddings_tiny
        self.config = w.config
        self.cls_token_tiny = w.cls_token_tiny
    
    def __call__(self, pixel_values, bool_masked_pos: Optional[Any] = None):
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token_tiny.expand(batch_size, -1, -1)
        embeddings = tinyTensor.cat(cls_tokens, embeddings, dim=1)
        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings_tiny

        # reshape for windows
        num_h_patches = height // 16
        num_w_patches = width // 16
        cls_token_with_pos_embed = embeddings[:, :1]
        pixel_tokens_with_pos_embed = embeddings[:, 1:]
        
        pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)
        num_w_patches_per_window = num_w_patches // 2
        num_h_patches_per_window = num_h_patches // 2
        num_windows = 2
        windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(batch_size * num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1)
        windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
        windowed_pixel_tokens = windowed_pixel_tokens.reshape(batch_size * num_windows ** 2, num_h_patches_per_window * num_w_patches_per_window, -1)
        windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows ** 2, 1, 1)
        embeddings = tinyTensor.cat(windowed_cls_token_with_pos_embed, windowed_pixel_tokens, dim=1)
        return embeddings

class Dinov2WithRegistersSelfOutput_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.dense_tiny = d.dense_tiny

    def __call__(self, x):
        x = self.dense_tiny(x)
        return x

class Dinov2WithRegistersSdpaSelfAttention_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.query_tiny = d.query_tiny
        self.key_tiny = d.key_tiny
        self.value_tiny = d.value_tiny
        self.num_attention_heads = d.num_attention_heads
        self.attention_head_size = d.attention_head_size
        self.all_head_size = d.all_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (6, 64)
        x = x.view(new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def __call__(
        self, hidden_states, head_mask: Optional[Any] = None, output_attentions: bool = False
    ) -> Union[Tuple[Any, Any], Tuple[Any]]:
        mixed_query_layer = self.query_tiny(hidden_states)


        key_layer = self.transpose_for_scores(self.key_tiny(hidden_states))
        value_layer = self.transpose_for_scores(self.value_tiny(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        d_k = query_layer.size(-1)
        attn_scores = tinyTensor.matmul(query_layer, key_layer.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = tinyTensor.softmax(attn_scores, axis=-1)
        context_layer = tinyTensor.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (384,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None

class Dinov2WithRegistersSdpaAttention_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.attention = d.attention
        self.output = d.output

    def __call__(
        self,
        hidden_states: Any,
        head_mask: Optional[Any] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Any, Any], Tuple[Any]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class Dinov2WithRegistersMLP_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.fc1_tiny = d.fc1_tiny
        self.fc2_tiny = d.fc2_tiny

    def __call__(self, hidden_state):
        hidden_state = self.fc1_tiny(hidden_state)
        hidden_state = hidden_state * 0.5 * (1.0 + tinyTensor.erf(hidden_state / math.sqrt(2.0)))
        hidden_state = self.fc2_tiny(hidden_state)
        return hidden_state

class Dinov2WithRegistersLayerScale_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.lambda1_tiny = d.lambda1_tiny

    def __call__(self, hidden_state):
        x = hidden_state * self.lambda1_tiny
        return x

class WindowedDinov2WithRegistersLayer_tiny():
    def __init__(self, w=None):
        if w is None: return
        self.norm1_tiny = w.norm1_tiny
        self.norm2_tiny = w.norm2_tiny
        self.attention = w.attention
        self.layer_scale1 = w.layer_scale1
        self.layer_scale2 = w.layer_scale2
        self.mlp = w.mlp
        self.num_windows = w.num_windows

    def __call__(
        self,
        hidden_states: Any,
        head_mask: Optional[Any] = None,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ):
        shortcut = hidden_states
        self.num_windows = 2
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

        outputs = (layer_output,) + outputs
        return outputs

class WindowedDinov2WithRegistersEncoder_tiny():
    def __init__(self, w=None):
        if w is None: return
        self.layer = w.layer
        self.config = w.config

    def __call__(
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
            run_full_attention = i not in [0, 1, 2, 4, 5, 7, 8, 10, 11]
            layer_head_mask = None
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention)
            hidden_states = layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,)
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

class WindowedDinov2WithRegistersBackbone_tiny():
    def __init__(self, w=None):
        if w is None:
          self.config = {}
          self.stage_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7', 'stage8', 'stage9', 'stage10', 'stage11', 'stage12']
          self.out_features = ['stage3', 'stage6', 'stage9', 'stage12']
          return None
        self.embeddings = w.embeddings
        self.encoder = w.encoder
        self.stage_names = w.stage_names
        self.layernorm_tiny = w.layernorm_tiny
        self.num_register_tokens = w.num_register_tokens
        self.config = w.config
        self.out_features = w.out_features

    def __call__(
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
                hidden_state = self.layernorm_tiny(hidden_state)
                hidden_state = hidden_state[:, 1 :]
                # this was actually a bug in the original implementation that we copied here,
                # cause normally the order is height, width
                batch_size, _, height, width = pixel_values.shape
                patch_size = 16


                num_h_patches = height // patch_size
                num_w_patches = width // patch_size

                # undo windowing
                num_windows_squared = 4
                B, HW, C = hidden_state.shape
                num_h_patches_per_window = num_h_patches // 2
                num_w_patches_per_window = num_w_patches // 2
                hidden_state = hidden_state.reshape(B // num_windows_squared, num_windows_squared * HW, C)
                hidden_state = hidden_state.reshape((B // num_windows_squared) * 2, 2, num_h_patches_per_window, num_w_patches_per_window, C)
                hidden_state = hidden_state.permute(0, 2, 1, 3, 4)

                hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        output = (feature_maps,) + outputs[2:]
        return output

class DinoV2_tiny():
    def __init__(self, d=None):
        if d is None: return
        self.patch_size = d.patch_size
        self.num_windows = d.num_windows
        self.encoder = d.encoder

    def __call__(self, x):
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        x = self.encoder(x)
        return list(x[0])

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    value_l_ = value.view(B * n_heads, head_dim, int(value_spatial_shapes), int(value_spatial_shapes))
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
    return ret

class MultiheadAttention_tiny():
    def __init__(self, m=None):
        if m is None: return
        self.out_proj = m.out_proj
        self.in_proj_weight = m.in_proj_weight
        self.in_proj_bias = m.in_proj_bias

class TransformerDecoderLayer_tiny():
    def __init__(self, t=None):
        if t is None: return
        self.self_attn = t.self_attn
        self.norm1_tiny = t.norm1_tiny
        self.norm2_tiny = t.norm2_tiny
        self.norm3_tiny = t.norm3_tiny
        self.cross_attn = t.cross_attn
        self.linear1_tiny = t.linear1_tiny
        self.linear2_tiny = t.linear2_tiny

    def __call__(self, tgt, memory,
                     tgt_mask: None,
                     memory_mask: None,
                     tgt_key_padding_mask: None,
                     memory_key_padding_mask: None,
                     pos: None,
                     query_pos: None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     ):
        
        q = k = tgt + query_pos
        v = tgt

        C = 256
        B, T, C = q.shape
        H = 8
        D = C // H
        w = self.self_attn.in_proj_weight
        b = self.self_attn.in_proj_bias
        wo = self.self_attn.out_proj_weight
        bo = self.self_attn.out_proj_bias
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
        tgt2 = attn @ wo.T + bo
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
    
def gen_sineembed_for_position(pos_tensor, dim=128):
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
    return pos

class TransformerDecoder_tiny():
    def __init__(self, t=None):
        if t is None: return
        self.d_model = t.d_model
        self.ref_point_head = t.ref_point_head
        self.layers = t.layers
        self.norm_tiny = t.norm_tiny

    def __call__(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                refpoints_unsigmoid=None,
                # for memory
                level_start_index=None, # num_levels
                spatial_shapes=None, # bs, num_levels, 2
                valid_ratios=None):
        output = tgt

        intermediate = []
        def get_reference(refpoints_unsigmoid, valid_ratios):
            obj_center = refpoints_unsigmoid[..., :4]
            refpoints_input = obj_center[:, :, None] * tinyTensor.cat(valid_ratios, valid_ratios, dim=-1)[:, None]
            query_sine_embed = gen_sineembed_for_position(
                refpoints_input[:, :, 0, :], 256 / 2) # bs, nq, 256*2
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

            x = self.norm_tiny(output)
            intermediate.append(x)
        
        output = self.norm_tiny(output)
        intermediate.pop()
        intermediate.append(output)
        return [(tinyTensor.stack(intermediate)), (refpoints_unsigmoid.unsqueeze(0))]

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shape, unsigmoid=True):
    memory_padding_mask = memory_padding_mask.cast(dtype=dtypes.bool)

    proposals = []
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

class MSDeformAttn_tiny():
    """Multi-Scale Deformable Attention Module
    """
    def __init__(self, m=None):
        if m is None: return
        self.value_proj_tiny = m.value_proj_tiny
        self.n_heads = m.n_heads
        self.n_levels = m.n_levels
        self.n_points = m.n_points
        self.sampling_offsets_tiny = m.sampling_offsets_tiny
        self.attention_weights_tiny = m.attention_weights_tiny
        self.d_model = m.d_model
        self.output_proj_tiny = m.output_proj_tiny

    def __call__(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj_tiny(input_flatten)
        value = value.masked_fill(input_padding_mask[..., None], float(0))
        sampling_offsets = self.sampling_offsets_tiny(query).view(N, Len_q, 16, 1, 2, 2)
        attention_weights = self.attention_weights_tiny(query).view(N, Len_q, 16, 1 * 2)
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                + sampling_offsets / 2 * reference_points[:, :, None, :, None, 2:] * 0.5
        attention_weights = attention_weights.softmax(-1)
        value = value.transpose(1, 2).contiguous().view(N, 16, 256 // 16, Len_in)
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj_tiny(output)
        return output

class Transformer_tiny():
    def __init__(self, t=None):
        if not t: return
        self.enc_out_class_embed = t.enc_out_class_embed
        self.bbox_reparam = t.bbox_reparam
        self.enc_output_norm_tiny = t.enc_output_norm_tiny
        self.enc_output_tiny = t.enc_output_tiny
        self.enc_out_bbox_embed = t.enc_out_bbox_embed
        self.num_queries = t.num_queries
        self.d_model = t.d_model
        self.decoder = t.decoder

    def __call__(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):

        self.enc_out_class_embed_w = self.enc_out_class_embed[0].weight
        self.enc_out_class_embed_b = self.enc_out_class_embed[0].bias

        src = srcs[0] if type(srcs) == list else srcs
        pos_embed = pos_embeds[0] if type(pos_embeds) == list else pos_embeds
        bs, _, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)              # bs, hw, c
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
        mask = masks[0].flatten(1) if type(masks) == list else masks.flatten(1)
        level_start_index = tinyTensor([0])
        output_memory, output_proposals = gen_encoder_output_proposals(
            src, mask, h, unsigmoid=True)
        
        output_memory_gidx = self.enc_output_norm_tiny(self.enc_output_tiny(output_memory))
        enc_outputs_class_unselected_gidx = output_memory_gidx @ self.enc_out_class_embed_w.T + self.enc_out_class_embed_b

        
        enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[0](output_memory_gidx)

        enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
            :2] * output_proposals[..., 2:] + output_proposals[..., :2]
        enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
        enc_outputs_coord_unselected_gidx = tinyTensor.cat(enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx, dim=-1)


        topk = min(300, enc_outputs_class_unselected_gidx.shape[-2])
        x = enc_outputs_class_unselected_gidx.max(-1)
        topk_proposals_gidx = tinyTensor.topk(x, topk, dim=1)[1] # bs, nq

        boxes_ts = enc_outputs_coord_unselected_gidx.gather(dim=1, index=topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4))

        # get memory tgt
        memory_ts = output_memory_gidx.gather(dim=1, index=topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 256))

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
                        spatial_shapes=h,
                        valid_ratios=tinyTensor([[[1., 1.]]]))

        return hs, references, memory_ts, boxes_ts

    
class ConvX_tiny():
    def __init__(self, c=None):
        if c is None: return
        self.conv_tiny = c.conv_tiny
        self.bn = c.bn

    def __call__(self, x):
        x = self.conv_tiny(x)
        x = self.bn(x)
        out = tinyTensor.silu(x)
        return out

class Bottleneck_tiny():
    def __init__(self, b=None):
        if b is None: return
        self.cv1 = b.cv1
        self.cv2 = b.cv2
        self.add = b.add

    def __call__(self, x): return self.cv2(self.cv1(x))

class C2f_tiny():
    def __init__(self, c=None):
        if c is None:
          self.c = 128
          return
        self.cv1 = c.cv1
        self.cv2 = c.cv2
        self.c = c.c
        self.m = c.m
        pass

    def __call__(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = tinyTensor.cat(*y, dim=1)
        y = self.cv2(y)
        return y

class LayerNorm_tiny():
    def __init__(self, l=None):
        if l is None:
          self.eps = 1e-6
          return
        self.eps = l.eps
        self.weight_tiny = l.weight_tiny
        self.bias_tiny = l.bias_tiny

    def __call__(self, x):
        x = x.permute(0, 2, 3, 1)
        x -= x.mean(axis=-1, keepdim=True)
        var = (x ** 2).mean(axis=-1, keepdim=True) + self.eps
        var = tinyTensor.sqrt(var)
        x_norm = x / var
        x_norm = x_norm * self.weight_tiny
        x_norm = x_norm + self.bias_tiny
        x = x_norm
        x = x.permute(0, 3, 1, 2)
        return x

class MultiScaleProjector_tiny():
    def __init__(self, m=None):
        if m is None: return
        self.stages = m.stages

    def __call__(self, x):
        feat_fuse = tinyTensor.cat(*x, dim=1)
        stage_output = self.stages[0](feat_fuse)
        return [stage_output]

class PositionEmbeddingSine_tiny():
    def __init__(self, p=None):
        if p is None:
          self.scale = 6.283185307179586
          self.num_pos_feats = 128
          self.temperature = 10000
          return
        self.scale = p.scale
        self.num_pos_feats = p.num_pos_feats
        self.temperature = p.temperature

    def __call__(self, tensors, mask, align_dim_orders = True):
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
        return pos
    
class Backbone_tiny():
    def __init__(self, b=None):
        if b is None: return
        self.encoder = b.encoder
        self.projector = b.projector

    def __call__(self, tensors ,mask):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        m = mask
        mask = ~tinyTensor.interpolate(m.unsqueeze(0), size=feats[0].shape[-2:])[0]
        return feats[0], mask

class MLP_tiny():
    def __init__(self, m=None):
        if not m: return
        self.num_layers = m.num_layers
        self.layers_tiny = m.layers_tiny
        self.layers = m.layers

    def __call__(self, x):            
        for i, layer in enumerate(self.layers): x = tinyTensor.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class LWDETR_tiny():
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self, l=None):
        if l is None:
          self.num_queries = 300
          return
        self.backbone = l.backbone
        self.refpoint_embed = l.refpoint_embed
        self.num_queries = l.num_queries
        self.query_feat = l.query_feat
        self.transformer = l.transformer
        self.bbox_embed = l.bbox_embed
        self.class_embed = l.class_embed

    def __call__(self, samples, targets=None):
        _, _, h, w = samples.shape
        mask = tinyTensor.zeros((1, h, w), dtype=dtypes.bool)
        feature, mask = self.backbone(samples, mask)
        pos = self.position_embedding(feature, mask)[0]
        refpoint_embed_weight = self.refpoint_embed_tiny[:self.num_queries]
        query_feat_weight = self.query_feat_tiny[:self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(feature, mask, [pos], refpoint_embed_weight, query_feat_weight)
        outputs_coord_delta = self.bbox_embed(hs)

        outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
        outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
        outputs_coord = tinyTensor.cat(outputs_coord_cxcy, outputs_coord_wh, dim=-1)

        outputs_class = self.class_embed(hs)[-1]
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord[-1]}
        hs_enc_list = hs_enc.chunk(1, dim=1)
        cls_enc = []
        cls_enc_gidx = self.transformer.enc_out_class_embed[0](hs_enc_list[0])
        cls_enc.append(cls_enc_gidx)
        return out

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

class tiny_seq:
    def __init__(self, size=0):
        self.size = size

    def __setitem__(self, key, value):
        setattr(self, str(key), value)

    def __getitem__(self, idx):
        try:
            return getattr(self, str(idx))
        except AttributeError:
            raise IndexError(idx)

    def __len__(self):
        return self.size

    def __call__(self, x):
        for i in range(self.size):
            layer = getattr(self, str(i))
            x = layer(x)
        return x
        

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
        out_logits.realize() # todo, why do we have to do this?
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
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values.numpy(), labels.numpy(), boxes.numpy())]

class Model:
    def __init__(self, resolution, name):
        self.resolution = resolution
        self.postprocess = PostProcess(num_select=self.resolution)

        config = {"nano":{"n_layers":2, "size": 577}, "small":{"n_layers":3, "size":1025},
                  "medium":{"n_layers":4, "size":1297}, "large":{"n_layers":4, "size":1937}}
        
        new_model = LWDETR_tiny()
        new_model.position_embedding = PositionEmbeddingSine_tiny()
        new_model.query_feat_tiny = tinyTensor.empty((3900, 256))
        new_model.refpoint_embed_tiny = tinyTensor.empty((3900, 4))
        new_model.class_embed = tinynn.Linear(256, 91)
        new_model.bbox_embed = MLP_tiny()
        new_model.bbox_embed.num_layers = 3
        new_model.bbox_embed.layers_tiny = tiny_seq(size=3)
        new_model.bbox_embed.layers_tiny[0] = tinynn.Linear(256, 256)
        new_model.bbox_embed.layers_tiny[1] = tinynn.Linear(256, 256)
        new_model.bbox_embed.layers_tiny[2] = tinynn.Linear(256, 4)

        new_model.bbox_embed.layers = tiny_seq(size=3)
        new_model.bbox_embed.layers[0] = tinynn.Linear(256, 256)
        new_model.bbox_embed.layers[1] = tinynn.Linear(256, 256)
        new_model.bbox_embed.layers[2] = tinynn.Linear(256, 4)


        new_model.transformer = Transformer_tiny()
        new_model.transformer.enc_output_tiny = tinynn.Linear(256, 256)
        new_model.transformer.enc_output_norm_tiny = tinynn.LayerNorm(256)
        new_model.transformer.decoder = TransformerDecoder_tiny()
        new_model.transformer.decoder.norm_tiny = tinynn.LayerNorm(256)
        new_model.transformer.decoder.layers = tiny_seq(config[name]["n_layers"])
        for i in range(config[name]["n_layers"]):
          new_model.transformer.decoder.layers[i] = TransformerDecoderLayer_tiny()
          new_model.transformer.decoder.layers[i].self_attn = MultiheadAttention_tiny()
          new_model.transformer.decoder.layers[i].cross_attn = MSDeformAttn_tiny()
          new_model.transformer.decoder.layers[i].linear1_tiny = tinynn.Linear(256, 2048)
          new_model.transformer.decoder.layers[i].linear2_tiny = tinynn.Linear(2048, 256)

          new_model.transformer.decoder.layers[i].self_attn.in_proj_weight = tinyTensor.empty(768, 256)
          new_model.transformer.decoder.layers[i].self_attn.in_proj_bias = tinyTensor.empty(768)
          new_model.transformer.decoder.layers[i].self_attn.out_proj_weight = tinyTensor.empty(256, 256)
          new_model.transformer.decoder.layers[i].self_attn.out_proj_bias = tinyTensor.empty(256)

          new_model.transformer.decoder.layers[i].norm1_tiny = tinynn.LayerNorm(256)
          new_model.transformer.decoder.layers[i].norm2_tiny = tinynn.LayerNorm(256)
          new_model.transformer.decoder.layers[i].norm3_tiny = tinynn.LayerNorm(256)

          new_model.transformer.decoder.layers[i].cross_attn.value_proj_tiny = tinynn.Linear(256, 256)
          new_model.transformer.decoder.layers[i].cross_attn.output_proj_tiny = tinynn.Linear(256, 256)
          new_model.transformer.decoder.layers[i].cross_attn.sampling_offsets_tiny = tinynn.Linear(256, 64)
          new_model.transformer.decoder.layers[i].cross_attn.attention_weights_tiny = tinynn.Linear(256, 32)

        new_model.transformer.enc_out_bbox_embed = tiny_seq(13)
        new_model.transformer.enc_out_class_embed = tiny_seq(13)
        for i in range(13):
          new_model.transformer.enc_out_bbox_embed[i] = MLP_tiny()
          new_model.transformer.enc_out_bbox_embed[i].num_layers = 3
          new_model.transformer.enc_out_bbox_embed[i].layers_tiny = tiny_seq(3)
          new_model.transformer.enc_out_bbox_embed[i].layers_tiny[0] = tinynn.Linear(256, 256)
          new_model.transformer.enc_out_bbox_embed[i].layers_tiny[1] = tinynn.Linear(256, 256)
          new_model.transformer.enc_out_bbox_embed[i].layers_tiny[2] = tinynn.Linear(256, 4)
          new_model.transformer.enc_out_bbox_embed[i].layers = tiny_seq(3)
          new_model.transformer.enc_out_bbox_embed[i].layers[0] = tinynn.Linear(256, 256)
          new_model.transformer.enc_out_bbox_embed[i].layers[1] = tinynn.Linear(256, 256)
          new_model.transformer.enc_out_bbox_embed[i].layers[2] = tinynn.Linear(256, 4)

          new_model.transformer.enc_out_class_embed[i] = tinynn.Linear(256, 91)         

        new_model.transformer.decoder.ref_point_head = MLP_tiny()
        new_model.transformer.decoder.ref_point_head.layers_tiny = tiny_seq(size=2)
        new_model.transformer.decoder.ref_point_head.num_layers = 2
        new_model.transformer.decoder.ref_point_head.layers = tiny_seq(size=2)
        new_model.transformer.decoder.ref_point_head.layers_tiny[0] = tinynn.Linear(512, 256)
        new_model.transformer.decoder.ref_point_head.layers_tiny[1] = tinynn.Linear(256, 256)
        new_model.transformer.decoder.ref_point_head.layers[0] = tinynn.Linear(512, 256) # todo remove
        new_model.transformer.decoder.ref_point_head.layers[1] = tinynn.Linear(256, 256)

        new_model.backbone = Backbone_tiny()
        new_model.backbone.projector = MultiScaleProjector_tiny()
        new_model.backbone.projector.stages = tiny_seq(1) # todo, this is dumb
        new_model.backbone.projector.stages[0] = tiny_seq(2)
        new_model.backbone.projector.stages[0][0] = C2f_tiny()
        new_model.backbone.projector.stages[0][1] = LayerNorm_tiny()
        new_model.backbone.projector.stages[0][1].weight_tiny = tinyTensor.empty((256))
        new_model.backbone.projector.stages[0][1].bias_tiny = tinyTensor.empty((256))
        new_model.backbone.projector.stages[0][0].cv1 = ConvX_tiny()
        new_model.backbone.projector.stages[0][0].cv2 = ConvX_tiny()          
        new_model.backbone.projector.stages[0][0].cv1.conv_tiny = tinynn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))
        new_model.backbone.projector.stages[0][0].cv2.conv_tiny = tinynn.Conv2d(in_channels=640, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))

        new_model.backbone.projector.stages[0][0].cv1.bn = LayerNorm_tiny()
        new_model.backbone.projector.stages[0][0].cv1.bn.weight_tiny = tinyTensor.empty((256))
        new_model.backbone.projector.stages[0][0].cv1.bn.bias_tiny = tinyTensor.empty((256))
        new_model.backbone.projector.stages[0][0].cv2.bn = LayerNorm_tiny()
        new_model.backbone.projector.stages[0][0].cv2.bn.weight_tiny = tinyTensor.empty((256))
        new_model.backbone.projector.stages[0][0].cv2.bn.bias_tiny = tinyTensor.empty((256))
      
        new_model.backbone.projector.stages[0][0].m = tiny_seq(size=3)
        for i in range(3):
          new_model.backbone.projector.stages[0][0].m[i] = Bottleneck_tiny()
          new_model.backbone.projector.stages[0][0].m[i].cv1 = ConvX_tiny()
          new_model.backbone.projector.stages[0][0].m[i].cv1.conv_tiny = tinynn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))
          new_model.backbone.projector.stages[0][0].m[i].cv2 = ConvX_tiny()
          new_model.backbone.projector.stages[0][0].m[i].cv2.conv_tiny = tinynn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))            
          new_model.backbone.projector.stages[0][0].m[i].cv1.bn = LayerNorm_tiny()
          new_model.backbone.projector.stages[0][0].m[i].cv1.bn.weight_tiny = tinyTensor.empty((128))
          new_model.backbone.projector.stages[0][0].m[i].cv1.bn.bias_tiny = tinyTensor.empty((128))
          new_model.backbone.projector.stages[0][0].m[i].cv2.bn = LayerNorm_tiny()
          new_model.backbone.projector.stages[0][0].m[i].cv2.bn.weight_tiny = tinyTensor.empty((128))
          new_model.backbone.projector.stages[0][0].m[i].cv2.bn.bias_tiny = tinyTensor.empty((128))
        

        new_model.backbone.encoder = DinoV2_tiny()
        new_model.backbone.encoder.patch_size = 16 # todo, not in state_dict?
        new_model.backbone.encoder.num_windows = 2 # todo, not in state_dict?
        new_model.backbone.encoder.encoder = WindowedDinov2WithRegistersBackbone_tiny()
        new_model.backbone.encoder.encoder.layernorm_tiny = tinynn.LayerNorm(384)
        new_model.backbone.encoder.encoder.encoder = WindowedDinov2WithRegistersEncoder_tiny()
        new_model.backbone.encoder.encoder.embeddings = WindowedDinov2WithRegistersEmbeddings_tiny()
        new_model.backbone.encoder.encoder.embeddings.patch_embeddings = Dinov2WithRegistersPatchEmbeddings_tiny()
        new_model.backbone.encoder.encoder.embeddings.patch_embeddings.projection_tiny = tinynn.Conv2d(in_channels=3, out_channels=384, kernel_size=16, stride=(16, 16))
        new_model.backbone.encoder.encoder.embeddings.cls_token_tiny = tinyTensor.empty((1, 1, 384))
        new_model.backbone.encoder.encoder.embeddings.position_embeddings_tiny = tinyTensor.empty((1, config[name]["size"], 384))
        new_model.backbone.encoder.encoder.encoder.layer = tiny_seq(size=13)
        for i in range(12):
          new_model.backbone.encoder.encoder.encoder.layer[i] = WindowedDinov2WithRegistersLayer_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].norm1_tiny = tinynn.LayerNorm(384)
          new_model.backbone.encoder.encoder.encoder.layer[i].norm2_tiny = tinynn.LayerNorm(384)
          new_model.backbone.encoder.encoder.encoder.layer[i].attention = Dinov2WithRegistersSdpaAttention_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.attention = Dinov2WithRegistersSdpaSelfAttention_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.attention.query_tiny = tinynn.Linear(384, 384)
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.attention.key_tiny = tinynn.Linear(384, 384)
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.attention.value_tiny = tinynn.Linear(384, 384)
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.output = Dinov2WithRegistersSelfOutput_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].attention.output.dense_tiny = tinynn.Linear(384, 384)
          new_model.backbone.encoder.encoder.encoder.layer[i].layer_scale1 = Dinov2WithRegistersLayerScale_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].layer_scale2 = Dinov2WithRegistersLayerScale_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].layer_scale1.lambda1_tiny = tinyTensor.empty((384))
          new_model.backbone.encoder.encoder.encoder.layer[i].layer_scale2.lambda1_tiny = tinyTensor.empty((384))
          new_model.backbone.encoder.encoder.encoder.layer[i].mlp = Dinov2WithRegistersMLP_tiny()
          new_model.backbone.encoder.encoder.encoder.layer[i].mlp.fc1_tiny = tinynn.Linear(384, 1536)
          new_model.backbone.encoder.encoder.encoder.layer[i].mlp.fc2_tiny = tinynn.Linear(1536, 384)

        state_dict = safe_load(f"{name}.safetensors")
        load_state_dict(new_model, state_dict)
        self.model = new_model

class RFDETR:
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None
    def __init__(self, resolution, name): self.model = Model(resolution, name)

    def predict(self, img, threshold: float = 0.5):
        img_np = np.asarray(img)
        h, w = img_np.shape[:2]
        img_np = img_np.astype(np.float32) / 255.0
        target = self.model.resolution
        interp = cv2.INTER_AREA if max(h, w) > target else cv2.INTER_LINEAR
        img_np = cv2.resize(img_np, (target, target), interpolation=interp)
        means = np.array(self.means, dtype=np.float32).reshape(1,1,3)
        stds = np.array(self.stds, dtype=np.float32).reshape(1,1,3)
        img_np = (img_np - means) / stds
        img_np = np.transpose(img_np, (2,0,1))
        processed_images = tinyTensor([img_np])
        batch_tensor = tinyTensor.stack(*processed_images)
        predictions = self.model.model(batch_tensor)
        target_sizes = tinyTensor([[h,w]])
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
            
            detections = sv.Detections(xyxy=boxes, confidence=scores,class_id=labels)
            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

excepted_xyxys = [
[[63.662533,247.56085,649.37244,933.79956,],
[1.341641,358.99182,652.4267,1263.1971,],
[622.907,721.52893,698.8878,787.8673,]],

[[68.807434,247.89589,623.2189,930.11993,],
[0.668149,658.804,443.2135,1268.3124,],
[-2.485571,349.83823,646.1853,1261.8707,],
[623.1051,716.02563,701.54877,787.0673,]],

[[68.89001,248.13159,622.48975,927.16235,],
[626.83704,733.20435,696.6672,788.11206,],
[0.5390167,355.58514,649.50793,1266.9197,]],

[[68.368454,249.53983,631.3086,928.4662,],
[625.03973,730.9523,696.16437,786.9667,],
[-0.46054602,661.33997,440.52988,1272.4196,],
[2.424996,357.59814,587.4715,1267.9884,]]
]

models = [[384, "nano"], [512, "small"], [576, "medium"], [704, "large"]]
for i in range(len(models)):
  image = Image.open('dog.jpg')
  model = RFDETR(models[i][0], models[i][1])
  detections = model.predict(image, threshold=0.5)
  labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
  annotated_image = sv.BoxAnnotator().annotate(image, detections)
  annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
  np.testing.assert_allclose(detections.xyxy, excepted_xyxys[i], atol=0.5)
  annotated_image.save(f"annotated_image_{i}.jpg")

print("PASSED")