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
from typing import List, Literal, Optional, Union, Tuple, Callable, Set
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

from tinygrad import Tensor as tinyTensor, nn as tinynn

def to_tiny(x): return tinyTensor(x.detach().numpy())
def to_torch(x): return Tensor(x.numpy())

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

class WindowedDinov2WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = "dinov2_with_registers"
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        num_register_tokens=4,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        num_windows=1,
        window_block_indexes=None,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.num_register_tokens = num_register_tokens
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        self._out_features = ['stage3', 'stage6', 'stage9', 'stage12']
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.num_windows = num_windows
        self.window_block_indexes = list(range(num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        self.gradient_checkpointing = gradient_checkpointing

class WindowedDinov2WithRegistersPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = WindowedDinov2WithRegistersConfig
    base_model_prefix = "dinov2_with_registers"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2WithRegistersSwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, WindowedDinov2WithRegistersEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


class Dinov2WithRegistersPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)

class WindowedDinov2WithRegistersEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, register tokens, position and patch embeddings.
    """

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size)) if config.num_register_tokens > 0 else None
        self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This implementation supports torch.jit tracing while maintaining backwards compatibility
        with the original implementation.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # Skip interpolation for matching dimensions (unless tracing)
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        # Handle class token and patch embeddings separately
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        # Calculate new dimensions
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        # Reshape for interpolation
        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Store original dtype for restoration after interpolation
        target_dtype = patch_pos_embed.dtype

        # Interpolate at float32 precision
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            size=(torch_int(height), torch_int(width)),  # Explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        # Validate output dimensions if not tracing
        if not torch.jit.is_tracing():
            if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
                raise ValueError("Width or height does not match with the interpolated position embeddings")

        # Reshape back to original format
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
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
            embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

        # add register tokens
        embeddings = torch.cat(
            (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
        ) if self.config.num_register_tokens > 0 else embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class Dinov2WithRegistersSelfAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

class Dinov2WithRegistersSelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2WithRegistersLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class Dinov2WithRegistersAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.attention = Dinov2WithRegistersSelfAttention(config)
        self.output = Dinov2WithRegistersSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class Dinov2WithRegistersSdpaSelfAttention(Dinov2WithRegistersSelfAttention):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

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
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = F.gelu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

class Dinov2WithRegistersLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1

class WindowedDinov2WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

        self.num_windows = config.num_windows

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOV2_WITH_REGISTERS_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = Dinov2WithRegistersMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        assert head_mask is None, "head_mask is not supported for windowed attention"
        assert not output_attentions, "output_attentions is not supported for windowed attention"
        shortcut = hidden_states
        if run_full_attention:
            # reshape x to remove windows
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            # reshape x to add windows back
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + shortcut

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs

class WindowedDinov2WithRegistersEncoder(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([WindowedDinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > int(self.config.out_features[-1][5:]):
                # early stop if we have reached the last output feature
                break

            run_full_attention = i not in self.config.window_block_indexes

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    run_full_attention,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class WindowedDinov2WithRegistersBackbone(WindowedDinov2WithRegistersPreTrainedModel, BackboneMixin):
    def __init__(self, config: WindowedDinov2WithRegistersConfig):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = WindowedDinov2WithRegistersEmbeddings(config)
        self.encoder = WindowedDinov2WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.num_register_tokens = config.num_register_tokens

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:
        Returns:

        Examples:


        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-with-registers-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size

                    if self.config.num_windows > 1:
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

                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

class DinoV2(nn.Module):
    def __init__(self,
            shape=(640, 640),
            out_feature_indexes=[2, 4, 5, 9],
            size="base",
            use_registers=True,
            use_windowed_attn=True,
            gradient_checkpointing=False,
            load_dinov2_weights=True,
            patch_size=14,
            num_windows=4,
            positional_encoding_size=37,
            ):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        window_block_indexes = set(range(out_feature_indexes[-1] + 1))
        window_block_indexes.difference_update(out_feature_indexes)
        window_block_indexes = list(window_block_indexes)

        dino_config = configs[size]

        dino_config["return_dict"] = False
        dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]

        implied_resolution = positional_encoding_size * patch_size

        if implied_resolution != dino_config["image_size"]:
            print("Using a different number of positional encodings than DINOv2, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.")
            dino_config["image_size"] = implied_resolution
            load_dinov2_weights = False

        if patch_size != 14:
            print(f"Using patch size {patch_size} instead of 14, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.")
            dino_config["patch_size"] = patch_size
            load_dinov2_weights = False

        if use_registers:
            windowed_dino_config = WindowedDinov2WithRegistersConfig(
                **dino_config,
                num_windows=num_windows,
                window_block_indexes=window_block_indexes,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            windowed_dino_config = WindowedDinov2WithRegistersConfig(
                **dino_config,
                num_windows=num_windows,
                window_block_indexes=window_block_indexes,
                num_register_tokens=0,
                gradient_checkpointing=gradient_checkpointing,
            )
        self.encoder = WindowedDinov2WithRegistersBackbone.from_pretrained(
            name,
            config=windowed_dino_config,
        ) if load_dinov2_weights else WindowedDinov2WithRegistersBackbone(windowed_dino_config)

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def forward(self, x):
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        x = self.encoder(x)
        return list(x[0])

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """"for debug and test only, need to use cuda version instead
    """
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # B, n_heads, head_dim, H, W
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1,
                 num_feature_levels=4, dec_n_points=4,
                 skip_self_attn=False):
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
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
        bs, num_queries, _ = tgt.shape

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        q = k = tgt + query_pos
        v = tgt
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask,
                            need_weights=False)[0]

        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask
        )
        # ========== End of Cross-Attention =============


        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = (tgt + self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
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
                level_start_index=None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 query_sine_embed, is_first,
                                 reference_points, spatial_shapes, level_start_index)

def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False,
                 d_model=256,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        self._export = False

    def forward(self, tgt, memory,
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
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4]

            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2) # bs, nq, 256*2
                refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
            else:
                refpoints_input = obj_center[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2) # bs, nq, 256*2
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

            # For the first decoder layer, we do not apply transformation over p_s
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

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                return hs, ref
            # box iterative update
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),
                    torch.stack(hs_refpoints_unsigmoid),
                ]
            else:
                return [
                    torch.stack(intermediate),
                    refpoints_unsigmoid.unsqueeze(0)
                ]

        return output.unsqueeze(0)

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._export = False

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


class Transformer(nn.Module):
    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1,
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, sa_nhead, ca_nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                group_detr=group_detr,
                                                num_feature_levels=num_feature_levels,
                                                dec_n_points=dec_n_points,
                                                skip_self_attn=False,)
        assert decoder_norm_type in ['LN', 'Identity']
        norm = {
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          bbox_reparam=bbox_reparam)


        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = [] if masks is not None else None
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)                    # bs, hw
                mask_flatten.append(mask)
        memory = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam)
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))

                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
                        :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1)
                else:
                    enc_outputs_coord_unselected_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx) + output_proposals # (bs, \sum{hw}, 4) unsigmoid

                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1] # bs, nq

                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))

                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts.append(tgt_undetach_gidx)
                boxes_ts.append(refpoint_embed_gidx_undetach)
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts, dim=1)#.transpose(0, 1)
            boxes_ts = torch.cat(boxes_ts, dim=1)#.transpose(0, 1)

        if self.dec_layers > 0:
            tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat(
                        [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                    )
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

                refpoint_embed = torch.concat(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten,
                            pos=lvl_pos_embed_flatten, refpoints_unsigmoid=refpoint_embed,
                            level_start_index=level_start_index,
                            spatial_shapes=spatial_shapes,
                            valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios)
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None

        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None

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
        self.conv_tiny.weight = to_tiny(self.conv.weight)
        self.bn = LayerNorm(out_planes)

    def forward(self, x):
        x = to_tiny(x)
        x = self.conv_tiny(x)
        x = self.bn(x)
        if type(x) != tinyTensor: x = to_tiny(x)
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
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.weight_tiny = to_tiny(self.weight)
        self.bias_tiny = to_tiny(self.bias)

        self.eps = eps
        self.normalized_shape = (normalized_shape,)

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

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
    ):
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling = []
        stages = []
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                stages_sampling[-1].append(nn.Sequential())
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])
            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm('LN', out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)
        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        results = []
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )
        return results

class NestedTensor(object):
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self._export = False

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

class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 patch_size: int = 14,
                 num_windows: int = 4,
                 positional_encoding_size: bool = False,
                 ):
        super().__init__()
        # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
        # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
        # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
        # the last part of the name should be the size
        # and the start should be dinov2
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        # name_parts[-1]
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
        )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

def build_backbone(
    encoder,
    vit_encoder_num_layers,
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
    force_no_pretrain,
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

def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    print_freq=10,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,

    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,

    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,

    # Backbone parameters
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,

    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale='P4',
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=4,
    decoder_norm='LN',
    bbox_reparam=False,
    freeze_batch_norm=False,

    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,

    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,

    # Dataset parameters
    dataset_file="coco",
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,

    # Output parameters
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,

    # Distributed training parameters
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,

    # FP16
    fp16_eval=False,

    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        print_freq=print_freq,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs
    )
    return args

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """ """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos
    
def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    # TODO make it support different-sized images
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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LWDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 segmentation_head,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def forward(self, samples: NestedTensor, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.tensors, feat.mask
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = self.segmentation_head.sparse_forward if self.training else self.segmentation_head.forward

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.segmentation_head is not None:
                out['pred_masks'] = outputs_masks[-1]

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)

            cls_enc = torch.cat(cls_enc, dim=1)

            if self.segmentation_head is not None:
                masks_enc = seg_head_fwd(features[0].tensors, [hs_enc,], samples.tensors.shape[-2:], skip_blocks=True)[0]

            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['enc_outputs']['pred_masks'] = masks_enc
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['pred_masks'] = masks_enc

        return out

def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    torch.device(args.device)


    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, 'shape') else (args.resolution, args.resolution) if hasattr(args, 'resolution') else (640, 640),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = SegmentationHead(args.hidden_dim, args.dec_layers, downsample_ratio=args.mask_downsample_ratio) if args.segmentation_head else None

    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    return model

def download_file(url: str, filename: str) -> None:
    print("rory downloading url",url)
    response = requests.get(url, stream=True)
    total_size = int(response.headers['content-length'])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Optionally gather masks corresponding to the same top-K queries and resize to original size
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]}
                k_idx = topk_boxes[i]
                masks_i = torch.gather(out_masks[i], 0, k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]))  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)), mode='bilinear', align_corners=False)  # [K,1,H,W]
                res_i['masks'] = masks_i > 0.0
                results.append(res_i)
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'], strict=False)

        self.model = self.model.to(self.device)
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
    """
    The configuration for an RF-DETR Base model.
    """
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
    positional_encoding_size: int = 37

class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

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
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        download_pretrain_weights(self.model_config.pretrain_weights)


    def get_model(self, config: ModelConfig):
        """
        Retrieve a model instance based on the provided configuration.
        """
        return Model(**config.dict())

    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, 'class_names') and self.model.class_names:
            return {i+1: name for i, name in enumerate(self.model.class_names)}

        return COCO_CLASSES

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            self._has_warned_about_not_being_optimized_for_inference = True
            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
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

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = vF.normalize(img_tensor, self.means, self.stds)
            img_tensor = vF.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                # this could happen if someone manually changes self.model.resolution after optimizing the model
                raise ValueError(f"Resolution mismatch. "
                                 f"Model was optimized for resolution {self._optimized_resolution}, "
                                 f"but got {batch_tensor.shape[2]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(f"Batch size mismatch. "
                                     f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                     f"but got {batch_tensor.shape[0]}. "
                                     "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                     "Alternatively, you can recompile the optimized model for a different batch size "
                                     "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

        with torch.no_grad():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                return_predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    return_predictions["pred_masks"] = predictions[2]
                predictions = return_predictions
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
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

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
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
    """
    The configuration for an RF-DETR Small model.
    """
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
    """
    The configuration for an RF-DETR Medium model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"

class RFDETRMedium(RFDETR):
    size = "rfdetr-medium"
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

class RFDETRLarge(RFDETR):
    size = "rfdetr-large"
    def __init__(self, **kwargs):
        self.init_error = None
        self.is_deprecated = False
        try:
            super().__init__(**kwargs)
        except Exception as e:
            self.init_error = e
            self.is_deprecated = True
            try:
                super().__init__(**kwargs)
                logger.warning(
                    "\n"
                    "="*100 + "\n"
                    "WARNING: Automatically switched to deprecated model configuration, due to using deprecated weights. "
                    "This will be removed in a future version.\n"
                    "Please retrain your model with the new weights and configuration.\n"
                    "="*100 + "\n"
                )
            except Exception:
                raise self.init_error

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
  image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
  detections = model.predict(image, threshold=0.5)
  labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
  annotated_image = sv.BoxAnnotator().annotate(image, detections)
  annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
  np.testing.assert_allclose(detections.xyxy, excepted_xyxys[i], rtol=1e-3, atol=1e-3)
  annotated_image.save("annotated_image.jpg")

print("PASSED")