from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import drop, drop_path, trunc_normal_
import math
from timm.models.vision_transformer import VisionTransformer

class ConvBNReLU(nn.Sequential):
    """Basic Conv-BatchNorm-ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# Placeholder for DropPath if not using timm
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for logging

# ================ REPLACEMENTS FOR MMSEG/MMCV ================ #
class Registry:
    """Replacement for mmseg.models.builder.BACKBONES"""
    _registry = {}
    
    @classmethod
    def register_module(cls, name=None):
        def decorator(module_class):
            module_name = name if name is not None else module_class.__name__
            cls._registry[module_name] = module_class
            return module_class
        return decorator
    
    @classmethod
    def build(cls, cfg, **kwargs):
        if isinstance(cfg, dict):
            obj_type = cfg.pop('type')
            return cls._registry[obj_type](**cfg, **kwargs)
        return cls._registry[cfg](**kwargs)

BACKBONES = Registry()

# ================ ORIGINAL MODEL CODE BELOW (WITH MINIMAL CHANGES) ================ #
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)  # (HW)BC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)BC

        pos = self.positional_embedding
        cls_pos = pos[0:1, :]
        spatial_pos = F.interpolate(
            pos[1:,].reshape(1, self.spacial_dim, self.spacial_dim, -1).permute(0, 3, 1, 2),
            size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(-1, H*W).permute(1, 0)
        pos = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + pos.unsqueeze(1)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        x = x.permute(1, 0, 2)
        global_feat = x[:, 0]
        feature_map = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return global_feat, feature_map

class CLIPResNet(nn.Module):
    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__(); self.pretrained = pretrained; self.output_dim = output_dim; self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(width//2)
        self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(width//2)
        self.conv3 = nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2); self.relu = nn.ReLU(inplace=True)
        self._inplanes = width; self.layer1 = self._make_layer(width, layers[0]); self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2); self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.init_weights() # Call init weights
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]; self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks): layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def init_weights(self, pretrained=None): # Keep init_weights
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            logger.info(f"Loading ResNet weights from: {pretrained}")
            try:
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
                state_dict = {k.replace('visual.', ''): v for k, v in checkpoint.items() if k.startswith('visual.')}
                load_msg = self.load_state_dict(state_dict, strict=False)
                logger.info(f"ResNet weight loading finished. Missing: {load_msg.missing_keys}. Unexpected: {load_msg.unexpected_keys}")
            except Exception as e: logger.error(f"Error loading ResNet weights: {e}", exc_info=True)
        else: logger.warning("No pretrained weights specified for ResNet.")
    def forward(self, x): # Ensure returns tuple of stage outputs
        def stem(x): x = self.relu(self.bn1(self.conv1(x))); x = self.relu(self.bn2(self.conv2(x))); x = self.relu(self.bn3(self.conv3(x))); x = self.avgpool(x); return x
        outs = []; x = x.type(self.conv1.weight.dtype); x = stem(x)
        x = self.layer1(x); outs.append(x)
        x = self.layer2(x); outs.append(x)
        x = self.layer3(x); outs.append(x)
        x = self.layer4(x); outs.append(x)
        return tuple(outs)

class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__(); self.pretrained = pretrained; self.output_dim = output_dim; self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(width//2)
        self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(width//2)
        self.conv3 = nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2); self.relu = nn.ReLU(inplace=True)
        self._inplanes = width; self.layer1 = self._make_layer(width, layers[0]); self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2); self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32; self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        self.init_weights()
    def init_weights(self, pretrained=None): # Keep init_weights
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            logger.info(f"Loading ResNetAttn weights from: {pretrained}")
            try:
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
                state_dict = {}; visual_prefix = 'visual.'
                for k, v in checkpoint.items():
                    if k.startswith(visual_prefix): state_dict[k.replace(visual_prefix, '')] = v
                # Handle pos embed resizing for attnpool
                if 'attnpool.positional_embedding' in state_dict:
                    loaded_pos = state_dict['attnpool.positional_embedding']
                    current_pos = self.attnpool.positional_embedding
                    if loaded_pos.shape != current_pos.shape:
                         logger.warning(f"Resizing ResNetAttn pos embed from {loaded_pos.shape} to {current_pos.shape}")
                         cls_pos = loaded_pos[0:1, :]; H = W = self.input_resolution // 32
                         old_h = int(math.sqrt(loaded_pos.shape[0]-1)); C = cls_pos.shape[-1]
                         spatial_pos = F.interpolate(loaded_pos[1:,].reshape(1,old_h,old_h,C).permute(0,3,1,2), size=(H,W), mode='bilinear', align_corners=False)
                         state_dict['attnpool.positional_embedding'] = torch.cat([cls_pos, spatial_pos.permute(0,2,3,1).reshape(-1,C)], dim=0)
                load_msg = self.load_state_dict(state_dict, strict=False)
                logger.info(f"ResNetAttn weight loading finished. Missing: {load_msg.missing_keys}. Unexpected: {load_msg.unexpected_keys}")
            except Exception as e: logger.error(f"Error loading ResNetAttn weights: {e}", exc_info=True)
        else: logger.warning("No pretrained weights specified for ResNetAttn.")
    def _make_layer(self, planes, blocks, stride=1): # Keep _make_layer
        layers = [Bottleneck(self._inplanes, planes, stride)]; self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks): layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x): # Keep forward, returns multi-stage + attn pool output
        def stem(x): x = self.relu(self.bn1(self.conv1(x))); x = self.relu(self.bn2(self.conv2(x))); x = self.relu(self.bn3(self.conv3(x))); x = self.avgpool(x); return x
        x = x.type(self.conv1.weight.dtype); x = stem(x); outs = []
        x = self.layer1(x); outs.append(x)
        x = self.layer2(x); outs.append(x)
        x = self.layer3(x); outs.append(x)
        x = self.layer4(x) # Don't append final layer4 output directly
        x_global, x_local = self.attnpool(x) # Attn pool operates on layer4 output
        outs.append(x_local) # Append the spatial map from attnpool
        outs.append([x_global, x_local]) # Append the tuple as the very last element
        return tuple(outs)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        for resblock in self.resblocks: x = resblock(x)
        return self.resblocks(x)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class CLIPVisionTransformer(nn.Module):
    """
    CLIP Vision Transformer (ViT) backbone.
    Modified to remove internal FPN and output spatial features.
    Compatible with official OpenAI ViT-B/16 weights.
    """
    def __init__(self,
                 input_resolution: int = 224,
                 patch_size: int = 16, # Default for ViT-B/16
                 width: int = 768,    # Embedding dim for ViT-B/16
                 layers: int = 12,    # Num layers for ViT-B/16
                 heads: int = 12,     # Num heads for ViT-B/16
                 output_dim: int = 768, # Set to width to output features before projection
                 drop_path_rate: float = 0.0,
                 # --- VVVVV MODIFIED/ADDED VVVVV ---
                 out_indices = None, # Default to None
                 # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                 pretrained: str = None, # Path to pre-trained weights
                 **kwargs): # Absorb extra kwargs
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        # Set output_dim to transformer width
        self.output_dim = width
        self.layers = layers # Store layers count
        logger.info(f"CLIPVisionTransformer: Initializing with width={width}, layers={layers}, heads={heads}")
        logger.info(f"Internal feature dimension set to {self.output_dim} (transformer width)")

        # Patch Embedding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        # Class Embedding
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        # Positional Embedding
        self.grid_size = input_resolution // patch_size
        seq_len = self.grid_size ** 2 + 1 # +1 for class token
        self.positional_embedding = nn.Parameter(scale * torch.randn(seq_len, width))
        logger.info(f"ViT grid size: {self.grid_size}x{self.grid_size}. Sequence length (inc CLS): {seq_len}")

        # Pre-Transformer LayerNorm
        self.ln_pre = LayerNorm(width)

        # Transformer Blocks
        # Ensure your Transformer class definition is available and imported correctly
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        # Standard final layers from CLIP ViT (defined to load weights)
        self.ln_post = LayerNorm(width)
        # Define proj layer even if not used in forward, to match checkpoint keys
        self._clip_proj_dim = 512 # Standard CLIP ViT-B/16 output dim
        self.proj = nn.Parameter(scale * torch.randn(width, self._clip_proj_dim))

        # --- VVVVV Store out_indices VVVVV ---
        if out_indices is None:
            self.out_indices = [layers - 1] # Default to only the last layer index (0-based)
            logger.info("out_indices not specified, defaulting to output from last layer only.")
        else:
             # Ensure indices are valid
            if not isinstance(out_indices, (list, tuple)): raise TypeError("out_indices must be list or tuple")
            for i in out_indices:
                if not 0 <= i < layers: raise ValueError(f"Index {i} in out_indices is out of range for {layers} layers.")
            self.out_indices = sorted(list(set(out_indices))) # Store unique sorted indices
        logger.info(f"CLIPVisionTransformer will output features from layer indices: {self.out_indices}")
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        # Initialize weights AFTER all layers are defined
        self.init_weights()

    def _init_weights_default(self, m):
        """ Default initialization if no pretrained weights are loaded. """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None: nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Load pre-trained weights if specified."""
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            logger.info(f"Loading ViT weights from: {pretrained}")
            try:
                clip_model_jit = torch.jit.load(pretrained, map_location='cpu').float()
                checkpoint = clip_model_jit.state_dict()
                del clip_model_jit

                state_dict = OrderedDict()
                visual_prefix = 'visual.'
                for k, v in checkpoint.items():
                    if k.startswith(visual_prefix): state_dict[k.replace(visual_prefix, '')] = v
                del checkpoint

                # Handle positional embedding resizing
                if 'positional_embedding' in state_dict:
                    loaded_pos_embed = state_dict['positional_embedding']
                    if self.positional_embedding.shape != loaded_pos_embed.shape:
                        logger.info(f'ViT Resizing pos_embed from {loaded_pos_embed.shape} to {self.positional_embedding.shape}')
                        cls_pos = loaded_pos_embed[0:1, :]; patch_pos_loaded = loaded_pos_embed[1:, :]
                        num_patches_orig = patch_pos_loaded.shape[0]; grid_size_orig = int(np.sqrt(num_patches_orig))
                        embed_dim = self.positional_embedding.shape[-1]
                        if grid_size_orig * grid_size_orig != num_patches_orig:
                            logger.error(f"Cannot infer original square grid from loaded pos_embed shape {loaded_pos_embed.shape}.")
                            state_dict.pop('positional_embedding') # Don't load if incorrect
                        else:
                            logger.info(f"Original grid size from weights: {grid_size_orig}")
                            spatial_pos_orig_reshaped = patch_pos_loaded.reshape(1, grid_size_orig, grid_size_orig, embed_dim).permute(0, 3, 1, 2)
                            spatial_pos_resized = F.interpolate(spatial_pos_orig_reshaped,size=(self.grid_size, self.grid_size), mode='bilinear',align_corners=False)
                            spatial_pos = spatial_pos_resized.permute(0, 2, 3, 1).reshape(-1, embed_dim)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            if self.positional_embedding.shape == positional_embedding.shape:
                                state_dict['positional_embedding'] = positional_embedding; logger.info("Successfully resized positional embedding.")
                            else: logger.error(...); state_dict.pop('positional_embedding')

                # Check if proj layer shape in checkpoint matches definition (it likely won't if output_dim=width)
                if 'proj' in state_dict and self.proj.shape != state_dict['proj'].shape:
                    logger.warning(f"Checkpoint 'proj' layer shape {state_dict['proj'].shape} differs from model definition {self.proj.shape}. This is expected if output_dim=width.")
                    # Keep self.proj as defined, but remove the incompatible weight from state_dict to load
                    state_dict.pop('proj')

                # Load state dict non-strictly
                load_msg = self.load_state_dict(state_dict, strict=False)
                logger.info("ViT weight loading finished.")
                if load_msg.missing_keys: logger.warning(f"ViT Missing Keys: {load_msg.missing_keys}")
                if load_msg.unexpected_keys: logger.warning(f"ViT Unexpected Keys: {load_msg.unexpected_keys}") # Should be empty now ideally

            except FileNotFoundError: logger.error(f"Pretrained ViT file not found: {pretrained}")
            except Exception as e: logger.error(f"Error loading ViT weights: {e}", exc_info=True)
        else:
            logger.warning("No pretrained weights specified for ViT. Applying default initialization.")
            self.apply(self._init_weights_default)

    def interpolate_pos_encoding(self, x, H, W):
        """ Helper function to interpolate positional encoding. """
        N_input = x.shape[1] - 1 # Num patch tokens in input x
        N_loaded = self.positional_embedding.shape[0] - 1 # Num patches in loaded weights
        if N_input == N_loaded: # If input matches loaded size (e.g., 224 -> 14x14)
            return self.positional_embedding.to(x.dtype)

        logger.debug(f"Interpolating ViT positional embedding for grid {H}x{W} (input N={N_input}) from loaded {N_loaded} patches.")
        class_pos_embed = self.positional_embedding[0].unsqueeze(0) # [1, C]
        patch_pos_embed = self.positional_embedding[1:] # [N_loaded, C]
        dim = self.positional_embedding.shape[-1]
        grid_size_orig = int(np.sqrt(N_loaded))
        if grid_size_orig * grid_size_orig != N_loaded: logger.error(...); return self.positional_embedding.to(x.dtype)

        # Reshape & Interpolate: [N_loaded, C] -> [1, C, H0, W0] -> [1, C, H, W]
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, grid_size_orig, grid_size_orig, dim).permute(0, 3, 1, 2),
            size=(H, W), mode='bilinear', align_corners=False
        )
        # Reshape back & unsqueeze: [1, C, H, W] -> [H*W, C] -> [1, H*W, C]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # Prepend class token embedding: [1, 1, C]
        class_pos_embed = class_pos_embed.unsqueeze(1) # Add sequence dim: [1, 1, C]
        # Concatenate: [1, 1+H*W, C]
        new_pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)

        return new_pos_embed.squeeze(0).to(x.dtype) # Return shape [1+H*W, C]


    def forward(self, x: torch.Tensor):
        B, Cin, Hin, Win = x.shape
        # Patch embedding
        x = self.conv1(x)
        H, W = x.shape[-2:] # Actual grid height/width
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, width] (NLC format)

        # Prepend class token
        class_embed = self.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([class_embed, x], dim=1)  # [B, 1 + H*W, width]

        # Add positional embedding (interpolating if necessary)
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed

        # Pre-LN
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLC -> LND

        # --- ---
        intermediate_features = []
        # Pass through transformer blocks
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x) # Input/Output is LND
            # Check if current layer index is one we want to output
            if i in self.out_indices:
                # Process output for storage: LND -> NLC -> spatial
                out_feat_seq = x.permute(1, 0, 2) # LND -> NLC [B, 1+HW, D]

                # Applying final LayerNorm ONLY if it's the very last layer index requested
                # AND the last layer index IS actually the final block of the transformer
                if i == self.layers - 1:
                     logger.debug(f"Applying ln_post to output index {i}")
                     out_feat_seq = self.ln_post(out_feat_seq)

                patch_tokens = out_feat_seq[:, 1:, :] # Exclude CLS token [B, HW, D]
                D = patch_tokens.shape[-1]
                # Reshape to spatial map: [B, D, H, W]
                spatial_map = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
                intermediate_features.append(spatial_map)
                logger.debug(f"Stored intermediate feature from layer {i}, shape: {spatial_map.shape}")

        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        # Ensure features were collected for all specified indices
        if len(intermediate_features) != len(self.out_indices):
             logger.warning(f"Collected {len(intermediate_features)} features, but expected {len(self.out_indices)} based on out_indices. Check transformer loop.")
             # Handle error? Maybe return empty list or raise?
             if not intermediate_features: # If nothing was collected
                  logger.error("No intermediate features collected.")
                  return [] # Return empty

        # Return the list of collected spatial feature maps
        logger.debug(f"ViT returning {len(intermediate_features)} feature maps.")
        return intermediate_features # Returns list: [feat_idx1, feat_idx2, ...]


class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512, # <<< Default to 512 to match CLIP standard
                 # out_dim is removed, embed_dim is the final output dim
                 pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width, layers=transformer_layers,
            heads=transformer_heads, attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        # --- VVVVV MODIFIED TEXT PROJECTION VVVVV ---
        # Define projection based on embed_dim config
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # Store the intended output dimension
        self._output_dim = embed_dim
        logger.info(f"CLIPTextEncoder: Output dimension set to {self._output_dim}")
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
        self.init_weights() # Call init weights

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            logger.info(f"Loading TextEncoder weights from: {pretrained}")
            try:
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
                state_dict = OrderedDict()
                # Keys to load for text encoder
                text_keys = ['transformer.', 'token_embedding', 'positional_embedding', 'ln_final.', 'text_projection']
                count = 0
                for k in checkpoint.keys():
                    if any(k.startswith(p) for p in text_keys):
                        # Handle positional embedding truncation
                        if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                             logger.warning(f'Truncating Text pos_embed from {checkpoint[k].size(0)} to {self.context_length}')
                             state_dict[k] = checkpoint[k][:self.context_length]
                        
                        elif k == 'text_projection':
                            # Check if loaded shape matches current model's projection shape
                            if checkpoint[k].shape == self.text_projection.shape:
                                state_dict[k] = checkpoint[k] # Shapes match, load it
                            else:
                                # Shapes mismatch (e.g., config embed_dim differs from ckpt)
                                logger.warning(f"Text projection shape mismatch! Checkpoint shape: {checkpoint[k].shape}, Model shape: {self.text_projection.shape}. NOT loading 'text_projection'. It will be randomly initialized.")
                                # Do NOT add checkpoint[k] to state_dict if shapes mismatch
                        
                        else:
                             state_dict[k] = checkpoint[k] # Load other matching keys
                        count +=1

                logger.info(f"Filtered {count} keys for TextEncoder.")
                load_msg = self.load_state_dict(state_dict, strict=False)
                logger.info("TextEncoder weight loading finished.")
                if load_msg.missing_keys: logger.warning(f"TextEncoder Missing Keys: {load_msg.missing_keys}")
                if load_msg.unexpected_keys: logger.warning(f"TextEncoder Unexpected Keys: {load_msg.unexpected_keys}")

            except FileNotFoundError: logger.error(f"Pretrained file not found: {pretrained}")
            except Exception as e: logger.error(f"Error loading TextEncoder weights: {e}", exc_info=True)
        else:
            logger.warning("No pretrained weights specified for TextEncoder. Applying default init.")
            self.apply(self._init_weights_default) # Apply default init if no pretrained

        
        # Initialize text_projection specifically if it wasn't loaded (missing or shape mismatch)
        if 'text_projection' not in self.state_dict() or (pretrained and 'text_projection' not in state_dict):
             logger.info("Applying specific init to text_projection layer.")
             # Example init (scale based on width = transformer_width)
             scale = self.transformer.width ** -0.5
             nn.init.normal_(self.text_projection, std=scale)
        

    def _init_weights_default(self, m):
        """ Default initialization if no pretrained weights are loaded. """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
             nn.init.normal_(m.weight, std=0.02)


    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf")); mask.triu_(1); return mask

    def forward(self, text):
        x = self.token_embedding(text)
        # Check positional embedding shape against input context length
        if x.shape[1] != self.positional_embedding.shape[0]:
             logger.error(f"Input text context length {x.shape[1]} != pos embed length {self.positional_embedding.shape[0]}")
             # Handle error or truncate/pad pos embed dynamically if needed
             pos_embed = self.positional_embedding[:x.shape[1], :] # Simple truncation
        else:
             pos_embed = self.positional_embedding

        x = x + pos_embed.to(x.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x[batch_size, n_ctx, transformer_width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # [batch_size, transformer_width] @ [transformer_width, embed_dim] -> [batch_size, embed_dim]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class ViTFeatureFusionNeck(nn.Module):
    """
    A simple neck to fuse features from different ViT layers.
    Assumes all input features have the same spatial size but different semantics.
    Applies separate convs, concatenates, then fuses with a final conv.
    """
    def __init__(self,
                 in_channels_list, # List of input channels (e.g., [768, 768] for ViT-B/16)
                 out_channels,     # Desired output channel dimension for the fused feature map
                 inter_channels=None # Optional intermediate channel size for processing layers
                 ):
        super().__init__()
        if not isinstance(in_channels_list, (list, tuple)):
             raise TypeError("in_channels_list must be a list or tuple")
        if inter_channels is None:
             inter_channels = out_channels # Default: process directly to out_channels

        self.num_inputs = len(in_channels_list)
        self.process_layers = nn.ModuleList()

        total_inter_channels = 0
        for i in range(self.num_inputs):
             # Apply a Conv block (e.g., 3x3) to each input feature level
             # This allows each level to be processed independently first
             # Use ConvBNReLU defined above
             layer = ConvBNReLU(in_channels_list[i], inter_channels, kernel_size=3, padding=1)
             self.process_layers.append(layer)
             total_inter_channels += inter_channels
             logger.info(f"Fusion Neck: Process layer {i} ({in_channels_list[i]} -> {inter_channels})")

        # Final fusion layer after concatenation
        # Takes concatenated features (total_inter_channels) and outputs 'out_channels'
        # Use ConvBNReLU defined above
        self.fusion_layer = ConvBNReLU(total_inter_channels, out_channels, kernel_size=1, padding=0) # Use 1x1 conv for fusion
        logger.info(f"Fusion Neck: Fusion layer ({total_inter_channels} -> {out_channels})")

        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, m):
         if isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
             nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # Input 'features' is a list of tensors [feat_idx1, feat_idx2, ...]
        if len(features) != self.num_inputs:
             logger.error(f"Fusion Neck received {len(features)} inputs, expected {self.num_inputs}")
             # Handle error gracefully, maybe return only first feature? Needs better handling.
             return [features[0]] if features else [] # Return list with first or empty list

        processed_features = []
        for i in range(self.num_inputs):
            feat = self.process_layers[i](features[i])
            processed_features.append(feat)

        # Concatenate processed features along the channel dimension
        concatenated_features = torch.cat(processed_features, dim=1)
        logger.debug(f"Fusion Neck: Concatenated shape: {concatenated_features.shape}")

        # Apply final fusion layer
        fused_features = self.fusion_layer(concatenated_features)
        logger.debug(f"Fusion Neck: Fused output shape: {fused_features.shape}")

        # Return the single fused feature map, wrapped in a list for consistency
        return [fused_features]


class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        x_text = self.token_embedding(text)  # n_clas, n_text, C
        K, N1, C = x_text.shape
        B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x
    

class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)