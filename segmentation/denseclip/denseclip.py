# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Use standard logging
import numpy as np
from collections import OrderedDict # Added for state dict filtering

# Explicitly import necessary components from within the denseclip package
# Assuming these are defined in a sub-module named 'models' or similar
from .models import (
    CLIPResNet, CLIPTextEncoder, CLIPVisionTransformer,
    CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder
)
# Import head classes if they are defined locally (adjust path if needed)
# Assuming IdentityHead might need custom implementations or replacements
# from .heads import IdentityHead # Uncomment if IdentityHead is defined and needed

# Setup logger for this module
logger = logging.getLogger(__name__)

# Need to handle FPN and FPNHead - replace with standard implementations if possible
# Example using torchvision FPN (might need adaptation)
try:
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    from torchvision.models.segmentation.fcn import FCNHead # Example replacement for FPNHead
    TORCHVISION_AVAILABLE = True
    logger.info("Successfully imported FeaturePyramidNetwork and FCNHead from torchvision.")
except ImportError:
    TORCHVISION_AVAILABLE = False
    FeaturePyramidNetwork = None
    FCNHead = None
    # Define dummy placeholders if torchvision is unavailable, to avoid errors if configs mention them
    class FeaturePyramidNetwork(nn.Module): 
         def __init__(self, **kwargs): 
               super().__init__(); self.dummy = nn.Identity()
    class FCNHead(nn.Module): 
         def __init__(self, **kwargs): 
          super().__init__(); self.dummy = nn.Identity()
    class LastLevelMaxPool(nn.Module): 
         def forward(self, *args): 
              return [torch.zeros(1)] # Needs a dummy forward
    logger.warning("Warning: torchvision not found or FPN/FCNHead not available. Neck and Decode Head using dummy placeholders.")


# Import tokenize from the utils module
from .utils import tokenize

# Configure logging (can be done once at the top level of your application)
# logging.basicConfig(level=logging.INFO) # Example basic config

# ================== REPLACEMENTS FOR MMSEG/MMCV (Keep as is) ================== #
# class Registry ...
# SEGMENTORS = Registry()
# def resize(...)
# def add_prefix(...)
# class BaseSegmentor(...)
# ================== END REPLACEMENTS ================== #


#@SEGMENTORS.register_module() # Decorator might not be needed if SEGMENTORS registry isn't used
class DenseCLIP(nn.Module): # Inherit directly from nn.Module
    """
    DenseCLIP segmentor implementation without mmsegmentation dependencies.
    Includes CLIP pre-trained weight loading.
    """
    def __init__(self,
                 backbone, # Config dict for backbone
                 text_encoder, # Config dict for text encoder
                 decode_head, # Config dict for decode head
                 class_names, # List of class names
                 context_length,
                 # --- Arguments with Defaults ---
                 context_decoder=None, # Optional config dict
                 neck=None, # Optional config dict
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False, # Whether to feed text embeddings to decode head
                 tau=0.07,
                 auxiliary_head=None, # Optional config dict
                 identity_head=None, # Optional config dict
                 train_cfg=None, # Keep for potential future use
                 test_cfg=None, # Keep for potential future use
                 token_embed_dim=512, # Usually related to text token embedding before transformer
                 text_dim=1024, # Target dimension for text features after projection/encoder
                 clip_pretrained_path=None, # <<< Path to CLIP weights <<<
                 **kwargs): # Use kwargs for flexibility
        super().__init__() # Call nn.Module's init

        # --- Store basic attributes ---
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.context_length = context_length
        self.context_feature = context_feature
        self.score_concat_index = score_concat_index
        self.text_head = text_head
        self.tau = tau
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.align_corners = False # Default, updated by decode_head config


        # --- Build Backbone ---
        backbone_cfg = backbone.copy()
        backbone_type = backbone_cfg.pop('type')
        logger.info(f"Building backbone: {backbone_type} with config: {backbone_cfg}")
        # ... (keep existing backbone build logic) ...
        if backbone_type == "CLIPResNet":
             self.backbone = CLIPResNet(**backbone_cfg)
             backbone_out_channels = backbone_cfg.get('width', 64) * 8 * 4
        elif backbone_type == "CLIPResNetWithAttention":
             self.backbone = CLIPResNetWithAttention(**backbone_cfg)
             backbone_out_channels = backbone_cfg.get('output_dim', 1024)
        elif backbone_type == "CLIPVisionTransformer":
              self.backbone = CLIPVisionTransformer(**backbone_cfg)
              backbone_out_channels = backbone_cfg.get('output_dim', backbone_cfg.get('width'))
              if backbone_out_channels is None: raise ValueError("Could not determine output_dim for CLIPVisionTransformer")
        else:
             raise ValueError(f"Unsupported backbone type: {backbone_type}")
        logger.info(f"Built backbone. Inferred output channels/dim: {backbone_out_channels}")


        # --- Build Text Encoder ---
        text_encoder_cfg = text_encoder.copy()
        text_encoder_type = text_encoder_cfg.pop('type')
        logger.info(f"Building text encoder: {text_encoder_type} with config: {text_encoder_cfg}")
        # ... (keep existing text encoder build logic) ...
        text_encoder_out_dim = text_encoder_cfg.get('embed_dim', text_dim)
        if text_encoder_out_dim != text_dim: logger.warning(f"text_encoder config embed_dim ({text_encoder_out_dim}) != model text_dim ({text_dim}). Ensure this is intended.")
        if text_encoder_type == "CLIPTextEncoder": self.text_encoder = CLIPTextEncoder(**text_encoder_cfg)
        elif text_encoder_type == "CLIPTextContextEncoder": self.text_encoder = CLIPTextContextEncoder(**text_encoder_cfg)
        else: raise ValueError(f"Unsupported text_encoder type: {text_encoder_type}")
        logger.info(f"Built text encoder.")


        # --- Load Pre-trained CLIP Weights ---
        if clip_pretrained_path:
            logger.info(f"Attempting to load pre-trained CLIP weights from: {clip_pretrained_path}")
            try:
                logger.info("Loading TorchScript model...")
                clip_model_jit = torch.jit.load(clip_pretrained_path, map_location="cpu")
                logger.info("TorchScript model loaded successfully.")
                clip_state_dict = clip_model_jit.state_dict()
                logger.info(f"Extracted state_dict with {len(clip_state_dict)} keys.")

                # Prepare and load Visual Weights
                visual_weights = OrderedDict(); visual_prefix = 'visual.'; count_visual = 0
                for k, v in clip_state_dict.items():
                    if k.startswith(visual_prefix): visual_weights[k[len(visual_prefix):]] = v; count_visual += 1
                if visual_weights:
                    logger.info(f"Loading {count_visual} keys into visual backbone (strict=False)...")
                    load_msg_visual = self.backbone.load_state_dict(visual_weights, strict=False)
                    logger.info(f"Visual backbone loading message: {load_msg_visual}")
                    if load_msg_visual.missing_keys: logger.warning(f"Visual backbone MISSING keys: {load_msg_visual.missing_keys}")
                    if load_msg_visual.unexpected_keys: logger.warning(f"Visual backbone UNEXPECTED keys: {load_msg_visual.unexpected_keys}")
                else: logger.warning(f"No keys matching '{visual_prefix}' prefix found...")

                # Prepare and load Text Weights
                text_weights = OrderedDict(); text_prefixes_or_keys = ('transformer.', 'token_embedding.', 'positional_embedding', 'ln_final.', 'text_projection'); count_text = 0
                for k, v in clip_state_dict.items():
                    if k.startswith(text_prefixes_or_keys): text_weights[k] = v; count_text += 1
                if text_weights:
                    logger.info(f"Loading {count_text} keys into text encoder (strict=False)...")
                    load_msg_text = self.text_encoder.load_state_dict(text_weights, strict=False)
                    logger.info(f"Text encoder loading message: {load_msg_text}")
                    if load_msg_text.missing_keys: logger.warning(f"Text encoder MISSING keys: {load_msg_text.missing_keys}")
                    if load_msg_text.unexpected_keys: logger.warning(f"Text encoder UNEXPECTED keys: {load_msg_text.unexpected_keys}")
                else: logger.warning("No keys matching typical text encoder prefixes/names found...")

                del clip_model_jit, clip_state_dict # Free memory
                logger.info("CLIP jit model and state_dict deleted from memory after loading.")

            except FileNotFoundError: logger.error(f"Pre-trained CLIP file not found: {clip_pretrained_path}")
            except Exception as e: logger.error(f"Error loading pre-trained CLIP weights: {e}", exc_info=True)
        else:
            logger.warning("No 'clip_pretrained_path' provided...")


        # --- Add Visual Projection Layers IF needed (MODIFIED) ---
        self.vis_proj = None
        self.global_proj = None # Initialize global_proj here
        if backbone_out_channels != text_dim:
            logger.info(f"Visual spatial feature dim ({backbone_out_channels}) != Text dim ({text_dim}). Adding spatial projection (Conv2d).")
            self.vis_proj = nn.Conv2d(backbone_out_channels, text_dim, kernel_size=1)

            logger.info(f"Visual global feature dim ({backbone_out_channels}) != Text dim ({text_dim}). Adding global projection (Linear).")
            self.global_proj = nn.Linear(backbone_out_channels, text_dim) # Add Linear layer
        else:
            logger.info(f"Visual feature dim ({backbone_out_channels}) matches text dim ({text_dim}). No projection needed.")


        # --- Build Context Decoder ---
        self.context_decoder = None
        if context_decoder:
            context_decoder_cfg = context_decoder.copy()
            context_decoder_type = context_decoder_cfg.pop('type')
            logger.info(f"Building context decoder: {context_decoder_type} with config: {context_decoder_cfg}")
            if context_decoder_type == "ContextDecoder":
                 context_decoder_cfg.setdefault('visual_dim', text_dim)
                 self.context_decoder = ContextDecoder(**context_decoder_cfg)
            else: raise ValueError(f"Unsupported context_decoder type: {context_decoder_type}")
        else: logger.info("No context decoder configured.")


        # --- Build Neck (MODIFIED with checks) ---
        self.neck = None
        self._neck_out_keys = None
        neck_out_channels = text_dim # Default if no neck
        if neck:
            neck_cfg = neck.copy()
            neck_type = neck_cfg.pop('type')
            logger.info(f"Building neck: {neck_type} with config: {neck_cfg}")
            if neck_type == "FPN" and FeaturePyramidNetwork is not None:
                 # --- Determine FPN input channels ---
                 default_fpn_in_channels = [
                     backbone_cfg.get('width', 64) * 1 * 4,
                     backbone_cfg.get('width', 64) * 2 * 4,
                     backbone_cfg.get('width', 64) * 4 * 4,
                     backbone_cfg.get('width', 64) * 8 * 4
                 ]
                 if isinstance(self.backbone, CLIPVisionTransformer):
                      logger.error("Using FPN neck with ViT backbone requires specific multi-scale feature extraction.")
                      in_channels_list = neck_cfg.get('in_channels', [backbone_out_channels] * 4)
                 else:
                      in_channels_list = neck_cfg.get('in_channels', default_fpn_in_channels)
                 logger.info(f"FPN using input channels: {in_channels_list}")

                 out_channels = neck_cfg.get('out_channels', 256)
                 num_outs = neck_cfg.get('num_outs', len(in_channels_list))
                 extra_blocks = None
                 if num_outs > len(in_channels_list):
                      if LastLevelMaxPool is not None: extra_blocks = LastLevelMaxPool(); logger.info("Adding LastLevelMaxPool...")
                      else: logger.warning("LastLevelMaxPool not available...")

                 # --- VVVVVV ADDED CHECKS VVVVVV ---
                 if not isinstance(in_channels_list, list) or not in_channels_list:
                     raise ValueError(f"FPN 'in_channels_list' is invalid or empty: {in_channels_list}")
                 if not isinstance(out_channels, int) or out_channels <= 0:
                     raise ValueError(f"FPN 'out_channels' must be a positive integer: {out_channels}")
                 logger.info(f"Instantiating FeaturePyramidNetwork with in_channels={in_channels_list}, out_channels={out_channels}")
                 # --- ^^^^^^ END CHECKS ^^^^^^ ---

                 # Instantiate FPN
                 self.neck = FeaturePyramidNetwork(
                      in_channels_list=in_channels_list,
                      out_channels=out_channels,
                      extra_blocks=extra_blocks
                 )
                 neck_out_channels = out_channels
                 self._neck_out_keys = [str(i) for i in range(num_outs)]
                 logger.info(f"Built torchvision FPN. Output channels: {neck_out_channels}. Assumed keys: {self._neck_out_keys}")

            elif neck_type == "FPN": logger.error("Torchvision FPN not available...")
            else: raise ValueError(f"Unsupported neck type: {neck_type}")
        else: logger.info("No neck configured.")


        # --- Build Decode Head ---
        self.decode_head = None
        self._decode_head_cfg = None
        if decode_head:
            # ... (decode head build logic - should be okay now) ...
            decode_head_cfg = decode_head.copy()
            decode_head_type = decode_head_cfg.pop('type')
            logger.info(f"Building decode head: {decode_head_type} with config: {decode_head_cfg}")
            self._decode_head_cfg = decode_head_cfg
            self.align_corners = decode_head_cfg.get('align_corners', False)
            decode_num_classes = decode_head_cfg.get('num_classes', self.num_classes)
            if decode_num_classes != self.num_classes: logger.warning(...); self.num_classes = decode_num_classes


            if decode_head_type == "FPNHead" and FCNHead is not None:
                    in_channels = decode_head_cfg.get('in_channels', neck_out_channels)
                    if in_channels != neck_out_channels: logger.warning(...)
                    channels = decode_head_cfg.get('channels', 256)
                    # Get dropout from config, but DON'T pass it to FCNHead constructor
                    dropout_ratio = decode_head_cfg.get('dropout_ratio', 0.1) # Keep this line if you plan to use it elsewhere
                    # --- VVVVVV REMOVE dropout=dropout VVVVVV ---
                    self.decode_head = FCNHead(in_channels=in_channels, channels=channels) # Removed dropout=dropout
                    # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                    num_intermediate_channels = channels
                    self.decode_head.classifier = nn.Conv2d(num_intermediate_channels, self.num_classes, kernel_size=1)
                    logger.info(f"Built torchvision FCNHead and replaced classifier for {self.num_classes} classes.")
                    # ... (rest of decode head logic) ...


            elif decode_head_type == "FPNHead": logger.error("Torchvision FCNHead not available...")
            elif decode_head_type == "IdentityHead" and IDENTITY_HEAD_AVAILABLE: self.decode_head = IdentityHead(**decode_head_cfg); logger.info("Built IdentityHead.")
            elif decode_head_type == "IdentityHead": raise ValueError("IdentityHead specified but class definition not found.")
            else: raise ValueError(f"Unsupported decode_head type: {decode_head_type}")

        self.with_decode_head = self.decode_head is not None
        if not self.with_decode_head: logger.warning("No decode head was built.")


        # --- Build Auxiliary Head ---
        self.auxiliary_head = None
        self.with_auxiliary_head = False


        if auxiliary_head:
               aux_head_cfg = auxiliary_head.copy()
               aux_head_type = aux_head_cfg.pop('type')
               logger.info(f"Building auxiliary head: {aux_head_type} with config: {aux_head_cfg}")
               if aux_head_type == "FCNHead" and FCNHead is not None:
                    default_aux_in_channels = backbone_cfg.get('width', 64) * 4 * 4
                    aux_in_channels = aux_head_cfg.get('in_channels', default_aux_in_channels)
                    logger.info(f"Auxiliary head using input channels: {aux_in_channels}")
                    aux_channels = aux_head_cfg.get('channels', 128)
                    # Get dropout, but don't pass it
                    aux_dropout_ratio = aux_head_cfg.get('dropout_ratio', 0.1) # Keep if needed elsewhere
                    # --- VVVVVV REMOVE dropout=aux_dropout VVVVVV ---
                    self.auxiliary_head = FCNHead(in_channels=aux_in_channels, channels=aux_channels) # Removed dropout=aux_dropout
                    # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                    self.auxiliary_head.classifier = nn.Conv2d(aux_channels, self.num_classes, kernel_size=1)
                    self.with_auxiliary_head = True
                    logger.info(f"Built auxiliary FCNHead for {self.num_classes} classes.")
               # ... (rest of aux head logic) ...


               elif aux_head_type == "FCNHead": logger.error("Torchvision FCNHead not available...")
               else: logger.warning(f"Auxiliary head type '{aux_head_type}' not explicitly supported.")


        # --- Build Identity Head ---
        self.identity_head = None
        self.with_identity_head = False
        if identity_head:
            # ... (identity head build logic - should be okay now) ...
            if IDENTITY_HEAD_AVAILABLE and not isinstance(self.decode_head, IdentityHead):
                id_head_cfg = identity_head.copy(); id_head_type = id_head_cfg.pop('type')
                logger.info(f"Building separate identity head: {id_head_type} with config: {id_head_cfg}")
                if id_head_type == "IdentityHead":
                    try: self.identity_head = IdentityHead(**id_head_cfg); self.with_identity_head = True
                    except Exception as e: logger.error(f"Error instantiating IdentityHead: {e}")
                else: logger.warning(f"Identity head type '{id_head_type}' not explicitly supported.")
            elif isinstance(self.decode_head, IdentityHead): logger.info(...); self.identity_head = self.decode_head; self.with_identity_head = True
            else: logger.warning("IdentityHead specified but class definition not found...")


        # --- Tokenization and Learnable Parameters ---
        logger.info(f"Tokenizing {len(self.class_names)} class names with context length {self.context_length}...")
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in self.class_names])
        # ... (prompt length calculation - should be okay now) ...
        if not hasattr(self, 'text_encoder'): raise RuntimeError("Text encoder not initialized before prompt calculation.")
        text_encoder_context_length = getattr(self.text_encoder, 'context_length', 77)
        logger.info(f"Text encoder context length capacity: {text_encoder_context_length}")
        if self.context_length > text_encoder_context_length: logger.warning(...); self.context_length = text_encoder_context_length
        prompt_context_length = text_encoder_context_length - self.context_length
        # ... (context/gamma initialization - should be okay now) ...
        if isinstance(self.text_encoder, CLIPTextContextEncoder):
            _token_embed_dim = text_encoder_cfg.get('transformer_width', token_embed_dim)
            _text_dim_gamma = text_dim
            if prompt_context_length > 0:
                 self.contexts = nn.Parameter(torch.randn(1, prompt_context_length, _token_embed_dim))
                 nn.init.trunc_normal_(self.contexts, std=.02)
                 logger.info(f"Initialized learnable text contexts ({prompt_context_length} tokens) with dim {_token_embed_dim}.")
            else: self.contexts = None; logger.info("No space for learnable contexts.")
            self.gamma = nn.Parameter(torch.ones(_text_dim_gamma) * 1e-4)
            logger.info(f"Initialized learnable gamma with dim {_text_dim_gamma}.")
        else:
            self.contexts = None; self.gamma = None
            logger.info("Standard text encoder used. Learnable contexts/gamma not initialized.")


        # --- Custom Weight Initialization for Non-CLIP parts ---
        logger.info("Applying custom weight initialization to non-CLIP modules...")
        self._init_non_clip_weights() # Call the helper function


    def _init_weights_fn(self, m):
        """Helper function for applying initialization."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
             # Kaiming initialization is common for Conv layers in segmentation heads/necks
             try: # Use try-except for robustness
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with Kaiming normal.")
             except AttributeError: pass # Skip modules without weight/bias like pooling
        elif classname.find('Linear') != -1:
             try:
                 nn.init.normal_(m.weight, 0, 0.01)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with Normal(0, 0.01).")
             except AttributeError: pass
        elif classname.find('BatchNorm') != -1: # Covers BatchNorm1d, BatchNorm2d etc.
             try:
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with constants (1, 0).")
             except AttributeError: pass
        elif classname.find('GroupNorm') != -1:
             try:
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with constants (1, 0).")
             except AttributeError: pass
        # Add other layer types if needed (e.g., LayerNorm)

    def _init_non_clip_weights(self):
        """Initialize weights for modules NOT loaded from CLIP."""
        logger.info("Applying custom initialization using _init_weights_fn...")
        modules_to_init = []
        if self.vis_proj is not None: modules_to_init.append(('vis_proj', self.vis_proj))
        if self.context_decoder is not None: modules_to_init.append(('context_decoder', self.context_decoder))
        if self.neck is not None: modules_to_init.append(('neck', self.neck))
        if self.decode_head is not None: modules_to_init.append(('decode_head', self.decode_head))
        if self.auxiliary_head is not None: modules_to_init.append(('auxiliary_head', self.auxiliary_head))
        # Handle identity_head only if it's a separate module instance
        if self.with_identity_head and self.identity_head is not None and self.identity_head is not self.decode_head:
            modules_to_init.append(('identity_head', self.identity_head))

        for name, module in modules_to_init:
             logger.info(f"Initializing module: {name}...")
             module.apply(self._init_weights_fn)
             # Special handling for final classifier layers (common practice)
             if name in ['decode_head', 'auxiliary_head'] and hasattr(module, 'classifier') and isinstance(module.classifier, nn.Conv2d):
                 logger.info(f"...Initializing final classifier of {name} with Normal(0, 0.01)...")
                 nn.init.normal_(module.classifier.weight, mean=0, std=0.01)
                 if module.classifier.bias is not None:
                      nn.init.constant_(module.classifier.bias, 0)

    def extract_feat(self, img):
        """Extract features from images using the backbone."""
        # Assumes backbone returns features in a format usable by _process_features
        # For ResNet, typically a tuple/list of feature maps from different stages
        # For ViT, might be different (e.g., sequence of patch embeddings + class token)
        # Ensure backbone's forward method matches expectations
        x = self.backbone(img)
        logger.debug(f"Backbone output type: {type(x)}")
        if isinstance(x, (list, tuple)):
             logger.debug(f"Backbone output features: {[f.shape for f in x]}")
        elif torch.is_tensor(x):
             logger.debug(f"Backbone output feature shape: {x.shape}")
        return x

    def _process_features(self, x):
        """
        Handles feature processing after backbone extraction.
        Applies projection, calculates text features, context fusion, and score map.
        Returns:
            text_embeddings (Tensor): Shape [B, K, C_text]
            features_for_head (list[Tensor] or Tensor): Features to be passed to neck/head.
            score_map (Tensor): Shape [B, K, H_vis, W_vis] (using potentially projected visual features).
            _x_orig (list[Tensor]): Original backbone feature maps (before score map concat).
        """
        # --- Input Validation ---
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError(f"Backbone output 'x' must be a non-empty list or tuple of features. Got: {type(x)}")

        _x_orig = list(x) # Keep original features
        visual_embeddings = None # Spatial features for score map
        global_feat = None # Global pooled feature

        # --- Extract Global and Spatial Features ---
        # ... (keep logic to get initial visual_embeddings and global_feat) ...
        if isinstance(x[-1], (list, tuple)) and len(x[-1]) == 2 and isinstance(x[-1][0], torch.Tensor) and isinstance(x[-1][1], torch.Tensor):
             if x[-1][0].ndim == 2 and x[-1][1].ndim == 4:
                 logger.debug("Using features from AttentionPool2d-like output (global, spatial).")
                 _x_orig = list(x[:-1])
                 global_feat, visual_embeddings = x[-1]
             else:
                 logger.debug("Last element tuple/list doesn't match attn pool. Using last element as spatial.")
                 visual_embeddings = x[-1] # Might need adjustment depending on actual structure
        elif torch.is_tensor(x[-1]):
             logger.debug("Using last tensor from backbone output as spatial features.")
             visual_embeddings = x[-1]
        else: raise ValueError(...)
        if visual_embeddings is None or visual_embeddings.ndim != 4: raise ValueError(...)
        if global_feat is None: global_feat = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).flatten(1); logger.debug(...)
        elif global_feat.ndim != 2: raise ValueError(...)
        # --- End Extract ---

        B, C_vis_orig, H_vis, W_vis = visual_embeddings.shape
        C_glob_orig = global_feat.shape[1]
        if C_vis_orig != C_glob_orig: logger.warning(...) # Keep this warning


        # --- VVVVVVVV APPLY GLOBAL PROJECTION VVVVVVVV ---
        if self.global_proj is not None:
             logger.debug(f"Applying global projection: {C_glob_orig} -> {self.global_proj.out_features}")
             global_feat = self.global_proj(global_feat) # Project [B, C_orig] -> [B, C_proj]
             C_glob = global_feat.shape[1] # Get projected global dim
             logger.debug(f"Projected global_feat shape: {global_feat.shape}")
        else:
             C_glob = C_glob_orig # Use original global dim if no projection
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---


        # --- Apply Visual Spatial Projection IF defined ---
        if self.vis_proj is not None:
             logger.debug(f"Applying visual spatial projection: {C_vis_orig} -> {self.vis_proj.out_channels}")
             visual_embeddings = self.vis_proj(visual_embeddings)
             B, C_vis, H_vis, W_vis = visual_embeddings.shape # C_vis is now projected dim
             logger.debug(f"Projected spatial visual_embeddings shape: {visual_embeddings.shape}")
        else:
             C_vis = C_vis_orig


        # --- Check consistency AFTER projections ---
        if C_vis != C_glob:
             # This should ideally not happen now if projections are set up correctly
             logger.error(f"Projected spatial dim C_vis ({C_vis}) != projected global dim C_glob ({C_glob}). Check projection layers.")


        # --- Prepare Visual Context for Context Decoder ---
        visual_context = None
        if self.context_decoder:
            if self.context_feature == 'attention':
                # --- VVVVVV USE PROJECTED global_feat, REMOVE WARNING/RE-POOL VVVVVV ---
                # Now global_feat (if projected) and C_vis (if projected) should both match text_dim (1024)
                global_feat_ctx = global_feat # Use the (potentially projected) global_feat directly

                # REMOVE these lines that caused the warning and re-pooling:
                # if global_feat.shape[1] != C_vis:
                #      logger.warning(f"Context feature 'attention': global_feat dim ({global_feat.shape[1]}) != C_vis ({C_vis}). Re-pooling projected features for context.") # REMOVE
                #      global_feat_ctx = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).view(B, C_vis) # REMOVE
                # else:
                #     global_feat_ctx = global_feat # Use directly

                # Create context [B, N, C] where N = 1 (global) + H*W (spatial)
                visual_context = torch.cat([global_feat_ctx.unsqueeze(1), visual_embeddings.flatten(2).permute(0, 2, 1)], dim=1)
                # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

            elif self.context_feature == 'backbone':
                 # ... (logic for 'backbone' context remains the same) ...
                 last_backbone_feat = _x_orig[-1]
                 B_b, C_b, H_b, W_b = last_backbone_feat.shape
                 visual_context = last_backbone_feat.view(B_b, C_b, -1).permute(0, 2, 1) # [B, H*W, C_b]
                 # Still might need projection check here depending on ContextDecoder needs
                 if hasattr(self.context_decoder, 'visual_dim') and C_b != self.context_decoder.visual_dim:
                     logger.warning(f"Context feature 'backbone': Feature dim ({C_b}) != context_decoder expected dim ({self.context_decoder.visual_dim}).")

            else:
                 raise ValueError(f"Invalid context_feature type: {self.context_feature}")

            if visual_context is not None:
                 logger.debug(f"Prepared visual context shape ({self.context_feature}): {visual_context.shape}")


        # --- Text Feature Calculation ---
        # ... (keep text feature calculation logic) ...
        if not hasattr(self, 'text_encoder'): raise AttributeError("text_encoder missing")
        text_embeddings_device = next(self.text_encoder.parameters()).device
        logger.debug(f"Moving text tokens to device: {text_embeddings_device}")
        tokenized_texts = self.texts.to(text_embeddings_device)
        if isinstance(self.text_encoder, CLIPTextContextEncoder) and self.contexts is not None:
             logger.debug("Using CLIPTextContextEncoder with learnable contexts.")
             contexts_device = self.contexts.to(text_embeddings_device)
             text_embeddings = self.text_encoder(tokenized_texts, contexts_device).expand(B, -1, -1)
        elif isinstance(self.text_encoder, CLIPTextEncoder):
             logger.debug("Using standard CLIPTextEncoder.")
             text_embeddings = self.text_encoder(tokenized_texts).expand(B, -1, -1)
        else: raise TypeError(...)
        logger.debug(f"Raw text embeddings shape: {text_embeddings.shape}")

        # Apply Context Decoder Fusion
        if self.context_decoder and visual_context is not None:
            # ... (keep context decoder fusion logic) ...
            if self.gamma is None: raise AttributeError(...)
            logger.debug(f"Applying context decoder...")
            visual_context_device = visual_context.to(text_embeddings_device)
            try:
                 text_diff = self.context_decoder(text_embeddings, visual_context_device)
                 logger.debug(f"Context decoder output (text_diff) shape: {text_diff.shape}")
                 gamma_device = self.gamma.to(text_embeddings_device)
                 text_embeddings = text_embeddings + gamma_device * text_diff
                 logger.debug("Applied context fusion to text embeddings.")
            except Exception as cd_e: logger.error(...) ; raise cd_e
        elif self.context_decoder and visual_context is None: logger.error(...)


        # --- Score Map Calculation ---
        # ... (keep score map calculation logic) ...
        B, K, C_text = text_embeddings.shape
        visual_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        if C_vis != C_text: raise ValueError(...)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)
        logger.debug(f"Calculated score map shape: {score_map.shape}")


        # --- Feature Concatenation for Neck/Head ---
        # ... (keep feature concatenation logic) ...
        features_for_head = [feat.clone() for feat in _x_orig]
        if 0 <= self.score_concat_index < len(features_for_head):
            target_feat_map = features_for_head[self.score_concat_index]
            # ... (resizing and concatenation) ...
            try:
                score_map_resized = F.interpolate(score_map, size=target_feat_map.shape[2:], mode='bilinear', align_corners=False)
                features_for_head[self.score_concat_index] = torch.cat([target_feat_map, score_map_resized], dim=1)
                logger.info(f"Concatenated score map to feature index {self.score_concat_index}. New shape: {features_for_head[self.score_concat_index].shape}")
            except Exception as concat_e: logger.error(...) ; logger.warning("Proceeding without score map concatenation...")
        else: logger.warning(...)


        return text_embeddings, features_for_head, score_map, _x_orig


    def forward(self, img, img_metas=None, gt_semantic_seg=None, return_loss=True, **kwargs):
        """
        Main forward pass. Determines train/inference mode.
        Args:
            img (Tensor): Input images (N, C, H, W).
            img_metas (list[dict]): List of image info dicts (Can be None). Ignored in this version.
            gt_semantic_seg (Tensor): Ground truth segmentation masks (N, H, W) (for training).
            return_loss (bool): Flag indicating training mode (passed from training loop).
        """
        # 1. Extract Backbone Features
        # x should be a list/tuple of feature maps, e.g., [stage1_out, stage2_out, stage3_out, stage4_out]
        x = self.extract_feat(img)

        # 2. Process Features (Projection, Text Features, Context, Score Map, Concat)
        # features_for_head is based on original backbone features, potentially with score map concatenated at one stage
        # _x_orig contains the unmodified original backbone features
        text_embeddings, features_for_head, score_map, _x_orig = self._process_features(x)

        # 3. Process through Neck (if exists)
        # Input to neck is 'features_for_head' (list)
        if self.neck:
            logger.debug("Passing ORIGINAL backbone features (_x_orig) through neck...")
            # --- VVVVVV CHANGE HERE VVVVVV ---
            # Convert ORIGINAL backbone features list to dict for FPN input
            # **Assumption**: _x_orig order matches expected FPN input order (low->high stage)
            neck_input_dict = {str(i): feat for i, feat in enumerate(_x_orig)} # Use _x_orig
            # --- ^^^^^^ END CHANGE ^^^^^^ ---
            neck_outputs_dict = self.neck(neck_input_dict) # Neck outputs a dict

            # Process neck output (Keep this part)
            if self._neck_out_keys:
                 features_after_neck = [neck_outputs_dict[k] for k in self._neck_out_keys if k in neck_outputs_dict]
                 logger.debug(f"Neck output shapes (ordered by keys): {[f.shape for f in features_after_neck]}")
            else:
                 logger.warning("Neck output keys unknown, using dict values directly. Order might be incorrect for head.")
                 features_after_neck = list(neck_outputs_dict.values())
        else:
            # If no neck, use the features prepared by _process_features (which includes concatenated score map)
            features_after_neck = features_for_head
            logger.debug("Skipping neck. Using features from _process_features (may include score map).")


        # 4. Prepare Input for Decode Head(s)
        # Determine input for the main decode head
        # **CRITICAL**: Adapt this based on head type (FCNHead vs others)
        input_for_decode_head = None
        if isinstance(self.decode_head, FCNHead): # torchvision FCNHead takes output from FPN (dict)
             if not isinstance(features_after_neck, dict):
                  # If neck didn't produce a dict (e.g., no neck), try creating one
                  if isinstance(features_after_neck, (list,tuple)) and self._neck_out_keys and len(features_after_neck)==len(self._neck_out_keys):
                       # input_for_decode_head = {k:v for k,v in zip(self._neck_out_keys, features_after_neck)} # DON'T create dict
                       # FCNHead needs the highest resolution tensor, assume it's key '0' or first in list
                       input_for_decode_head = features_after_neck[0]
                       logger.debug("Selecting first feature tensor as input for FCNHead.")
                  else:
                       logger.error("Cannot provide valid input tensor required by FCNHead.")
                       input_for_decode_head = None # Will cause error later
             else:
                  # If neck produced a dict, select the highest resolution feature map
                  # **Assumption**: Key '0' corresponds to the highest resolution output from FPN
                  if '0' in features_after_neck:
                      input_for_decode_head = features_after_neck['0']
                      logger.debug("Selecting feature tensor with key '0' from neck output dict for FCNHead.")
                  else:
                      logger.error("Cannot find feature with key '0' in neck output dict for FCNHead.")
                      input_for_decode_head = None
             if input_for_decode_head is not None: logger.debug(f"Using Tensor input shape for FCNHead: {input_for_decode_head.shape}")

        elif isinstance(features_after_neck, (list, tuple)) and len(features_after_neck) > 0:
             # **Assumption for non-FCN Heads**: Use the highest resolution feature map
             # which is assumed to be the FIRST element in the list (verify!).
             input_for_decode_head = features_after_neck[0]
             logger.debug(f"Using first feature map for non-FCN decode head. Shape: {input_for_decode_head.shape}")
        # ... (rest of the logic for single tensor input remains the same) ...
        # --- ^^^^^^ END MODIFICATION ^^^^^^ ---


        # 5. Forward through Decode Head(s)
        if not self.decode_head:
             raise RuntimeError("Decode head is not defined.")
        if input_for_decode_head is None:
            # Handle case where input couldn't be prepared
             logger.error("Input for decode head is None. Cannot proceed.")
             if return_loss and self.training: return {'main_output': None, 'aux_losses': {}}
             else: return None

        # Main head forward pass -> logits [N, C, H', W']
        # Now input_for_decode_head should be a Tensor
        output_logits = self.decode_head(input_for_decode_head)


        # 6. Handle Training vs Inference
        if return_loss and self.training:
             if gt_semantic_seg is None:
                  raise ValueError("gt_semantic_seg is required for training (return_loss=True)")

             # --- Loss Calculation ---
             losses = {} # Dictionary to store calculated losses (if done here)

             # Resize main logits to match GT label size
             gt_h, gt_w = gt_semantic_seg.shape[-2:]
             if output_logits is not None and output_logits.shape[-2:] != (gt_h, gt_w):
                  output_logits_resized = F.interpolate(
                      output_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners
                  )
                  logger.debug(f"Resized main output logits to GT shape: {output_logits_resized.shape}")
             else:
                  output_logits_resized = output_logits # Use directly if shapes match or if None

             # NOTE: Primary loss is usually calculated in the train_worker loop using output_logits_resized.
             # We return the logits needed for that calculation.

             # --- Auxiliary Head Loss (if exists) ---
             if self.with_auxiliary_head and self.auxiliary_head and input_for_aux_head is not None:
                  aux_logits = self.auxiliary_head(input_for_aux_head)
                  if aux_logits.shape[-2:] != (gt_h, gt_w):
                       aux_logits_resized = F.interpolate(aux_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners) # Use main head alignment? Check config.
                  else:
                       aux_logits_resized = aux_logits
                  logger.debug(f"Auxiliary head output logits (resized) shape: {aux_logits_resized.shape}")
                  # Store aux logits for loss calculation in train_worker (using weights from config)
                  losses['aux_output'] = aux_logits_resized # Name it something clear

             # --- Identity Head Loss (if exists) ---
             if self.with_identity_head and self.identity_head:
                  # Identity head might take score_map or other features
                  # Assuming it takes score_map / tau here
                  id_input = score_map / self.tau
                  id_logits = self.identity_head(id_input) # Forward pass
                  if id_logits.shape[-2:] != (gt_h, gt_w):
                       id_logits_resized = F.interpolate(id_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners) # Use main head alignment?
                  else:
                       id_logits_resized = id_logits
                  logger.debug(f"Identity head output logits (resized) shape: {id_logits_resized.shape}")
                  losses['identity_output'] = id_logits_resized # Store for loss calculation

             # Return main logits and dict of auxiliary logits/losses
             return {'main_output': output_logits_resized, 'aux_losses': losses}

        else: # Inference mode (return_loss=False)
             if output_logits is None: return None # Handle head failure

             # Resize final output logits to match the original input image size for prediction
             logger.debug(f"Resizing inference output {output_logits.shape} to image shape {img.shape[2:]}")
             output = F.interpolate(
                 input=output_logits, # Use logits from main decode_head
                 size=img.shape[2:],
                 mode='bilinear',
                 align_corners=self.align_corners # Use head's align_corners setting
             )
             logger.debug(f"Final inference output shape: {output.shape}")
             return output # Return final resized logits [N, C, H_img, W_img]

    # --- Inference Helper Methods (Keep as is or adapt based on evaluation needs) ---
    def inference(self, img, img_meta, rescale):
         """Simple inference, returns logits potentially rescaled to original image size."""
         # test_cfg might control sliding window etc. - not implemented here
         seg_logit = self.forward(img, img_metas=img_meta, return_loss=False) # Call main forward in inference mode

         if seg_logit is None: return None # Handle forward pass failure

         # Rescaling logic to original image shape
         if rescale and img_meta is not None and len(img_meta) > 0 and 'ori_shape' in img_meta[0]:
              ori_shape = img_meta[0]['ori_shape'][:2] # H, W
              if seg_logit.shape[2:] != tuple(ori_shape):
                  logger.debug(f"Rescaling inference output from {seg_logit.shape[2:]} to original shape {ori_shape}")
                  seg_logit = F.interpolate(
                      seg_logit,
                      size=ori_shape,
                      mode='bilinear',
                      align_corners=self.align_corners # Use head's setting
                  )
         elif rescale:
              logger.warning("Rescale=True but ori_shape not found in img_meta. Returning logits at input image size.")

         return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image. Returns numpy prediction."""
        seg_logit = self.inference(img, img_meta, rescale)
        if seg_logit is None: return None

        seg_pred = seg_logit.argmax(dim=1) # Get class indices [N, H, W]
        seg_pred = seg_pred.cpu().numpy()
        # Assuming batch size 1 for simple_test, return the single prediction
        return seg_pred[0] if len(seg_pred) > 0 else None

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations by averaging logits. Returns numpy prediction."""
        # imgs: list of augmented images [tensor(C,H,W), ...]
        # img_metas: list of corresponding meta dicts
        seg_logits = []
        for img, meta in zip(imgs, img_metas):
             # Add batch dimension for inference call
             logit = self.inference(img.unsqueeze(0), [meta], rescale)
             if logit is not None:
                 seg_logits.append(logit)

        if not seg_logits: return None # Handle case where all inferences failed

        # Stack logits [N_aug, C, H, W] and average
        avg_seg_logit = torch.stack(seg_logits).mean(dim=0) # [C, H, W]
        seg_pred = avg_seg_logit.argmax(dim=0) # Get class indices [H, W]
        seg_pred = seg_pred.cpu().numpy()
        return seg_pred

    # forward_dummy might not be needed if FLOPs calculation uses another method
    def forward_dummy(self, img):
        """ Dummy forward for FLOPs calculation or similar. Tries to simulate main path. """
        logger.warning("forward_dummy provides a simplified path and may not accurately reflect full model complexity or handle all configurations.")
        try:
            # 1. Backbone
            x = self.extract_feat(img)
            # 2. Process (simplified - just take last feature, maybe project)
            visual_embeddings = x[-1]
            if self.vis_proj: visual_embeddings = self.vis_proj(visual_embeddings)
            features_for_head = [visual_embeddings] # Simplified input for head
            # 3. Neck (simplified - pass only last feature if neck expects list)
            if self.neck:
                neck_input = {str(len(x)-1): visual_embeddings} # Create dummy dict input
                neck_output = self.neck(neck_input)
                features_for_head = list(neck_output.values()) # Use neck output
            # 4. Head
            if self.decode_head:
                head_input = features_for_head[0] # Assume head takes first feature
                if isinstance(self.decode_head, FCNHead): # FCNHead needs dict
                     head_input = {'out': features_for_head[0]} # Provide dummy key 'out' or similar expected key
                out = self.decode_head(head_input)
                # Resize to original image size
                out = F.interpolate(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                return out
            else:
                return features_for_head # Return intermediate features if no head
        except Exception as e:
            logger.error(f"Error during forward_dummy: {e}. Returning input image shape.")
            # Return something with expected dimension characteristics if possible
            return torch.zeros(img.shape[0], self.num_classes, *img.shape[2:], device=img.device)