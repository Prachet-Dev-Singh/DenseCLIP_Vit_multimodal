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
    CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder, ViTFeatureFusionNeck, ConvBNReLU
)


# Setup logger for this module
logger = logging.getLogger(__name__)


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
    class ViTFeatureFusionNeck(nn.Module): pass
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
                 context_length, # <<< Length for FIXED text part (e.g., class name tokens)
                 # --- Arguments with Defaults ---
                 # --- VVVVV ADD depth_head config VVVVV ---
                 depth_head=None, # Config dict for depth head (Optional)
                 # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                 context_decoder=None, # Optional config dict
                 neck=None, # Optional config dict for the neck
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False, # Whether to feed text embeddings to decode head
                 tau=0.07,
                 auxiliary_head=None, # Optional config dict
                 identity_head=None, # Optional config dict
                 train_cfg=None, # Keep for potential future use
                 test_cfg=None, # Keep for potential future use
                 token_embed_dim=512, # <<< Dim for learnable context vectors if using ContextEncoder
                 text_dim=512,        # <<< Final text embedding dimension (TARGET)
                 clip_pretrained_path=None, # <<< Path to CLIP weights <<<
                 **kwargs): # Use kwargs for flexibility
        super().__init__() # Call nn.Module's init

        # --- Store basic attributes ---
        self.class_names = class_names
        self.num_classes = len(class_names)
        # Store the length used for tokenizing fixed class names
        self.fixed_text_context_length = context_length
        logger.info(f"Fixed text context length (for tokenizer): {self.fixed_text_context_length}")

        self.context_feature = context_feature
        self.score_concat_index = score_concat_index
        self.text_head = text_head
        self.tau = tau
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.align_corners = False # Default, updated by decode_head config
        # Store target text dimension, potentially updated by text_encoder config
        self.text_dim = text_dim
        logger.info(f"Target text dimension (text_dim): {self.text_dim}")


        # --- Build Backbone ---
        backbone_cfg_copy = backbone.copy()
        backbone_type = backbone_cfg_copy.pop('type')
        logger.info(f"Building backbone: {backbone_type} with config: {backbone_cfg_copy}")
        if backbone_type == "CLIPResNet":
             self.backbone = CLIPResNet(**backbone_cfg_copy)
             backbone_out_channels = backbone.get('width', 64) * 8 * 4
        elif backbone_type == "CLIPResNetWithAttention":
             self.backbone = CLIPResNetWithAttention(**backbone_cfg_copy)
             backbone_out_channels = backbone.get('output_dim', 1024)
        elif backbone_type == "CLIPVisionTransformer":
              self.backbone = CLIPVisionTransformer(**backbone_cfg_copy)
              backbone_out_channels = backbone.get('width', 768)
              config_output_dim = backbone.get('output_dim')
              if config_output_dim is not None and config_output_dim != backbone_out_channels: logger.warning(...)
        else: raise ValueError(f"Unsupported backbone type: {backbone_type}")
        logger.info(f"Built backbone. Output channels/dim: {backbone_out_channels}")


        # --- Build Text Encoder ---
        text_encoder_cfg_copy = text_encoder.copy()
        text_encoder_type = text_encoder_cfg_copy.pop('type')
        logger.info(f"Building text encoder: {text_encoder_type}...")

        # Determine and verify the final output dimension
        encoder_embed_dim = text_encoder.get('embed_dim')
        if encoder_embed_dim is None: encoder_embed_dim = self.text_dim; logger.warning(...)
        elif encoder_embed_dim != self.text_dim: logger.warning(...); self.text_dim = encoder_embed_dim
        text_encoder_cfg_copy['embed_dim'] = self.text_dim # Ensure consistency

        self.is_context_encoder = False # Flag
        if text_encoder_type == "CLIPTextEncoder":
             text_encoder_cfg_copy['context_length'] = self.fixed_text_context_length
             self.text_encoder = CLIPTextEncoder(**text_encoder_cfg_copy)
             logger.info(f"Built standard CLIPTextEncoder. Output dim: {self.text_dim}. Expects input len: {self.fixed_text_context_length}")
        elif text_encoder_type == "CLIPTextContextEncoder":
             total_encoder_len = text_encoder.get('context_length')
             if total_encoder_len is None: raise ValueError("`context_length` required in CLIPTextContextEncoder config.")
             text_encoder_cfg_copy['context_length'] = total_encoder_len
             self.text_encoder = CLIPTextContextEncoder(**text_encoder_cfg_copy)
             self.is_context_encoder = True
             logger.info(f"Built CLIPTextContextEncoder. Output dim: {self.text_dim}. Total internal capacity: {total_encoder_len}")
        else: raise ValueError(f"Unsupported text_encoder type: {text_encoder_type}")


        # --- Load Pre-trained CLIP Weights ---
        if clip_pretrained_path:
             logger.info(f"Attempting to load pre-trained CLIP weights from: {clip_pretrained_path}")
             try:
                 clip_model_jit = torch.jit.load(clip_pretrained_path, map_location="cpu")
                 clip_state_dict = clip_model_jit.state_dict()
                 logger.info(f"Loaded CLIP checkpoint with {len(clip_state_dict)} keys.")

                 # Load Visual Weights
                 visual_weights = OrderedDict(); visual_prefix = 'visual.'; count=0
                 for k,v in clip_state_dict.items(): # ... (filter logic) ...
                    if k.startswith(visual_prefix): visual_weights[k[len(visual_prefix):]] = v; count+=1
                 if visual_weights: load_msg_vis = self.backbone.load_state_dict(visual_weights, strict=False); logger.info(f"Loaded {count} visual keys. Msg: {load_msg_vis}")
                 else: logger.warning("No visual keys found.")

                 # Load Text Weights
                 text_weights = OrderedDict(); text_prefixes = ('transformer.', 'token_embedding.', 'positional_embedding', 'ln_final.', 'text_projection'); count=0
                 model_text_proj_shape = self.text_encoder.text_projection.shape if hasattr(self.text_encoder, 'text_projection') else None
                 for k, v in clip_state_dict.items():
                    if any(k.startswith(p) for p in text_prefixes):
                        if k == 'positional_embedding': # Handle pos embed length
                           loaded_len = v.shape[0]; model_len = self.text_encoder.positional_embedding.shape[0]
                           if loaded_len > model_len: text_weights[k] = v[:model_len]; logger.warning(...)
                           elif loaded_len == model_len: text_weights[k] = v
                           else: logger.warning(...) # Skip load
                        elif k == 'text_projection': # Handle projection shape
                           if model_text_proj_shape is not None and v.shape == model_text_proj_shape: text_weights[k] = v
                           else: logger.warning(...) # Skip load
                        else: text_weights[k] = v
                        count+=1
                 if text_weights: load_msg_txt = self.text_encoder.load_state_dict(text_weights, strict=False); logger.info(f"Loaded {len(text_weights)} text keys. Msg: {load_msg_txt}")
                 else: logger.warning("No text keys suitable for loading.")

                 del clip_model_jit, clip_state_dict
                 logger.info("CLIP model & state_dict deleted from memory.")
             except Exception as e: logger.error(f"Error loading CLIP weights: {e}", exc_info=True)
        else: logger.warning("No 'clip_pretrained_path' provided.")


        # --- Add Visual Projection Layers IF needed ---
        self.vis_proj = None; self.global_proj = None
        if backbone_out_channels != self.text_dim:
            logger.info(f"Backbone out dim ({backbone_out_channels}) != Text dim ({self.text_dim}). Adding projection layers...")
            self.vis_proj = nn.Conv2d(backbone_out_channels, self.text_dim, kernel_size=1)
            self.global_proj = nn.Linear(backbone_out_channels, self.text_dim)
        else: logger.info(f"Backbone/Text dims match ({self.text_dim}). No projection needed.")


        # --- Build Context Decoder ---
        self.context_decoder = None
        if context_decoder:
            context_decoder_cfg = context_decoder.copy(); cd_type = context_decoder_cfg.pop('type'); logger.info(f"Building ContextDecoder: {cd_type}...")
            if cd_type == "ContextDecoder":
                context_decoder_cfg['visual_dim'] = self.text_dim # Align expected dim
                self.context_decoder = ContextDecoder(**context_decoder_cfg)
            else: raise ValueError(...)
        else: logger.info("No context decoder configured.")


        # --- Build Neck ---
        self.neck = None
        self._neck_out_keys = None
        head_in_channels = backbone_out_channels # Start with backbone output dim

        if neck: # Check if neck config is provided from YAML
             neck_cfg_dict = neck.copy(); # Make copy of neck config
             neck_type = neck_cfg_dict.pop('type'); # Get neck type
             logger.info(f"Building neck: {neck_type}...")

             if neck_type == "ViTFeatureFusionNeck":
                  # --- VVVVV CORRECTED LOGIC VVVVV ---
                  # 1. Get 'out_indices' from the BACKBONE config
                  # Use original 'backbone' dict passed to __init__
                  backbone_out_indices = backbone.get('out_indices', [])
                  if not backbone_out_indices:
                       raise ValueError("Backbone config must specify 'out_indices' when using ViTFeatureFusionNeck.")

                  # 2. Construct in_channels_list based on BACKBONE config
                  neck_in_channels_dim = backbone.get('width', 768) # ViT width from backbone config
                  in_channels_list = [neck_in_channels_dim] * len(backbone_out_indices)
                  logger.info(f"Neck expecting {len(in_channels_list)} inputs, each with {neck_in_channels_dim} channels (derived from backbone config).")

                  # 3. Get neck-specific params (out_channels, etc.) from NECK config
                  # Use original 'neck' dict passed to __init__
                  out_channels = neck.get('out_channels')
                  inter_channels = neck.get('inter_channels') # Optional

                  # 4. Validate required neck params
                  if out_channels is None:
                      raise ValueError("Neck config for 'ViTFeatureFusionNeck' requires 'out_channels' key.")
                  if not isinstance(out_channels, int) or out_channels <= 0:
                      raise ValueError(f"Neck 'out_channels' must be a positive integer, got: {out_channels}")
                  # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

                  # 5. Import and Instantiate Neck
                  try:
                      from .models import ViTFeatureFusionNeck # Adjust import path
                  except ImportError:
                      logger.error("Cannot import ViTFeatureFusionNeck from .models")
                      raise
                  self.neck = ViTFeatureFusionNeck(
                      in_channels_list=in_channels_list, # Pass the *calculated* list
                      out_channels=out_channels,
                      inter_channels=inter_channels,
                      # Pass any remaining args from neck_cfg_dict if needed: **neck_cfg_dict
                  )
                  head_in_channels = out_channels # Update head input dim based on neck output
                  logger.info(f"Built ViTFeatureFusionNeck. Output channels for head: {head_in_channels}")

             elif neck_type == "FPN" and FeaturePyramidNetwork is not None:
                  # FPN logic remains the same (uses ResNet conventions mostly)
                  logger.info("Building standard FPN neck...")
                  default_fpn_in = [ backbone.get('width', 64) * 2**(i) * 4 for i in range(4)] # Based on ResNet width
                  in_channels_list = neck.get('in_channels', default_fpn_in) # Get from neck config or default
                  out_channels = neck.get('out_channels', 256)
                  num_outs = neck.get('num_outs', len(in_channels_list))
                  extra_blocks = None
                  if num_outs > len(in_channels_list):
                       if LastLevelMaxPool: extra_blocks = LastLevelMaxPool()
                       else: logger.warning(...)
                  if not isinstance(in_channels_list, list) or not in_channels_list: raise ValueError(f"FPN 'in_channels_list' invalid: {in_channels_list}")
                  if not isinstance(out_channels, int) or out_channels <= 0: raise ValueError(f"FPN 'out_channels' invalid: {out_channels}")
                  self.neck = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks)
                  head_in_channels = out_channels
                  self._neck_out_keys = [str(i) for i in range(num_outs)]
                  logger.info(f"Built torchvision FPN. Output channels: {head_in_channels}...")

             elif neck_type == "FPN": logger.error("Torchvision FPN specified but not available.")
             else: raise ValueError(f"Unsupported neck type: {neck_type}")
        else:
             # No neck configured path
             head_in_channels = backbone_out_channels
             logger.info(f"No neck configured. Head receives {head_in_channels} channels directly from backbone.")


        # --- Build Decode Head ---
        self.decode_head = None
        self._decode_head_cfg = None
        if decode_head:
            decode_head_cfg_copy = decode_head.copy(); decode_head_type = decode_head_cfg_copy.pop('type')
            logger.info(f"Building decode head: {decode_head_type}...")
            self.align_corners = decode_head.get('align_corners', False)
            self.num_classes = decode_head.get('num_classes', self.num_classes)
            in_channels_cfg = decode_head.get('in_channels')
            final_head_in_channels = head_in_channels
            if in_channels_cfg is not None:
                if in_channels_cfg != head_in_channels: logger.warning(f"Decode head config 'in_channels' ({in_channels_cfg}) != inferred input ({head_in_channels}). USING CONFIG VALUE.")
                final_head_in_channels = in_channels_cfg
            logger.info(f"Decode head using final input channels: {final_head_in_channels}")

            if decode_head_type == "FPNHead" and FCNHead is not None:
                 channels = decode_head.get('channels', 256)
                 self.decode_head = FCNHead(in_channels=final_head_in_channels, channels=channels)
                 self.decode_head.classifier = nn.Conv2d(channels, self.num_classes, kernel_size=1)
                 logger.info(f"Built torchvision FCNHead (classifier replaced for {self.num_classes} classes).")
            elif decode_head_type == "ViTSegmentationDecoder":
                 try: from .heads import ViTSegmentationDecoder
                 except ImportError: raise ValueError("ViTSegmentationDecoder needed but not found.")
                 encoder_channels = decode_head.get('encoder_channels'); decoder_channels = decode_head.get('decoder_channels')
                 if encoder_channels is None or decoder_channels is None: raise ValueError(...)
                 if encoder_channels != final_head_in_channels: logger.warning(...)
                 self.decode_head = ViTSegmentationDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=self.num_classes, align_corners=self.align_corners)
                 logger.info("Built ViTSegmentationDecoder.")
            elif decode_head_type == "IdentityHead" and IDENTITY_HEAD_AVAILABLE: self.decode_head = IdentityHead(**decode_head_cfg_copy); logger.info("Built IdentityHead.")
            elif decode_head_type == "IdentityHead": raise ValueError("IdentityHead not found.")
            else: raise ValueError(f"Unsupported/unavailable decode_head type: {decode_head_type}")

        self.with_decode_head = self.decode_head is not None
        if not self.with_decode_head: logger.warning("No decode head was built.")


        # --- VVVVV BUILD DEPTH HEAD VVVVV ---
        self.depth_head = None
        self.with_depth_head = False
        if depth_head: # Check if depth_head config is provided
            depth_head_cfg_copy = depth_head.copy()
            depth_head_type = depth_head_cfg_copy.pop('type')
            logger.info(f"Building depth head: {depth_head_type}...")
            # Get input channels - usually same as seg head (output of neck/backbone)
            depth_in_channels_cfg = depth_head.get('in_channels')
            final_depth_head_in_channels = head_in_channels # Use same input source as seg head
            if depth_in_channels_cfg is not None:
                 if depth_in_channels_cfg != head_in_channels: logger.warning(...)
                 final_depth_head_in_channels = depth_in_channels_cfg # Prioritize config
            logger.info(f"Depth head using final input channels: {final_depth_head_in_channels}")

            # Example: Using FCNHead structure but outputting 1 channel
            # Define a unique type name like 'FCNHeadDepth' in YAML
            if depth_head_type == "FCNHeadDepth" and FCNHead is not None:
                 channels = depth_head.get('channels', 128) # Internal channels can be different
                 self.depth_head = FCNHead(in_channels=final_depth_head_in_channels, channels=channels)
                 # IMPORTANT: Replace classifier to output 1 channel for depth
                 self.depth_head.classifier = nn.Conv2d(channels, 1, kernel_size=1)
                 logger.info(f"Built FCNHeadDepth (classifier replaced for 1 depth channel).")
                 self.with_depth_head = True
            # Add elif for other custom depth head architectures here
            # elif depth_head_type == "MyCustomDepthDecoder":
            #    self.depth_head = MyCustomDepthDecoder(**depth_head_cfg_copy)
            #    self.with_depth_head = True
            else:
                 logger.warning(f"Unsupported or unavailable depth_head type: {depth_head_type}")

        if not self.with_depth_head: logger.info("No depth head configured.")
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---


        # --- Build Auxiliary Head ---
        self.auxiliary_head = None; self.with_auxiliary_head = False
        if auxiliary_head: logger.warning("Auxiliary head configured but currently ignored/not built.")


        # --- Build Identity Head ---
        self.identity_head = None; self.with_identity_head = False
        if identity_head: # ... (logic as before) ...
            pass


        # --- Tokenization and Learnable Parameters ---
        logger.info(f"Tokenizing {len(self.class_names)} class names with fixed length {self.fixed_text_context_length}...")
        try: self.texts = torch.cat([tokenize(c, context_length=self.fixed_text_context_length) for c in self.class_names]); logger.info(f"Tokenized text shape: {self.texts.shape}")
        except NameError: logger.error("'tokenize' function not imported/defined!"); raise

        # Initialize learnable contexts/gamma ONLY if using the Context Encoder
        self.contexts = None
        self.gamma = None
        if self.is_context_encoder: # Check the flag set during encoder build
            text_encoder_total_capacity = getattr(self.text_encoder, 'context_length', 77)
            logger.info(f"CLIPTextContextEncoder total capacity: {text_encoder_total_capacity}")
            num_learnable_tokens = text_encoder_total_capacity - self.fixed_text_context_length
            if num_learnable_tokens <= 0:
                 logger.warning(f"Num_learnable_tokens ({num_learnable_tokens}) <= 0. No learnable contexts created.")
                 self.contexts = None
            else:
                 _token_embed_dim = token_embed_dim # Use dim from main args
                 logger.info(f"Initializing {num_learnable_tokens} learnable context tokens with dimension {_token_embed_dim}.")
                 # Create the parameter
                 self.contexts = nn.Parameter(torch.randn(1, num_learnable_tokens, _token_embed_dim))
                 # Initialize the parameter - Use trunc_normal_
                 # Assuming trunc_normal_ is available (e.g., from timm or manually defined)
                 # If not available, use standard normal init: nn.init.normal_(self.contexts, std=0.02)
                 try:
                    from timm.layers import trunc_normal_ # Try to import if timm used elsewhere
                    trunc_normal_(self.contexts, std=.02)
                 except ImportError:
                    logger.warning("timm.layers.trunc_normal_ not found. Using nn.init.normal_ for contexts.")
                    nn.init.normal_(self.contexts, std=0.02)
                 logger.info(f"Initialized learnable text contexts parameter: {self.contexts.shape}")

            # Gamma dimension should match the final text embedding output dimension
            _text_dim_gamma = self.text_dim
            self.gamma = nn.Parameter(torch.ones(_text_dim_gamma) * 1e-4)
            logger.info(f"Initialized learnable gamma parameter with dim {_text_dim_gamma}.")
        else:
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
        """
        Initialize weights for modules NOT loaded from CLIP.
        Applies _init_weights_fn to trainable components like projections,
        neck, heads, context decoder. Also initializes final classifiers specially.
        """
        logger.info("Applying custom weight initialization using _init_weights_fn...")
        modules_to_init = [] # List to store (name, module) tuples

        # Add projection layers if they exist
        if self.vis_proj is not None:
            modules_to_init.append(('vis_proj', self.vis_proj))
        if self.global_proj is not None:
            modules_to_init.append(('global_proj', self.global_proj))

        # Add context decoder if it exists
        if self.context_decoder is not None:
            modules_to_init.append(('context_decoder', self.context_decoder))

        # Add neck if it exists
        if self.neck is not None:
            modules_to_init.append(('neck', self.neck))

        # Add decode head if it exists
        if self.decode_head is not None:
            modules_to_init.append(('decode_head', self.decode_head))

        # --- VVVVV ADD DEPTH HEAD VVVVV ---
        if self.depth_head is not None:
            modules_to_init.append(('depth_head', self.depth_head))
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        # Add auxiliary head if it exists
        if self.auxiliary_head is not None:
            modules_to_init.append(('auxiliary_head', self.auxiliary_head))

        # Add identity head ONLY if it's a separate module instance
        if self.with_identity_head and self.identity_head is not None and self.identity_head is not self.decode_head:
            modules_to_init.append(('identity_head', self.identity_head))

        # --- Apply Initialization ---
        for name, module in modules_to_init:
             # Prevent applying to CLIP backbone/text encoder if somehow listed (safety check)
             if name in ['backbone', 'text_encoder']:
                 logger.warning(f"Skipping initialization for protected module: {name}")
                 continue

             logger.info(f"Initializing module layers: {name}...")
             # Apply the helper function recursively to all submodules
             module.apply(self._init_weights_fn)

             # --- Special handling for final classifier layers ---
             # Common practice to initialize the final layer predicting classes differently
             if name in ['decode_head', 'auxiliary_head', 'depth_head'] and hasattr(module, 'classifier'):
                 classifier_layer = module.classifier
                 if isinstance(classifier_layer, (nn.Conv2d, nn.Linear)):
                     logger.info(f"...Re-Initializing final classifier of {name} with Normal(0, 0.01)...")
                     nn.init.normal_(classifier_layer.weight, mean=0, std=0.01)
                     if classifier_layer.bias is not None:
                          nn.init.constant_(classifier_layer.bias, 0)
                 else:
                      logger.warning(f"Classifier layer in {name} is not Conv2d or Linear, skipping special init.")

        # Note: nn.Parameters like self.contexts and self.gamma are initialized
        # directly during their creation in __init__ and do not need module.apply().
        logger.info("Custom weight initialization complete.")

    # --- MODIFIED extract_feat ---
    def extract_feat(self, img):
        """
        Extract features from images using the backbone.
        Assumes backbone returns a list of tensors.
        """
        logger.debug("Extracting features with backbone...")
        features = self.backbone(img) # Call the backbone's forward method

        # --- Detailed Debug Print ---
        logger.debug(f"Raw backbone output type: {type(features)}")
        if isinstance(features, (list, tuple)):
             logger.debug(f"Raw backbone output length: {len(features)}")
             if features: # Check if not empty
                # Log shape of first element for verification
                logger.debug(f"  Element 0 type: {type(features[0])}, shape: {features[0].shape if isinstance(features[0], torch.Tensor) else 'N/A'}")
        elif isinstance(features, torch.Tensor):
             logger.debug(f"Raw backbone output shape: {features.shape}")
        # --- End Detailed Debug Print ---


        # ------
        # Check if the output is a list or tuple
        if isinstance(features, (list, tuple)):
            # Check if it's NOT empty
            if not features:
                logger.error("Backbone returned an empty list/tuple.")
                return []

            # Check if ALL elements are Tensors (more robust check)
            if not all(isinstance(f, torch.Tensor) for f in features):
                 logger.error(f"Backbone output list/tuple contains non-Tensor elements: {[type(f) for f in features]}")
                 return []

            # Optional: Check if all elements are 4D (expected for spatial features)
            if not all(f.ndim == 4 for f in features):
                 logger.warning(f"Backbone output list contains non-4D Tensors: {[f.ndim for f in features]}. Check backbone output.")
                 # Depending on neck/head, might need error here, but let's allow for now

            # If checks pass, return the full list of features
            logger.debug(f"Backbone returned a list/tuple of {len(features)} feature tensors.")
            return list(features) # Return the validated list

        elif isinstance(features, torch.Tensor) and features.ndim == 4:
             # Handle case where backbone *directly* returned a 4D tensor
             logger.warning("Backbone returned single tensor directly instead of list/tuple. Wrapping in list.")
             return [features]
        else:
             # Handle completely unexpected output formats
             logger.error(f"Unknown or unsupported backbone output format: {type(features)}. Returning empty list.")
             return []
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        
    # --- _process_features ---
    def _process_features(self, x):
        """
        Handles feature processing after backbone extraction.
        Applies projection, calculates text features, context fusion, and score map.
        Adapts for single feature map input from ViT (after extract_feat).
        Returns:
            text_embeddings (Tensor): Shape [B, K, C_text]
            features_for_head (list[Tensor]): Original backbone feature maps (before score map concat).
                                             For ViT, this will be a list with one tensor.
            score_map (Tensor): Shape [B, K, H_vis, W_vis] (using potentially projected visual features).
            _x_orig (list[Tensor]): Copy of original backbone feature maps list.
        """
        # --- Input Validation ---
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError(f"Expected _process_features input 'x' to be a non-empty list/tuple. Got: {type(x)}")

        _x_orig = [feat.clone() for feat in x] # Keep original features list (clone for safety)

        # --- Extract Global and Spatial Features ---
        # For ViT, x is likely [_ViT_spatial_map_], so x[-1] is the main spatial feature
        # For ResNet, x is [stage1, ..., stage4], x[-1] is last stage spatial feature
        visual_embeddings = x[-1] # Assume last element is the primary spatial map
        if visual_embeddings.ndim != 4:
            raise ValueError(f"Expected last backbone feature map to be 4D, got {visual_embeddings.ndim}D")

        # Calculate global feature by pooling the last spatial map
        global_feat = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).flatten(1)
        logger.debug(f"Calculated global_feat shape: {global_feat.shape}")

        B, C_vis_orig, H_vis, W_vis = visual_embeddings.shape
        C_glob_orig = global_feat.shape[1]
        if C_vis_orig != C_glob_orig:
            logger.warning(f"Initial spatial feature channels ({C_vis_orig}) != global feature channels ({C_glob_orig}). Check backbone output or pooling.")

        # --- Apply Global Projection (If layer exists) ---
        if self.global_proj is not None:
             logger.debug(f"Applying global projection: {C_glob_orig} -> {self.global_proj.out_features}")
             global_feat = self.global_proj(global_feat) # Project [B, C_orig] -> [B, C_proj]
             C_glob = global_feat.shape[1]
             logger.debug(f"Projected global_feat shape: {global_feat.shape}")
        else:
             C_glob = C_glob_orig

        # --- Apply Visual Spatial Projection (If layer exists) ---
        if self.vis_proj is not None:
             logger.debug(f"Applying visual spatial projection: {C_vis_orig} -> {self.vis_proj.out_channels}")
             visual_embeddings = self.vis_proj(visual_embeddings) # Project [B, C_vis_orig, H, W] -> [B, C_vis, H, W]
             B, C_vis, H_vis, W_vis = visual_embeddings.shape
             logger.debug(f"Projected spatial visual_embeddings shape: {visual_embeddings.shape}")
        else:
             C_vis = C_vis_orig

        # --- Check consistency AFTER projections ---
        if C_vis != C_glob:
             logger.error(f"Projected spatial dim C_vis ({C_vis}) != projected global dim C_glob ({C_glob}). Check projection layers.")

        # --- Prepare Visual Context for Context Decoder ---
        visual_context = None
        if self.context_decoder:
            if self.context_feature == 'attention':
                # Use projected global feature + projected spatial features
                global_feat_ctx = global_feat # Should have correct projected dimension now
                visual_context = torch.cat([global_feat_ctx.unsqueeze(1), visual_embeddings.flatten(2).permute(0, 2, 1)], dim=1)
                logger.debug(f"Prepared visual context ('attention') shape: {visual_context.shape}")

            elif self.context_feature == 'backbone':
                 # Use the *potentially projected* spatial features for consistency if vis_proj exists
                 visual_context_spatial = visual_embeddings # Use features AFTER vis_proj
                 C_context = visual_context_spatial.shape[1]
                 visual_context = visual_context_spatial.flatten(2).permute(0, 2, 1) # [B, H*W, C_vis]
                 # Check dimension against decoder's expected dim
                 if hasattr(self.context_decoder, 'visual_dim') and C_context != self.context_decoder.visual_dim:
                     logger.warning(f"Context feature 'backbone': Feature dim ({C_context}) != context_decoder expected dim ({self.context_decoder.visual_dim}).")
                 logger.debug(f"Prepared visual context ('backbone' using spatial feats) shape: {visual_context.shape}")

            else: raise ValueError(f"Invalid context_feature type: {self.context_feature}")


        # --- Text Feature Calculation ---
        if not hasattr(self, 'text_encoder'): raise AttributeError("text_encoder missing")
        text_embeddings_device = next(self.text_encoder.parameters()).device
        tokenized_texts = self.texts.to(text_embeddings_device)
        if isinstance(self.text_encoder, CLIPTextContextEncoder) and self.contexts is not None:
             contexts_device = self.contexts.to(text_embeddings_device)
             text_embeddings = self.text_encoder(tokenized_texts, contexts_device).expand(B, -1, -1)
        elif isinstance(self.text_encoder, CLIPTextEncoder):
             text_embeddings = self.text_encoder(tokenized_texts).expand(B, -1, -1)
        else: raise TypeError(...)
        logger.debug(f"Raw text embeddings shape: {text_embeddings.shape}")

        # Apply Context Decoder Fusion
        if self.context_decoder and visual_context is not None:
            if self.gamma is None: raise AttributeError(...)
            logger.debug(f"Applying context decoder...")
            visual_context_device = visual_context.to(text_embeddings_device)
            try: text_diff = self.context_decoder(text_embeddings, visual_context_device); gamma_device = self.gamma.to(text_embeddings_device); text_embeddings = text_embeddings + gamma_device * text_diff; logger.debug("Applied context fusion.")
            except Exception as cd_e: logger.error(...) ; raise cd_e
        elif self.context_decoder and visual_context is None: logger.error("Context decoder configured but no visual context.")


        # --- Score Map Calculation ---
        B, K, C_text = text_embeddings.shape
        visual_norm = F.normalize(visual_embeddings, dim=1, p=2) # Use potentially projected spatial features
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        if C_vis != C_text: raise ValueError(f"Visual dim after proj ({C_vis}) != Text dim ({C_text}).")
        score_map = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)
        logger.debug(f"Calculated score map shape: {score_map.shape}")


        # --- Feature Concatenation ---
        # features_for_head will be used if neck is None, otherwise _x_orig goes to neck
        # Let's return _x_orig as features_for_head consistently and handle concat later if needed
        features_for_head = _x_orig # Return original backbone features list

        if 0 <= self.score_concat_index < len(features_for_head):
            # Apply concat to the COPY we are returning
            target_feat_map = features_for_head[self.score_concat_index]
            logger.warning(f"Applying score map concatenation at index {self.score_concat_index}. Ensure neck/head handles this.")
            try:
                score_map_resized = F.interpolate(score_map, size=target_feat_map.shape[2:], mode='bilinear', align_corners=False)
                features_for_head[self.score_concat_index] = torch.cat([target_feat_map, score_map_resized], dim=1)
                logger.info(f"Concatenated score map. New shape at index {self.score_concat_index}: {features_for_head[self.score_concat_index].shape}")
            except Exception as concat_e: logger.error(...) ; logger.warning("Proceeding without concat due to error.")
        elif self.score_concat_index != -1: # Only warn if index is not explicitly disabled (-1)
            logger.warning(f"score_concat_index {self.score_concat_index} invalid. Score map not concatenated.")


        # Return: text embeddings, features for neck/head, score map, original backbone features
        return text_embeddings, features_for_head, score_map, _x_orig


    # --- forward ---
    def forward(self, img, img_metas=None, gt_semantic_seg=None, return_loss=True, **kwargs):
        """
        Main forward pass. Handles backbone, neck (optional), heads,
        and text/visual feature processing. Incorporates depth head.
        Returns outputs for segmentation and depth.

        Args:
            img (Tensor): Input images (N, C, H, W).
            img_metas (list[dict]): List of image info dicts (Ignored in this version).
            gt_semantic_seg (Tensor): Ground truth segmentation masks (N, H, W) (used for training).
            return_loss (bool): Flag indicating training mode. If True, expects gt_semantic_seg
                                or gt_depth in kwargs for resizing purposes.
            **kwargs: Catches potential extra arguments like gt_depth, gt_depth_mask
                      passed from train_worker if loss calculated externally.

        Returns:
            dict: During training (return_loss=True), returns a dict:
                  {
                      'main_output': segmentation logits [N, NumClasses, H, W] (potentially resized), or None on error,
                      'depth_output': depth prediction [N, 1, H, W] (potentially resized), or None on error,
                      'aux_losses': dictionary containing logits from auxiliary/identity heads (if any)
                  }
            dict: During inference (return_loss=False), returns a dict:
                  {
                      'seg': segmentation logits [N, NumClasses, H_img, W_img] (resized to img), or None on error,
                      'depth': depth prediction [N, 1, H_img, W_img] (resized to img), or None on error
                  }
        """
        logger.debug(f"FORWARD_START: img shape: {img.shape}, return_loss: {return_loss}, training: {self.training}")

        # 1. Extract Features from Backbone
        try:
            backbone_features = self.extract_feat(img)
            if not backbone_features: # Handle empty list return
                raise ValueError("Backbone feature extraction returned empty list.")
            logger.debug(f"FORWARD_DEBUG: Backbone features extracted. Count: {len(backbone_features)}. First feature shape: {backbone_features[0].shape if backbone_features else 'N/A'}")
        except Exception as bb_e:
            logger.error(f"Error during backbone feature extraction: {bb_e}", exc_info=True)
            if return_loss and self.training: return {'main_output': None, 'depth_output': None, 'aux_losses': {}}
            else: return {'seg': None, 'depth': None}

        _x_orig = [feat.clone() for feat in backbone_features]

        # 2. Process Features (for Score Map & Context Decoder)
        try:
            text_embeddings, _, score_map, _ = self._process_features(_x_orig)
            logger.debug(f"FORWARD_DEBUG: _process_features completed. text_embeddings shape: {text_embeddings.shape if text_embeddings is not None else 'None'}, score_map shape: {score_map.shape if score_map is not None else 'None'}")
        except Exception as proc_e:
             logger.error(f"Error during _process_features: {proc_e}", exc_info=True)
             if return_loss and self.training: return {'main_output': None, 'depth_output': None, 'aux_losses': {}}
             else: return {'seg': None, 'depth': None}

        # 3. Process Features through Neck (if exists) & Select Input for Heads
        input_for_heads = None
        if self.neck:
            try:
                logger.debug("FORWARD_DEBUG: Passing ORIGINAL backbone features (_x_orig) through neck...")
                neck_input = {str(i): feat for i, feat in enumerate(_x_orig)} if isinstance(self.neck, FeaturePyramidNetwork) else _x_orig
                features_after_neck = self.neck(neck_input)
                if not features_after_neck:
                    raise ValueError("Neck processing returned empty features.")
                elif isinstance(features_after_neck, (list, tuple)) and features_after_neck:
                     input_for_heads = features_after_neck[0]
                     logger.debug(f"FORWARD_DEBUG: Using neck output feature 0 (shape {input_for_heads.shape}) for heads.")
                elif isinstance(features_after_neck, torch.Tensor):
                     input_for_heads = features_after_neck
                     logger.debug(f"FORWARD_DEBUG: Using single tensor neck output (shape {input_for_heads.shape}) for heads.")
                else:
                     raise TypeError(f"Could not determine valid feature tensor from neck output: {type(features_after_neck)}")
            except Exception as neck_e:
                 logger.error(f"Error during neck processing: {neck_e}", exc_info=True)
        else:
            if not _x_orig:
                logger.error("FORWARD_ERROR: Original backbone features (_x_orig) are empty in 'no neck' path.")
            else:
                input_for_heads = _x_orig[-1]
                logger.debug(f"FORWARD_DEBUG: Skipping neck. Using last backbone feature (shape {input_for_heads.shape}) for heads.")

        # 4. Prepare Input for Auxiliary Head (if exists) - Assuming this part is not critical for the current error
        input_for_aux_head = None
        if self.with_auxiliary_head:
             aux_input_index = self._decode_head_cfg.get('aux_input_index', 2) if self._decode_head_cfg else 2
             if 0 <= aux_input_index < len(_x_orig):
                  input_for_aux_head = _x_orig[aux_input_index]
             # else: logger.warning(...) # Already logs this

        # 5. Forward through Decode Head(s)
        output_logits = None
        output_depth = None

        if input_for_heads is None:
             logger.error("FORWARD_ERROR: Input tensor for heads (input_for_heads) is None. Cannot proceed with head forward pass.")
        else:
            logger.debug(f"FORWARD_DEBUG: input_for_heads shape: {input_for_heads.shape}, dtype: {input_for_heads.dtype}, device: {input_for_heads.device}")
            if self.with_decode_head:
                try:
                    logger.debug("FORWARD_DEBUG: Calling segmentation decode_head...")
                    output_logits = self.decode_head(input_for_heads)
                    if output_logits is None: logger.warning("FORWARD_WARNING: Segmentation decode_head returned None.")
                    else: logger.debug(f"FORWARD_DEBUG: Segmentation head output_logits shape: {output_logits.shape}")
                except Exception as head_e:
                    logger.error(f"Error in segmentation head forward: {head_e}", exc_info=True)
                    output_logits = None
            else:
                logger.debug("FORWARD_DEBUG: No segmentation decode_head (self.with_decode_head is False).")


            if self.with_depth_head:
                try:
                    logger.debug("FORWARD_DEBUG: Calling depth_head...")
                    output_depth = self.depth_head(input_for_heads)
                    if output_depth is None: logger.warning("FORWARD_WARNING: Depth head returned None.")
                    else: logger.debug(f"FORWARD_DEBUG: Depth head output_depth shape: {output_depth.shape}")
                except Exception as depth_head_e:
                    logger.error(f"Error during depth head forward: {depth_head_e}", exc_info=True)
                    output_depth = None
            else:
                logger.debug("FORWARD_DEBUG: No depth head (self.with_depth_head is False).")

        # 6. Handle Training vs Inference Return Values
        if return_loss and self.training:
             gt_h, gt_w = -1, -1
             if gt_semantic_seg is not None:
                 gt_h, gt_w = gt_semantic_seg.shape[-2:]
                 logger.debug(f"FORWARD_DEBUG: Training mode, GT shape from seg: ({gt_h}, {gt_w})")
             elif kwargs.get('gt_depth') is not None:
                 gt_h, gt_w = kwargs['gt_depth'].shape[-2:]
                 logger.debug(f"FORWARD_DEBUG: Training mode, GT shape from depth: ({gt_h}, {gt_w})")
             elif kwargs.get('depth_targets') is not None:
                  gt_h, gt_w = kwargs['depth_targets'].shape[-2:]
                  logger.debug(f"FORWARD_DEBUG: Training mode, GT shape from depth_targets kwarg: ({gt_h}, {gt_w})")
             elif kwargs.get('seg_targets') is not None:
                  gt_h, gt_w = kwargs['seg_targets'].shape[-2:]
                  logger.debug(f"FORWARD_DEBUG: Training mode, GT shape from seg_targets kwarg: ({gt_h}, {gt_w})")


             can_resize = gt_h > 0 and gt_w > 0
             if not can_resize:
                  logger.warning("FORWARD_WARNING: Could not determine target GT shape for resizing model outputs in training.")
             losses = {}

             output_logits_resized = output_logits
             if output_logits is not None: # Check if it's not None before trying to access shape
                if can_resize and output_logits.shape[-2:] != (gt_h, gt_w):
                    try:
                        output_logits_resized = F.interpolate(output_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners)
                        logger.debug(f"FORWARD_DEBUG: Resized main logits to GT shape: {output_logits_resized.shape}")
                    except Exception as resize_e:
                        logger.error(f"Error resizing main logits: {resize_e}. Returning unresized.")
                        output_logits_resized = output_logits
             else: # output_logits was None
                logger.warning("FORWARD_WARNING: output_logits is None, cannot resize for training return.")


             output_depth_resized = output_depth
             if output_depth is not None: # Check if it's not None
                if can_resize and output_depth.shape[-2:] != (gt_h, gt_w):
                    try:
                        output_depth_resized = F.interpolate(output_depth, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners)
                        logger.debug(f"FORWARD_DEBUG: Resized depth output to GT shape: {output_depth_resized.shape}")
                    except Exception as resize_e:
                        logger.error(f"Error resizing depth output: {resize_e}. Returning unresized.")
                        output_depth_resized = output_depth
             else: # output_depth was None
                #logger.warning("FORWARD_WARNING: output_depth is None, cannot resize for training return.")
                logger.debug("FORWARD_DEBUG: output_depth is None, cannot resize for training return (expected for seg-only).") 


             # --- Auxiliary / Identity Head Logits ---
             # (Keep existing logic, add debug prints if needed)

             # ===== VVVVV FINAL DEBUG BEFORE RETURN (TRAINING) VVVVV =====
             logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_logits_resized is None? {output_logits_resized is None}")
             if output_logits_resized is not None:
                 logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_logits_resized shape: {output_logits_resized.shape}, dtype: {output_logits_resized.dtype}, device: {output_logits_resized.device}")
                 if torch.is_tensor(output_logits_resized) and output_logits_resized.numel() > 0: # Check if tensor is not empty
                     logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_logits_resized min: {output_logits_resized.min().item():.4f}, max: {output_logits_resized.max().item():.4f}, has_nan: {torch.isnan(output_logits_resized).any().item()}")

             logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_depth_resized is None? {output_depth_resized is None}")
             if output_depth_resized is not None:
                 logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_depth_resized shape: {output_depth_resized.shape}, dtype: {output_depth_resized.dtype}, device: {output_depth_resized.device}")
                 if torch.is_tensor(output_depth_resized) and output_depth_resized.numel() > 0: # Check if tensor is not empty
                     logger.debug(f"TRAIN_FWD_RETURN_DEBUG: output_depth_resized min: {output_depth_resized.min().item():.4f}, max: {output_depth_resized.max().item():.4f}, has_nan: {torch.isnan(output_depth_resized).any().item()}")
            # ===== ^^^^^ END FINAL DEBUG ^^^^^ =====

             return {
                 'main_output': output_logits_resized,
                 'depth_output': output_depth_resized,
                 'aux_losses': losses
             }
        else: # Inference mode
             final_output_seg = None; final_output_depth = None
             img_h, img_w = img.shape[2:]

             if output_logits is not None:
                  try:
                      if output_logits.shape[-2:] != (img_h, img_w):
                           final_output_seg = F.interpolate(output_logits, size=(img_h, img_w), mode='bilinear', align_corners=self.align_corners)
                      else: final_output_seg = output_logits
                      logger.debug(f"FORWARD_DEBUG: Final inference seg shape: {final_output_seg.shape}")
                  except Exception as e:
                      logger.error(f"Error resizing inference seg logits: {e}")
                      final_output_seg = output_logits # Fallback

             if output_depth is not None:
                  try:
                      if output_depth.shape[-2:] != (img_h, img_w):
                           final_output_depth = F.interpolate(output_depth, size=(img_h, img_w), mode='bilinear', align_corners=self.align_corners)
                      else: final_output_depth = output_depth
                      logger.debug(f"FORWARD_DEBUG: Final inference depth shape: {final_output_depth.shape}")
                  except Exception as e:
                      logger.error(f"Error resizing inference depth pred: {e}")
                      final_output_depth = output_depth # Fallback

             return {'seg': final_output_seg, 'depth': final_output_depth}
        


    
    def _get_final_visual_embeddings(self, x):
        """ Helper to get the spatial visual features AFTER potential projection """
        if not isinstance(x, (list, tuple)) or not x: return None # Handle bad input
        visual_embeddings = x[-1] # Assume last element is spatial features
        # Handle edge case where backbone returns (global, spatial) tuple as last element
        if isinstance(visual_embeddings, (list, tuple)) and len(visual_embeddings)==2:
            visual_embeddings = visual_embeddings[1] # Take the spatial part
        if not isinstance(visual_embeddings, torch.Tensor) or visual_embeddings.ndim != 4:
            logger.error(f"Could not extract valid 4D spatial tensor in _get_final_visual_embeddings. Got: {type(visual_embeddings)}")
            return None
        # Apply projection if it exists
        if self.vis_proj is not None:
            visual_embeddings = self.vis_proj(visual_embeddings)
        return visual_embeddings


    # --- MODIFIED Inference Helper Methods ---
    def inference(self, img, img_meta, rescale):
         """
         Performs inference, returning dict with 'seg' and 'depth' logits/predictions,
         potentially rescaled to original image size.
         """
         # Call main forward in inference mode
         outputs = self.forward(img, img_metas=img_meta, return_loss=False)
         # outputs is now {'seg': Tensor|None, 'depth': Tensor|None}

         if outputs is None: # Handle complete forward pass failure
             return {'seg': None, 'depth': None}

         seg_logit = outputs.get('seg')
         depth_pred = outputs.get('depth') # Depth output might be direct prediction or logits

         # Rescaling logic to original image shape
         if rescale and img_meta is not None and len(img_meta) > 0 and 'ori_shape' in img_meta[0]:
              ori_shape = img_meta[0]['ori_shape'][:2] # H, W
              # Rescale segmentation logits if they exist
              if seg_logit is not None and seg_logit.shape[-2:] != tuple(ori_shape):
                  try:
                      seg_logit = F.interpolate(
                          seg_logit, size=ori_shape, mode='bilinear',
                          align_corners=self.align_corners
                      )
                      logger.debug(f"Rescaled seg logits to {ori_shape}")
                  except Exception as e: logger.error(f"Error rescaling seg logits: {e}")
              # Rescale depth prediction if it exists
              if depth_pred is not None and depth_pred.shape[-2:] != tuple(ori_shape):
                  try:
                      # Use bilinear for depth resizing as well (common practice)
                      depth_pred = F.interpolate(
                          depth_pred, size=ori_shape, mode='bilinear',
                          align_corners=self.align_corners
                      )
                      logger.debug(f"Rescaled depth pred to {ori_shape}")
                  except Exception as e: logger.error(f"Error rescaling depth pred: {e}")
         elif rescale:
              logger.warning("Rescale=True but ori_shape not found in img_meta.")

         # Return dictionary containing both outputs
         return {'seg': seg_logit, 'depth': depth_pred}


    def simple_test(self, img, img_meta, rescale=True):
        """
        Simple test with single image.
        Returns dict: {'seg': numpy prediction map, 'depth': numpy prediction map}
        """
        outputs = self.inference(img, img_meta, rescale) # Returns dict {'seg':..., 'depth':...}

        seg_logit = outputs.get('seg')
        depth_pred = outputs.get('depth') # Depth output (already resized)

        seg_pred_map = None
        if seg_logit is not None:
            # Get class indices [N, H, W], convert to numpy, take first batch item
            seg_pred_map = seg_logit.argmax(dim=1).cpu().numpy()[0]

        depth_pred_map = None
        if depth_pred is not None:
             # Squeeze channel dim (if present), convert to numpy, take first batch item
             depth_pred_map = depth_pred.squeeze(1).cpu().numpy()[0] # Assumes depth output is [N,1,H,W]

        return {'seg': seg_pred_map, 'depth': depth_pred_map} # Return dict of numpy arrays


    def aug_test(self, imgs, img_metas, rescale=True):
        """
        Test with augmentations by averaging logits/predictions.
        Returns dict: {'seg': numpy prediction map, 'depth': numpy prediction map}
        """
        all_seg_logits = []
        all_depth_preds = []

        for img, meta in zip(imgs, img_metas):
             # Add batch dimension for inference call
             outputs = self.inference(img.unsqueeze(0), [meta], rescale) # Returns dict
             if outputs: # Check if inference was successful
                 if outputs.get('seg') is not None:
                     all_seg_logits.append(outputs['seg'])
                 if outputs.get('depth') is not None:
                     all_depth_preds.append(outputs['depth'])

        # --- Process Segmentation ---
        avg_seg_logit = None
        seg_pred_map = None
        if all_seg_logits:
            try:
                avg_seg_logit = torch.stack(all_seg_logits).mean(dim=0) # Average over augmentations [1, C, H, W]
                seg_pred_map = avg_seg_logit.argmax(dim=1).squeeze(0).cpu().numpy() # Get class indices [H, W]
            except Exception as e: logger.error(f"Error averaging/processing aug test seg logits: {e}")

        # --- Process Depth ---
        avg_depth_pred = None
        depth_pred_map = None
        if all_depth_preds:
            try:
                # Average depth predictions directly
                avg_depth_pred = torch.stack(all_depth_preds).mean(dim=0) # [1, 1, H, W]
                depth_pred_map = avg_depth_pred.squeeze().cpu().numpy() # Get [H, W] numpy array
            except Exception as e: logger.error(f"Error averaging/processing aug test depth preds: {e}")

        return {'seg': seg_pred_map, 'depth': depth_pred_map} # Return dict

    # --- MODIFIED forward_dummy ---
    def forward_dummy(self, img):
        """
        Dummy forward for FLOPs calculation or similar.
        Tries to simulate main path and return outputs for both heads.
        """
        logger.warning("forward_dummy provides a simplified path and may not accurately reflect full model complexity or handle all configurations.")
        output_logits = None
        output_depth = None

        try:
            # Simplified flow, mimicking the main forward path structure
            # 1. Backbone
            backbone_features = self.extract_feat(img)
            if not backbone_features: return None # Or return zeros?
            _x_orig = backbone_features # Use directly for simplicity

            # 2. Neck (if exists)
            if self.neck:
                neck_input = {str(i): feat for i, feat in enumerate(_x_orig)} if isinstance(self.neck, FeaturePyramidNetwork) else _x_orig
                features_after_neck = self.neck(neck_input)
                if isinstance(features_after_neck, (list, tuple)) and features_after_neck: input_for_heads = features_after_neck[0]
                elif isinstance(features_after_neck, torch.Tensor): input_for_heads = features_after_neck
                else: input_for_heads = _x_orig[-1] # Fallback
            else:
                input_for_heads = _x_orig[-1] # No neck case

            # 3. Heads
            if self.decode_head:
                 output_logits = self.decode_head(input_for_heads)
            if self.with_depth_head: # Use the flag here
                 output_depth = self.depth_head(input_for_heads)

            # 4. Resize (Optional, often skipped for FLOPs, but can add if needed)
            # output_logits = F.interpolate(output_logits, size=img.shape[2:], ...)
            # output_depth = F.interpolate(output_depth, size=img.shape[2:], ...)

            # Return a tuple or dict matching expected structure if possible
            # Returning just seg logits for simplicity if depth fails
            return output_logits if output_logits is not None else torch.zeros(img.shape[0], self.num_classes, *img.shape[2:], device=img.device)

        except Exception as e:
            logger.error(f"Error during forward_dummy: {e}. Returning dummy zeros.")
            # Return dummy zeros matching segmentation output shape
            return torch.zeros(img.shape[0], self.num_classes, *img.shape[2:], device=img.device)
