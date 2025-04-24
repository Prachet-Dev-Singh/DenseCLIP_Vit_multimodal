# /home/22dcs005/DenseCLIP/segmentation/train_denseclip.py

print("--- Starting train_denseclip.py ---")

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import subprocess
from tqdm import tqdm
import logging
import warnings
import csv # For CSV logging
import cv2 # For visualization


from datasets.cityscapes_depth_seg import CityscapesDepthSegDataset, DEPTH_IGNORE_VALUE
from denseclip.losses import SILogLoss

# --- Albumentations for Transforms ---
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image # Needed for interpolation modes in A.Resize/RandomScale
    ALBUMENTATIONS_AVAILABLE = True
    print("Albumentations found.")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not found. Install (`pip install albumentations opencv-python`) for data augmentation.")

# --- TorchMetrics ---
try:
    import torchmetrics
    # Try importing specific metrics needed
    from torchmetrics.regression import MeanSquaredError
    # from torchmetrics.classification import MulticlassJaccardIndex # Example if needed later
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: TorchMetrics library not found or specific metrics missing.")
    # Define placeholders as None if import fails
    MeanSquaredError = None
    # MulticlassJaccardIndex = None # Example


# --- Model and Utils Imports ---
from denseclip import (
    CLIPResNet, CLIPTextEncoder, CLIPVisionTransformer,
    CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder, DenseCLIP
)
# Make sure datasets are importable relative to this script's location
# Adjust path if needed, e.g., from ..datasets import ...
from datasets.ade20k import ADE20KSegmentation
from datasets.cityscapes import CityscapesDataset
from denseclip.utils import setup_logger, set_random_seed, collect_env_info, init_distributed

# Add current directory to path if needed for denseclip module
import sys
sys.path.append("./")

logger = logging.getLogger(__name__)

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train DenseCLIP on Cityscapes or ADE20K')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument('--resume', help='Checkpoint to resume from')
    parser.add_argument('--load', help='Checkpoint/Pretrained weights to load')
    parser.add_argument('--seed', type=int, default=None, help='Random seed. Overrides config if set.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for DDP (via mp.spawn)')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('--local_rank', type=int, default=0) # Handled by init_distributed
    parser.add_argument('--no-validate', action='store_true', help='Skip validation during training')
    # Add any other args you might need

    args = parser.parse_args()
    # Ensure local_rank is correctly set for DDP launched via torch.distributed.launch or mp.spawn
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.gpus > 1:
         # If using mp.spawn, local_rank might be passed differently or inferred later.
         # init_distributed should handle the rank assignment based on the worker function argument.
         pass
    return args

# --- Distributed Utilities ---
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Data Augmentation (using Albumentations) ---
def get_transform(cfg_data, split='train'):
    """
    Creates Albumentations transform pipelines based on config.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        warnings.warn("Albumentations unavailable, returning None transform.")
        return None

    # Default values compatible with CLIP official / OpenCLIP
    default_mean = (0.48145466, 0.4578275, 0.40821073)
    default_std = (0.26862954, 0.26130258, 0.27577711)

    # Get parameters from config, with defaults
    crop_size = cfg_data.get('crop_size', 512) # Default crop size
    mean = cfg_data.get('norm_mean', default_mean)
    std = cfg_data.get('norm_std', default_std)
    ignore_index = cfg_data.get('ignore_label', 255)

    # Handle crop_size format (int or list/tuple)
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    elif isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_h, crop_w = crop_size
    else:
        raise ValueError(f"Invalid crop_size format in config: {crop_size}. Expecting int or list/tuple of 2 ints.")

    if split == 'train':
        # Training augmentations: Random Scaling, Padding, Random Crop, Flipping, Normalization
        scale_range = cfg_data.get('scale_range', (0.5, 2.0)) # e.g., (0.5, 2.0) means 50% to 200% scaling
        # Ensure scale_limit for A.RandomScale is relative to 1.0
        scale_limit = (scale_range[0] - 1.0, scale_range[1] - 1.0)

        transform_list = [
            # Randomly resize image and mask
            A.RandomScale(scale_limit=scale_limit, interpolation=Image.BILINEAR, p=1.0),
            # Pad if needed to ensure the image is at least crop_size before random cropping
            # Use mean for image padding (often 0 is fine), mask_value for segmentation mask
            A.PadIfNeeded(min_height=crop_h, min_width=crop_w, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=ignore_index),
            # Randomly crop to the target size
            A.RandomCrop(height=crop_h, width=crop_w),
            # Random horizontal flip
            A.HorizontalFlip(p=0.5),
            # Normalize image (using CLIP's default or provided mean/std)
            A.Normalize(mean=mean, std=std),
            # Convert image and mask to PyTorch tensors
            ToTensorV2(), # Handles image (HWC->CHW) and mask (HW->HW or 1HW)
        ]
        # Optional: Add Color Jitter (apply before Normalize)
        if cfg_data.get('color_jitter', False): # Add 'color_jitter: true' to config if needed
             # Insert ColorJitter before Normalize
             normalize_idx = next(i for i, t in enumerate(transform_list) if isinstance(t, A.Normalize))
             transform_list.insert(normalize_idx, A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8))

        transform = A.Compose(transform_list)
        print(f"Train Transform: {transform}")

    else: # Validation/Test augmentations: Resize, Normalize
        transform = A.Compose([
            # Resize to the evaluation size (usually same as crop_size)
            # Use BILINEAR for image, NEAREST for mask ideally, but Albumentations Resize applies same interpolation.
            # Bilinear is often acceptable for masks if resolution isn't drastically changed.
            # If precise boundaries are critical, consider separate resize or ensure model handles minor interpolation artifacts.
            A.Resize(height=crop_h, width=crop_w, interpolation=Image.BILINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        print(f"Validation Transform: {transform}")

    return transform

# --- Dataloader Builder ---
def build_dataloader(cfg, rank=0, world_size=1):
    dataset_type = cfg['data'].get('dataset_type', 'CityscapesDataset')
    dataset_cfg = cfg['data']
    loader_cfg = cfg['training'] # Batch size, workers usually under training config

    # Get transforms using the function defined above
    train_transform = get_transform(dataset_cfg, split='train')
    val_transform = get_transform(dataset_cfg, split='val')

    # Instantiate datasets
    if dataset_type == 'CityscapesDataset':
        train_dataset = CityscapesDataset(
            root=dataset_cfg['path'],
            split='train',
            transform=train_transform, # Pass Albumentations transform
            remap_labels=True          # Ensure labels are 0-18, 255
        )
        val_dataset = CityscapesDataset(
            root=dataset_cfg['path'],
            split='val',
            transform=val_transform,   # Pass Albumentations transform
            remap_labels=True
        )
        CLASSES = CityscapesDataset.CLASSES
        IGNORE_INDEX = CityscapesDataset.IGNORE_INDEX # Should be 255
    elif dataset_type == 'ADE20KSegmentation':
        # Assuming ADE20KSegmentation is adapted similarly to CityscapesDataset
        # to accept an albumentations transform
        train_dataset = ADE20KSegmentation(
            root=dataset_cfg['path'],
            split='train',
            transform=train_transform # Pass Albumentations transform
        )
        val_dataset = ADE20KSegmentation(
            root=dataset_cfg['path'],
            split='val',
            transform=val_transform # Pass Albumentations transform
        )
        CLASSES = ADE20KSegmentation.CLASSES
        IGNORE_INDEX = ADE20KSegmentation.IGNORE_INDEX # Usually 255 or 0 depending on impl.

    elif dataset_type == 'CityscapesDepthSegDataset':
         logger.info("Using CityscapesDepthSegDataset (Seg + Depth).")
         # Get depth_max from config if provided
         depth_max_val = dataset_cfg.get('depth_max', 80.0) # Default 80m
         logger.info(f"  - Using depth_max: {depth_max_val}")
         train_dataset = CityscapesDepthSegDataset(
             root=dataset_cfg['path'],
             split='train',
             transform=train_transform,
             remap_labels=True,
             depth_max=depth_max_val
         )
         val_dataset = CityscapesDepthSegDataset(
             root=dataset_cfg['path'],
             split='val',
             transform=val_transform,
             remap_labels=True,
             depth_max=depth_max_val
          )
         # Get class names and seg ignore index from this dataset class
         CLASSES = CityscapesDepthSegDataset.CLASSES
         IGNORE_INDEX = CityscapesDepthSegDataset.SEG_IGNORE_INDEX # Use specific ignore index
    else:
        raise ValueError(f"Unknown dataset_type in config: {dataset_type}")

    # --- Distributed Samplers ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # --- Collate Function to Handle Potential Errors in __getitem__ ---
    def collate_fn_skip_none(batch):
        """ Collate function that filters out None results. """
        original_len = len(batch)
        batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
        filtered_len = len(batch)
        if original_len > filtered_len:
             # Log only once per dataloader instance perhaps, or less frequently
             # logger.warning(f"Skipped {original_len - filtered_len} invalid samples in a batch.")
             pass
        if not batch: # If the whole batch is invalid
            return None
        # Use default collate on the filtered batch
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
            # logger.error(f"Error during default_collate: {e}")
            return None # Return None if even collation fails

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg['batch_size'],
        shuffle=(train_sampler is None), # Shuffle only if not using distributed sampler
        sampler=train_sampler,
        num_workers=loader_cfg['workers'],
        pin_memory=True,
        drop_last=True, # Drop last incomplete batch for training stability
        collate_fn=collate_fn_skip_none # Use the robust collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg.get('val_batch_size', 1), # Often use batch_size 1 for validation unless memory allows more
        shuffle=False,
        sampler=val_sampler,
        num_workers=loader_cfg['workers'],
        pin_memory=True,
        drop_last=False, # Keep all validation samples
        collate_fn=collate_fn_skip_none # Use the robust collate function
    )

    logger.info(f"Built dataset '{dataset_type}'. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Ignore Index: {IGNORE_INDEX}, Num Classes: {len(CLASSES)}")

    return train_loader, val_loader, CLASSES, IGNORE_INDEX # Return ignore index too


# --- Validation Function (Enhanced with TorchMetrics and Visualization) ---
@torch.no_grad() # Ensure no gradients are computed during validation
def validate(model, val_loader, criterions, loss_weights, epoch, writer, logger, device, work_dir,
             num_seg_classes, ignore_index_seg, rank, depth_ignore_value=DEPTH_IGNORE_VALUE):
    """
    Multi-task validation function (Segmentation + Depth).
    Calculates losses (optional) and metrics for both tasks.
    Runs only on the primary process (rank 0) in DDP.

    Args:
        model: The model (potentially DDP wrapped).
        val_loader: DataLoader for the validation set (yielding image, seg_gt, depth_gt, depth_mask).
        criterions (dict): Dictionary containing loss functions, e.g., {'seg': criterion_ce, 'silog': criterion_silog}.
        loss_weights (dict): Dictionary containing weights for each loss component.
        epoch (int): Current epoch number.
        writer: TensorBoard SummaryWriter instance.
        logger: Logger instance.
        device: Target device (cuda/cpu).
        work_dir (str): Path to save logs/visualizations.
        num_seg_classes (int): Number of classes for segmentation metrics.
        ignore_index_seg (int): Ignore index for segmentation metrics.
        depth_ignore_value (float): Value indicating invalid pixels in depth GT.
    """

    print(f"--- DEBUG: ENTERING validate function for Epoch {epoch} ---")
    logger.info(f"--- Starting Validation Execution for Epoch: {epoch} ---") # Use logger too
    is_primary_process = not dist.is_initialized() or rank == 0 
    if not is_primary_process:
        if dist.is_initialized(): dist.barrier(); return # Sync non-primary processes and exit

    logger.info(f"--- Starting Validation Epoch: {epoch} ---")
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval() # Set model to evaluation mode

    # Accumulators for losses
    total_loss_combined = 0.0
    total_loss_seg = 0.0
    total_loss_depth = 0.0 # Sum of depth components (e.g., SILog + L1)
    num_valid_batches = 0 # Count batches where loss was successfully computed

    # --- Initialize Metrics ---
    val_seg_jaccard = None; val_seg_accuracy = None # Segmentation
    val_depth_rmse = None # Depth (Example: RMSE)
    metrics_available = False # Flag for whether ANY metrics could be initialized

    if TORCHMETRICS_AVAILABLE:
        try:
            logger.info("Initializing validation metrics (Seg + Depth)...")
            # Segmentation
            val_seg_jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_seg_classes, ignore_index=ignore_index_seg).to(device)
            val_seg_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_seg_classes, ignore_index=ignore_index_seg, average='micro').to(device)
            # Depth (Check if MeanSquaredError was imported)
            if MeanSquaredError is not None:
                 val_depth_rmse = MeanSquaredError(squared=False).to(device) # RMSE
                 logger.info("  - Depth RMSE metric initialized.")
            else: logger.warning("MeanSquaredError not available from TorchMetrics. Cannot calculate RMSE.")
            # (Initialize other depth metrics here)
            metrics_available = True # Set flag if at least basic setup worked
            logger.info("Validation metrics initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize validation torchmetrics: {e}. Metrics calculation skipped.", exc_info=True)
            metrics_available = False # Ensure flag is false on error
    else:
         logger.warning("TorchMetrics not available globally. Validation metrics calculation skipped.")

    # Reset metrics at the start of validation
    if metrics_available:
        logger.debug("Resetting validation metrics.")
        if val_seg_jaccard: val_seg_jaccard.reset()
        if val_seg_accuracy: val_seg_accuracy.reset()
        if val_depth_rmse: val_depth_rmse.reset()
        # (Reset other depth metrics)

    # --- For saving the best performing image ---
    best_batch_seg_accuracy = -1.0 # Based on seg accuracy
    best_image_data = None # Stores (image, seg_pred, seg_target, depth_pred, depth_target, depth_mask)

    # --- Validation Loop ---
    num_total_batches = len(val_loader)
    if num_total_batches == 0: logger.warning("Val loader empty."); return # Exit if loader empty

    val_pbar = tqdm(total=num_total_batches, desc=f"Epoch {epoch} Validate", unit="batch", leave=False, disable=(rank!=0))
    output_vis_dir = osp.join(work_dir, "val_vis"); os.makedirs(output_vis_dir, exist_ok=True)

    with torch.no_grad(): # Ensure no gradients are computed
        for i, batch_data in enumerate(val_loader):
            if batch_data is None: logger.warning(f"Skipping empty val batch {i}."); val_pbar.update(1); continue

            try:
                # --- Unpack Data (4 items) ---
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 4:
                    images, seg_targets, depth_targets, depth_masks = batch_data
                else: logger.error(f"Unexpected val batch format: {type(batch_data)}. Skipping."); val_pbar.update(1); continue
                if images is None or seg_targets is None or depth_targets is None or depth_masks is None:
                    logger.error(f"Val batch {i} has None data. Skipping."); val_pbar.update(1); continue

                # Move data, ensure shapes/types
                images = images.to(device, non_blocking=True)
                seg_targets = seg_targets.to(device, non_blocking=True).long()       # [B, H, W]
                depth_targets = depth_targets.to(device, non_blocking=True).float().unsqueeze(1) # [B, 1, H, W]
                depth_masks = depth_masks.to(device, non_blocking=True).bool().unsqueeze(1)      # [B, 1, H, W]

                # --- Forward Pass ---
                outputs = model_to_eval(images, return_loss=False) # Returns {'seg':..., 'depth':...}

                # --- Extract Outputs ---
                seg_logits = outputs.get('seg')   # [B, NumCls, H, W] or None
                depth_pred = outputs.get('depth') # [B, 1, H, W] or None
                batch_has_seg = seg_logits is not None
                batch_has_depth = depth_pred is not None
                if not batch_has_seg and not batch_has_depth: logger.error(...); val_pbar.update(1); continue

                # --- Resize Outputs (Optional but safer if model output size varies) ---
                gt_h, gt_w = seg_targets.shape[-2:] if batch_has_seg else depth_targets.shape[-2:]
                seg_logits_resized = seg_logits; depth_pred_resized = depth_pred
                align_corners_flag = getattr(model_to_eval, 'align_corners', False)
                if batch_has_seg and seg_logits.shape[-2:] != (gt_h, gt_w):
                     try: seg_logits_resized = F.interpolate(seg_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=align_corners_flag)
                     except Exception as e: logger.warning(f"Error resizing val seg logits: {e}"); # Use original on error
                if batch_has_depth and depth_pred.shape[-2:] != (gt_h, gt_w):
                     try: depth_pred_resized = F.interpolate(depth_pred, size=(gt_h, gt_w), mode='bilinear', align_corners=align_corners_flag)
                     except Exception as e: logger.warning(f"Error resizing val depth pred: {e}"); # Use original on error


                # --- Calculate Losses (Optional) ---
                batch_loss_combined = 0.0
                loss_seg = torch.tensor(0.0, device=device)
                loss_depth = torch.tensor(0.0, device=device)
                if criterions:
                    # Seg Loss
                    if batch_has_seg and 'seg' in criterions:
                         try: loss_seg = criterions['seg'](seg_logits_resized, seg_targets)
                         except Exception as e: logger.warning(f"Error calc val seg loss: {e}")
                    # Depth Loss
                    if batch_has_depth and 'silog' in criterions:
                        try: loss_depth_silog = criterions['silog'](depth_pred_resized, depth_targets, depth_masks); loss_depth += loss_depth_silog
                        except Exception as e: logger.warning(f"Error calc val depth SILog loss: {e}")
                    # (Add other depth loss calcs here)

                    batch_loss_combined = loss_weights.get('seg', 1.0)*loss_seg + loss_weights.get('silog', 1.0)*loss_depth
                    if not (torch.isnan(batch_loss_combined) or torch.isinf(batch_loss_combined)):
                        total_loss_combined += batch_loss_combined.item(); total_loss_seg += loss_seg.item(); total_loss_depth += loss_depth.item()
                        num_valid_batches += 1
                    else: logger.warning(f"NaN/Inf loss in val batch {i}.")


                # --- Update Metrics ---
                if metrics_available:
                    # Seg
                    if batch_has_seg:
                        try:
                            preds_seg = torch.argmax(seg_logits_resized.detach(), dim=1)
                            if val_seg_jaccard: val_seg_jaccard.update(preds_seg, seg_targets)
                            if val_seg_accuracy: val_seg_accuracy.update(preds_seg, seg_targets)
                        except Exception as e: logger.error(f"Error updating val seg metrics: {e}")
                    # Depth
                    if batch_has_depth:
                        try:
                             valid_mask_metric = depth_masks.squeeze(1) # B,H,W
                             valid_preds = depth_pred_resized.detach().squeeze(1)[valid_mask_metric]
                             valid_targets = depth_targets.squeeze(1)[valid_mask_metric]
                             if valid_preds.numel() > 0:
                                 if val_depth_rmse: val_depth_rmse.update(valid_preds, valid_targets)
                                 # if val_depth_absrel: val_depth_absrel.update(...)
                        except Exception as e: logger.error(f"Error updating val depth metrics: {e}")

                    # Check for Best Image (Seg Accuracy)
                    if batch_has_seg and 'preds_seg' in locals():
                        try:
                             # Re-init is inefficient but simple for batch scope
                             current_batch_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_seg_classes, ignore_index=ignore_index_seg, average='micro').to(device)
                             current_batch_acc = current_batch_acc_metric(preds_seg, seg_targets).item() * 100
                             if current_batch_acc > best_batch_seg_accuracy:
                                  best_batch_seg_accuracy = current_batch_acc
                                  best_image_data = (images[0].cpu(), preds_seg[0].cpu(), seg_targets[0].cpu(),
                                                     depth_pred_resized[0].cpu() if batch_has_depth else None,
                                                     depth_targets[0].cpu() if batch_has_depth else None,
                                                     depth_masks[0].cpu() if batch_has_depth else None)
                        except Exception as e: logger.error(f"Error checking best image: {e}")

            except Exception as batch_e: logger.error(f"Critical error in val batch {i}: {batch_e}", exc_info=True)
            finally: val_pbar.update(1);

    val_pbar.close()

    # --- Compute Final Metrics ---
    avg_loss_combined = total_loss_combined / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_loss_seg = total_loss_seg / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_loss_depth = total_loss_depth / num_valid_batches if num_valid_batches > 0 else 0.0
    log_msg = f"--- Validation Epoch: {epoch} ---"; log_msg += f"\n  Avg Loss (Combined): {avg_loss_combined:.4f} (Seg: {avg_loss_seg:.4f}, Depth: {avg_loss_depth:.4f})"
    metrics_dict_for_csv = {'epoch': epoch, 'avg_loss_comb': f"{avg_loss_combined:.4f}", 'avg_loss_seg': f"{avg_loss_seg:.4f}", 'avg_loss_depth': f"{avg_loss_depth:.4f}"}

    # Seg Metrics
    if metrics_available and val_seg_jaccard and val_seg_accuracy and num_valid_batches > 0:
        try: mean_iou_seg = val_seg_jaccard.compute().item(); pixel_acc_seg = val_seg_accuracy.compute().item() * 100; log_msg += f'\n  Seg Metrics: Pixel Acc: {pixel_acc_seg:.2f}%, Mean IoU: {mean_iou_seg:.4f}'; metrics_dict_for_csv.update({'seg_pixel_accuracy': f"{pixel_acc_seg:.2f}", 'seg_mean_iou': f"{mean_iou_seg:.4f}"});
        except Exception as e: logger.error(f"Error computing final seg metrics: {e}"); log_msg += '\n  Seg Metrics: Error'
    else: log_msg += '\n  Seg Metrics: Skipped'
    metrics_dict_for_csv.setdefault('seg_pixel_accuracy', "N/A"); metrics_dict_for_csv.setdefault('seg_mean_iou', "N/A")

    # Depth Metrics
    if metrics_available and val_depth_rmse and num_valid_batches > 0:
        try: rmse_depth = val_depth_rmse.compute().item(); log_msg += f'\n  Depth Metrics: RMSE: {rmse_depth:.4f}'; metrics_dict_for_csv.update({'depth_rmse': f"{rmse_depth:.4f}"});
        except Exception as e: logger.error(f"Error computing final depth metrics: {e}"); log_msg += '\n  Depth Metrics: Error'
    else: log_msg += '\n  Depth Metrics: Skipped'
    metrics_dict_for_csv.setdefault('depth_rmse', "N/A")

    # Log results
    logger.info(log_msg + '\n--- Validation Finished ---')
    if writer: writer.add_scalar('val/epoch_loss_combined', avg_loss_combined, epoch); writer.add_scalar('val/epoch_loss_seg', avg_loss_seg, epoch); writer.add_scalar('val/epoch_loss_depth', avg_loss_depth, epoch);
    if metrics_available: # Log computed metrics to writer
        if val_seg_jaccard and mean_iou_seg is not None: writer.add_scalar('val/seg_mean_iou', mean_iou_seg, epoch)
        if val_seg_accuracy and pixel_acc_seg is not None: writer.add_scalar('val/seg_pixel_accuracy', pixel_acc_seg, epoch)
        if val_depth_rmse and rmse_depth is not None: writer.add_scalar('val/depth_rmse', rmse_depth, epoch)


    # --- Save Metrics to CSV ---
    csv_path = osp.join(work_dir, 'validation_metrics.csv'); file_exists = osp.isfile(csv_path)
    csv_fieldnames = list(metrics_dict_for_csv.keys())
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            writer_csv = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            if not file_exists or os.path.getsize(csv_path) == 0: writer_csv.writeheader()
            writer_csv.writerow(metrics_dict_for_csv)
    except Exception as csv_e: logger.error(f"Error writing validation metrics to CSV: {csv_e}")

    # --- Visualize Best Image ---
    if best_image_data:
        img, seg_pred, seg_gt, depth_pred, depth_gt, depth_mask = best_image_data
        try: save_path = osp.join(output_vis_dir, f"epoch{epoch}_best_seg_acc_{best_batch_seg_accuracy:.2f}.png"); visualize_multi_task(img, seg_pred, seg_gt, depth_pred, depth_gt, depth_mask, save_path, epoch); logger.info(f"Saved best validation image visualization to: {save_path}")
        except NameError: logger.warning("'visualize_multi_task' not defined. Skipping visualization.")
        except Exception as vis_e: logger.error(f"Error during best image visualization: {vis_e}")

    # --- DDP Synchronization ---
    if dist.is_initialized(): dist.barrier()

# --- END OF validate FUNCTION ---

# --- Need to define or import visualize_multi_task ---
# Example placeholder:
def visualize_multi_task(img_tensor, seg_pred_tensor, seg_target_tensor,
                         depth_pred_tensor, depth_target_tensor, depth_mask_tensor,
                         save_path, epoch, mean=None, std=None, depth_ignore_value=DEPTH_IGNORE_VALUE):
     logger.warning(f"Multi-task visualization to {save_path} needs full implementation.")
     # Basic implementation to save at least segmentation:
     try:
         if mean is None: mean = (0.48145466, 0.4578275, 0.40821073)
         if std is None: std = (0.26862954, 0.26130258, 0.27577711)
         img_np = img_tensor.numpy().transpose(1, 2, 0); img_np = std * img_np + mean
         img_np = np.clip(img_np, 0, 1) * 255; img_np = img_np.astype(np.uint8)
         seg_pred_np = seg_pred_tensor.numpy().astype(np.uint8)
         seg_target_np = seg_target_tensor.numpy().astype(np.uint8)
         num_seg_classes = max(seg_pred_np.max(), seg_target_np.max()) + 1
         seg_pred_colored = cv2.applyColorMap((seg_pred_np * 255 // num_seg_classes), cv2.COLORMAP_JET)
         seg_target_colored = cv2.applyColorMap((seg_target_np * 255 // num_seg_classes), cv2.COLORMAP_JET)
         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
         axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis('off')
         axes[1].imshow(seg_target_colored); axes[1].set_title("Seg GT"); axes[1].axis('off')
         axes[2].imshow(seg_pred_colored); axes[2].set_title("Seg Pred"); axes[2].axis('off')
         plt.suptitle(f"Epoch {epoch} - Best Seg Accuracy Sample (Seg Only Viz)"); plt.tight_layout()
         plt.savefig(save_path); plt.close(fig)
     except Exception as e: logger.error(f"Basic visualization failed: {e}")
     pass

# --- Visualization Helper Function ---
def visualize_comparison(img_tensor, pred_tensor, target_tensor, save_path, epoch, mean=None, std=None):
    """ Visualizes input image, prediction, and ground truth. """
    if mean is None: mean = (0.48145466, 0.4578275, 0.40821073) # Default CLIP mean
    if std is None: std = (0.26862954, 0.26130258, 0.27577711) # Default CLIP std

    # De-normalize image tensor (assuming CHW)
    img_np = img_tensor.numpy().transpose(1, 2, 0) # CHW -> HWC
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255 # Clip, scale to 0-255
    img_np = img_np.astype(np.uint8)

    pred_np = pred_tensor.numpy() # Keep original dtype for scaling calculation
    target_np = target_tensor.numpy()

    # --- FIX: Convert to np.uint8 AFTER scaling ---
    pred_scaled_np = (pred_np * 255 / (pred_np.max() + 1e-6)).astype(np.uint8) # Add epsilon for safety
    target_scaled_np = (target_np * 255 / (target_np.max() + 1e-6)).astype(np.uint8) # Add epsilon for safety

    # Generate color maps (simple version using JET, consider Cityscapes palette if available)
    pred_colored = cv2.applyColorMap(pred_scaled_np, cv2.COLORMAP_JET)
    target_colored = cv2.applyColorMap(target_scaled_np, cv2.COLORMAP_JET)
    # --- End Fix ---

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Validation Epoch {epoch} - Best Accuracy Image", fontsize=16)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(pred_colored)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    axes[2].imshow(target_colored)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    try:
        plt.savefig(save_path)
    except Exception as e:
         logger.error(f"Failed to save visualization '{save_path}': {e}")
    plt.close(fig) # Close the figure to free memory
# --- save_checkpoint function ---
def save_checkpoint(model, optimizer, epoch, path):
    """Saves model and optimizer state."""
    # Get state dict, handling DDP wrapper
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': model_state,
        'optimizer': optimizer.state_dict(),
        # Optionally save scheduler state: 'scheduler': scheduler.state_dict()
    }
    # Ensure directory exists
    os.makedirs(osp.dirname(path), exist_ok=True)
    try:
        torch.save(state, path)
        # logger.info(f"Checkpoint saved to {path}") # Logged in train_worker
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")

# --- Main Training Worker ---
def train_worker(rank, world_size, args, cfg, state_dict=None):
    """
    The main training function executed by each process.
    Handles multi-task training for segmentation and depth estimation.
    """
    is_ddp = world_size > 1
    if is_ddp:
        init_distributed(rank, world_size) # Initialize process group

    # --- Setup Logging, Device, Work Directory ---
    effective_work_dir = args.work_dir if args.work_dir else cfg.get('training', {}).get('work_dir', f'./work_dirs/{osp.splitext(osp.basename(args.config))[0]}')
    if rank == 0: # Create dir only on rank 0
        os.makedirs(effective_work_dir, exist_ok=True)
        os.makedirs(osp.join(effective_work_dir, 'checkpoints'), exist_ok=True)
    if is_ddp: dist.barrier() # Sync after dir creation

    logger = setup_logger(effective_work_dir, rank) # Setup logger for this process
    logger.info(f'---> Process rank {rank}/{world_size} started.')
    device = torch.device('cuda', rank) if is_ddp else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info(f"Using device: {device}")

    if rank == 0:
        logger.info(f'--- Effective Work Directory: {effective_work_dir}')
        logger.info(f'--- Full Config:\n{yaml.dump(cfg)}')
        try:
            env_info_str = collect_env_info(); logger.info(f"--- Environment Info:\n{env_info_str}")
        except Exception as e: logger.warning(f"Could not collect environment info: {e}")

    # --- Set Random Seed ---
    seed = args.seed if args.seed is not None else cfg['training'].get('seed', 42)
    set_random_seed(seed + rank, deterministic=args.deterministic)
    logger.info(f"Set random seed to {seed + rank} (base seed {seed}, rank {rank})")

    # --- Build Dataloaders ---
    # Assumes build_dataloader reads cfg['data']['dataset_type'] ('CityscapesDepthSegDataset')
    # and returns class names + seg ignore index from that dataset class.
    try:
        train_loader, val_loader, class_names_from_data, ignore_idx_from_data = build_dataloader(cfg, rank, world_size)
        num_classes = len(class_names_from_data) # Num classes for segmentation
        logger.info(f"Dataloaders built successfully. Num seg classes: {num_classes}, Seg Ignore Index: {ignore_idx_from_data}")
    except Exception as e:
        logger.error(f"Failed to build dataloaders: {e}", exc_info=True);
        if is_ddp: cleanup(); return

    # --- Build Model ---
    # Assumes DenseCLIP.__init__ now accepts and builds 'depth_head' from config
    try:
        model_type = cfg['model']['type']; model_cfg = cfg['model'].copy()
        # Pop unused keys
        model_cfg.pop('pretrained', None)
        model_cfg.pop('init_cfg', None)
        model_cfg.pop('train_cfg', None)
        model_cfg.pop('test_cfg', None)
        model_cfg.pop('download_dir', None)
        model_cfg.pop('type', None)

        # Extract component configs
        backbone_cfg = model_cfg.pop('backbone', None)
        text_encoder_cfg = model_cfg.pop('text_encoder', None)
        decode_head_cfg = model_cfg.pop('decode_head', None)
        context_decoder_cfg = model_cfg.pop('context_decoder', None)
        neck_cfg = model_cfg.pop('neck', None)
        auxiliary_head_cfg = model_cfg.pop('auxiliary_head', None)
        identity_head_cfg = model_cfg.pop('identity_head', None)
        depth_head_cfg = model_cfg.pop('depth_head', None) # Extract depth head config

        # Extract specific arguments handled explicitly
        clip_path = model_cfg.pop('clip_pretrained', None)
        explicit_context_length = model_cfg.pop('context_length', None) # Get fixed text length
        explicit_text_dim = model_cfg.pop('text_dim', None) # <<< POP text_dim
        explicit_token_embed_dim = model_cfg.pop('token_embed_dim', None) # <<< POP token_embed_dim

        # Use explicit args if provided, otherwise try getting from root level of cfg['model'] as fallback
        if explicit_context_length is None: explicit_context_length = cfg['model'].get('context_length', 77)
        if explicit_text_dim is None: explicit_text_dim = cfg['model'].get('text_dim', 512)
        if explicit_token_embed_dim is None: explicit_token_embed_dim = cfg['model'].get('token_embed_dim', 512)
        logger.info(f"Initializing model type: {model_type}")
        if model_type == "DenseCLIP":
            model = DenseCLIP(
                backbone=backbone_cfg,
                text_encoder=text_encoder_cfg,
                decode_head=decode_head_cfg,
                class_names=class_names_from_data,
                context_length=explicit_context_length, # Fixed text length for tokenizer
                context_decoder=context_decoder_cfg,
                neck=neck_cfg,
                auxiliary_head=auxiliary_head_cfg,
                identity_head=identity_head_cfg,
                depth_head=depth_head_cfg, # Pass depth head config
                clip_pretrained_path=clip_path,
                # Pass other necessary args extracted from model_cfg or direct from model root config
                token_embed_dim=cfg['model'].get('token_embed_dim', 512), # Pass if needed by context encoder
                text_dim=cfg['model'].get('text_dim', 512), # Pass target text dim
                **model_cfg # Pass remaining args
            )
        else: raise ValueError(f"Model type '{model_type}' not recognized.")

        model = model.to(device)
        if rank == 0: logger.info("Model built and moved to device successfully.")
    except Exception as e: logger.error(f"Failed to build model: {e}", exc_info=True); return

    # --- Load Full Checkpoints (--load/load_from/state_dict) ---
    load_path = args.load or cfg.get('load_from')
    if load_path:
        if osp.exists(load_path):
             logger.info(f"Attempting to load FULL model state from --load/load_from path: {load_path}")
             try:
                 checkpoint = torch.load(load_path, map_location=device)
                 weights_key = 'state_dict' if 'state_dict' in checkpoint else ('model_state_dict' if 'model_state_dict' in checkpoint else ('model' if 'model' in checkpoint else None))
                 weights_to_load = checkpoint[weights_key] if weights_key else checkpoint
                 if all(key.startswith('module.') for key in weights_to_load): weights_to_load = {k.replace('module.', '', 1): v for k,v in weights_to_load.items()}
                 msg = model.load_state_dict(weights_to_load, strict=False)
                 logger.info(f"Loaded FULL model weights from: {load_path}. Load message: {msg}")
             except Exception as e: logger.error(f"Error loading FULL model weights from {load_path}: {e}", exc_info=True)
        else: logger.warning(f"Specified --load/load_from path does not exist: {load_path}")
    elif state_dict:
         logger.info("Attempting to load pre-fetched FULL model state_dict passed from main process.")
         try:
             weights_to_load = state_dict
             if 'state_dict' in weights_to_load: weights_to_load = weights_to_load['state_dict']
             if all(key.startswith('module.') for key in weights_to_load): weights_to_load = {k.replace('module.', '', 1): v for k,v in weights_to_load.items()}
             msg = model.load_state_dict(weights_to_load, strict=False)
             logger.info(f"Loaded passed pre-fetched FULL model weights. Load message: {msg}")
         except Exception as e: logger.error(f"Error loading passed state_dict: {e}", exc_info=True)

    # --- Parameter Freezing ---
    params_to_optimize = []; params_frozen_count = 0; params_trained_count = 0
    logger.info("Filtering parameters for optimizer. Freezing 'backbone' and 'text_encoder'.")
    model_to_iterate = model # Before DDP wrapping
    for name, param in model_to_iterate.named_parameters():
        if name.startswith('backbone.') or name.startswith('text_encoder.'):
            param.requires_grad = False; params_frozen_count += param.numel()
        else:
            param.requires_grad = True; params_to_optimize.append(param); params_trained_count += param.numel()
    logger.info(f"Optimizer will train {params_trained_count:,} parameters.")
    logger.info(f"Froze {params_frozen_count:,} parameters (backbone + text_encoder).")
    if not params_to_optimize: logger.error("No parameters found to optimize!"); return

    # --- DDP Wrapping ---
    if is_ddp:
        find_unused = cfg['training'].get('find_unused_parameters', True)
        if find_unused: logger.warning("Using find_unused_parameters=True in DDP...")
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused)
        logger.info(f"Model wrapped with DistributedDataParallel on rank {rank}.")

    # --- Optimizer ---
    opt_cfg = cfg['training'].get('optimizer', {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0001}).copy()
    opt_type = opt_cfg.pop('type', 'AdamW')
    optimizer_params = params_to_optimize
    logger.info(f"Building optimizer: {opt_type} with effective config: {opt_cfg}")
    if opt_type.lower() == 'adamw': optimizer = torch.optim.AdamW(optimizer_params, **opt_cfg)
    elif opt_type.lower() == 'sgd': optimizer = torch.optim.SGD(optimizer_params, **opt_cfg)
    else: raise ValueError(f"Unsupported optimizer type: {opt_type}")

    # --- Scheduler ---
    sched_cfg = cfg['training'].get('scheduler', {'type': 'CosineAnnealingLR', 'T_max': cfg['training']['epochs'], 'eta_min': 1e-6}).copy()
    sched_type = sched_cfg.pop('type', 'CosineAnnealingLR')
    scheduler = None
    try:
        if sched_type.lower() == 'cosineannealinglr':
            if 'T_max' not in sched_cfg: sched_cfg['T_max'] = cfg['training']['epochs']
            logger.info(f"Building scheduler: CosineAnnealingLR with effective config: {sched_cfg}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_cfg)
        elif sched_type.lower() == 'steplr':
            logger.info(f"Building scheduler: StepLR with effective config: {sched_cfg}")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sched_cfg)
        elif sched_type.lower() == 'poly':
             if 'total_iters' not in sched_cfg: sched_cfg['total_iters'] = cfg['training']['epochs'] # Step per epoch
             if 'power' not in sched_cfg: sched_cfg['power'] = 0.9
             logger.info(f"Building scheduler: PolynomialLR with effective config: {sched_cfg}")
             scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, **sched_cfg)
        else: logger.warning(f"Scheduler type '{sched_type}' not handled. Using constant LR.")
    except Exception as sch_e: logger.error(f"Failed to build scheduler: {sch_e}", exc_info=True); scheduler = None

    # --- Loss Functions ---
    criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx_from_data).to(device)
    logger.info(f"Segmentation Loss: CrossEntropyLoss (ignore_index={ignore_idx_from_data})")
    silog_cfg = cfg['training'].get('silog_loss', {})
    criterion_depth_silog = SILogLoss(lambd=silog_cfg.get('lambda', 0.5), eps=silog_cfg.get('eps', 1e-6)).to(device)
    logger.info(f"Depth Loss: SILogLoss (lambda={criterion_depth_silog.lambd})")
    # (Optional: criterion_depth_l1 = torch.nn.L1Loss(reduction='none').to(device))

    # --- Loss Weights ---
    default_loss_weights = {'seg': 1.0, 'silog': 0.1}
    loss_weights = cfg['training'].get('loss_weights', default_loss_weights)
    logger.info(f"Using loss weights: {loss_weights}")

    # --- TensorBoard Writer ---
    writer = None
    if rank == 0:
        tensorboard_log_dir = osp.join(effective_work_dir, 'tf_logs')
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        try: writer = SummaryWriter(log_dir=tensorboard_log_dir); logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
        except Exception as e: logger.error(f"Failed to create TensorBoard SummaryWriter: {e}")

    # --- Resume Training Logic ---
    start_epoch = 0
    if args.resume:
        if osp.exists(args.resume):
            logger.info(f"Attempting to resume from checkpoint: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint.get('epoch', -1) + 1
                logger.info(f"Resuming from epoch {start_epoch}")
                # Resume Model State
                if 'state_dict' in checkpoint:
                    weights_to_load = checkpoint['state_dict']; is_ckpt_ddp = all(k.startswith('module.') for k in weights_to_load); is_model_now_ddp = isinstance(model, DDP)
                    if is_model_now_ddp and not is_ckpt_ddp: weights_to_load = {'module.' + k: v for k, v in weights_to_load.items()}
                    elif not is_model_now_ddp and is_ckpt_ddp: weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}
                    msg = model.load_state_dict(weights_to_load, strict=False); logger.info(f"Resumed model state. Load message: {msg}")
                else: logger.warning("Resume checkpoint missing 'state_dict'.")
                # Resume Optimizer State
                if 'optimizer' in checkpoint:
                    try: optimizer.load_state_dict(checkpoint['optimizer']); logger.info("Resumed optimizer state.")
                    except Exception as opt_e: logger.warning(f"Could not load optimizer state: {opt_e}", exc_info=True)
                else: logger.warning("Resume checkpoint missing 'optimizer' key.")
                # Resume Scheduler State
                if scheduler and 'scheduler' in checkpoint:
                     try: scheduler.load_state_dict(checkpoint['scheduler']); logger.info("Resumed scheduler state.")
                     except Exception as sch_e: logger.warning(f"Could not load scheduler state: {sch_e}", exc_info=True)
                elif scheduler: logger.warning("Resume checkpoint missing 'scheduler' key.")
            except Exception as e: logger.error(f"Error loading resume ckpt: {e}. Starting fresh.", exc_info=True); start_epoch = 0
        else: logger.warning(f"Resume checkpoint not found: {args.resume}. Starting fresh.")

    # --- Initialize Training Metrics ---
    train_seg_jaccard = None; train_seg_accuracy = None # Segmentation
    train_depth_rmse = None # Depth (Example: RMSE)
    train_metrics_available = False
    if TORCHMETRICS_AVAILABLE and MeanSquaredError is not None:
        logger.info("Initializing TorchMetrics for Training Set (Seg + Depth)...")
        try:
            train_seg_jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_idx_from_data).to(device)
            train_seg_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_idx_from_data, average='micro').to(device)
            train_depth_rmse = MeanSquaredError(squared=False).to(device)
            train_metrics_available = True
            logger.info("TorchMetrics initialized for training set.")
        except Exception as e: logger.error(f"Failed to initialize training torchmetrics: {e}", exc_info=True); train_metrics_available = False
    else: logger.warning("TorchMetrics/MSE not available. Training metrics calculation skipped.")

    # --- Training Loop ---
    total_epochs = cfg['training']['epochs']
    eval_interval = cfg['training'].get('eval_interval', 1)
    save_interval = cfg['training'].get('save_interval', 1)
    grad_accum_steps = cfg['training'].get('grad_accum_steps', 1)
    clip_grad_norm_val = cfg['training'].get('clip_grad_norm', None)
    skip_validation = args.no_validate
    aux_weights = cfg['training'].get('aux_loss_weights', {})

    logger.info(f"--- Starting Training --- Total Epochs: {total_epochs}, Start Epoch: {start_epoch}")
    logger.info(f"Batch size: {cfg['training']['batch_size']}, Grad Accum: {grad_accum_steps}, Effective Batch: {cfg['training']['batch_size'] * grad_accum_steps * world_size}")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        if is_ddp: train_loader.sampler.set_epoch(epoch)

        # Accumulators for epoch stats
        epoch_loss_total, epoch_loss_seg, epoch_loss_depth = 0.0, 0.0, 0.0
        num_processed_batches = 0

        # Reset Training Metrics
        if train_metrics_available:
            if train_seg_jaccard: train_seg_jaccard.reset()
            if train_seg_accuracy: train_seg_accuracy.reset()
            if train_depth_rmse: train_depth_rmse.reset()

        pbar = None
        if rank == 0: pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{total_epochs-1} Train", unit="batch", leave=False)

        # Zero grad at start only if NOT accumulating
        if grad_accum_steps == 1: optimizer.zero_grad()

        batch_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
             if batch_data is None:
                 if pbar: pbar.update(1); continue

             try: # Unpack Data
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) == 4:
                     images, seg_targets, depth_targets, depth_masks = batch_data
                 else: logger.error(...)
                 if pbar: pbar.update(1); continue
                 if images is None or seg_targets is None or depth_targets is None or depth_masks is None: logger.error(...)
                 if pbar: pbar.update(1); continue
                 images = images.to(device, non_blocking=True)
                 seg_targets = seg_targets.to(device, non_blocking=True).long()
                 depth_targets = depth_targets.to(device, non_blocking=True).float().unsqueeze(1) # B,1,H,W
                 depth_masks = depth_masks.to(device, non_blocking=True).bool().unsqueeze(1)   # B,1,H,W
             except Exception as data_e: logger.error(f"Error unpacking batch {i}: {data_e}")
             if pbar: pbar.update(1); continue

             try: # Process batch
                 # --- Forward Pass ---
                 model_output = model(images, gt_semantic_seg=seg_targets, gt_depth=depth_targets, return_loss=True)

                 # --- Extract Outputs ---
                 main_logits = model_output.get('main_output')
                 depth_pred = model_output.get('depth_output')
                 aux_losses_dict = model_output.get('aux_losses', {})

                 # --- Calculate Combined Loss ---
                 loss_seg = torch.tensor(0.0, device=device); loss_depth = torch.tensor(0.0, device=device)
                 total_aux_loss_weighted = torch.tensor(0.0, device=device)
                 valid_batch = True

                 # Seg Loss
                 if main_logits is not None:
                     try: loss_seg = criterion_seg(main_logits, seg_targets)
                     except Exception as e: logger.warning(f"Seg loss error: {e}"); loss_seg = torch.tensor(0.0, device=device)
                 else: logger.warning("main_output None."); valid_batch = False

                 # Depth Loss(es)
                 if depth_pred is not None:
                     loss_depth_silog_val = torch.tensor(0.0, device=device) # Store value for logging
                     try: # SILog
                         # --- VVVVV CORRECTED SILog CALL (positional or keyword) VVVVV ---
                         loss_depth_silog = criterion_depth_silog(depth_pred, depth_targets, depth_masks)
                         loss_depth_silog_val = loss_depth_silog.item() # Get value before accumulation
                         loss_depth += loss_depth_silog # Accumulate depth components
                         # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                     except Exception as e: logger.error(f"Error calc SILog loss: {e}", exc_info=True)
                     # (Optional: Add other depth losses like L1, add to loss_depth)
                 else: logger.warning("depth_output None."); valid_batch = False # Consider if depth failure invalidates batch

                 # (Calculate aux losses here if applicable)

                 # Combine
                 if valid_batch:
                     total_batch_loss = (loss_weights.get('seg', 1.0) * loss_seg +
                                         loss_weights.get('silog', 1.0) * loss_depth + # Use accumulated depth loss
                                         total_aux_loss_weighted)
                     loss_for_backward = total_batch_loss / grad_accum_steps
                 else: logger.warning(f"Skipping backward for batch {i}.")
                 if pbar: pbar.update(1); continue

                 # --- Check NaN/Inf & Backward Pass ---
                 if torch.isnan(loss_for_backward) or torch.isinf(loss_for_backward): logger.error(...); continue
                 loss_for_backward.backward()

                 # --- Update Training Metrics ---
                 if train_metrics_available:
                 # Segmentation Metrics
                    if main_logits is not None:
                        try: # Add try for seg metrics
                            with torch.no_grad():
                                preds_seg = torch.argmax(main_logits.detach(), dim=1)
                                if train_seg_jaccard: train_seg_jaccard.update(preds_seg, seg_targets)
                                if train_seg_accuracy: train_seg_accuracy.update(preds_seg, seg_targets)
                        except Exception as e: logger.warning(f"Error updating train seg metrics: {e}")
                    # Depth Metrics
                    if depth_pred is not None:
                        try: # Add try for depth metrics
                            with torch.no_grad():
                                valid_mask_metric = depth_masks.squeeze(1)
                                valid_preds = depth_pred.detach().squeeze(1)[valid_mask_metric]
                                valid_targets = depth_targets.squeeze(1)[valid_mask_metric]
                                if valid_preds.numel() > 0:
                                    if train_depth_rmse: train_depth_rmse.update(valid_preds, valid_targets)
                        except Exception as e: logger.warning(f"Error updating train depth metrics: {e}")

                 # --- Optimizer Step ---
                 if (i + 1) % grad_accum_steps == 0:
                      if clip_grad_norm_val is not None: torch.nn.utils.clip_grad_norm_(...)
                      optimizer.step(); optimizer.zero_grad()

                 # --- Logging ---
                 batch_loss = total_batch_loss.item()
                 epoch_loss_total += batch_loss; epoch_loss_seg += loss_seg.item(); epoch_loss_depth += loss_depth.item() # Use accumulated depth loss value
                 num_processed_batches += 1
                 if rank == 0:
                      current_lr = optimizer.param_groups[0]['lr']
                      if writer: # ... (log batch losses to writer) ...
                          pass
                      if pbar: pbar.set_postfix(loss=f"{batch_loss:.3f}(S:{loss_seg.item():.2f},D:{loss_depth_silog_val:.3f})", lr=f"{current_lr:.6f}") # Use silog value here

             except Exception as batch_e: logger.error(f"Error processing train batch {i}: {batch_e}", exc_info=True)
             finally: 
                 if pbar: pbar.update(1); batch_start_time = time.time()
        # --- End Batch Loop ---

        if pbar: pbar.close()

        # --- Epoch End ---
        avg_epoch_loss_total = epoch_loss_total / num_processed_batches if num_processed_batches > 0 else 0.0
        avg_epoch_loss_seg = epoch_loss_seg / num_processed_batches if num_processed_batches > 0 else 0.0
        avg_epoch_loss_depth = epoch_loss_depth / num_processed_batches if num_processed_batches > 0 else 0.0
        current_lr_end = optimizer.param_groups[0]['lr']

        # --- Compute and Log Training Metrics (Both Tasks) ---
        log_msg_train = f"--- Epoch {epoch}/{total_epochs-1} Finished ---\n"
        log_msg_train += f"  Avg Train Loss (Total): {avg_epoch_loss_total:.4f} (Seg: {avg_epoch_loss_seg:.4f}, Depth: {avg_epoch_loss_depth:.4f})\n" # Use accumulated depth loss avg

        # Initialize metric values to indicate failure/unavailability
        mean_iou_train = None; pixel_acc_train = None; rmse_train = None

        if train_metrics_available and rank == 0:
             # --- Compute Seg metrics ---
             try:
                 mean_iou_train = train_seg_jaccard.compute().item()
                 pixel_acc_train = train_seg_accuracy.compute().item() * 100
                 log_msg_train += f"  Train Seg Acc: {pixel_acc_train:.2f}% | Train Seg mIoU: {mean_iou_train:.4f}\n"
             except Exception as e:
                 logger.error(f"Error computing train seg metrics: {e}", exc_info=True)
                 log_msg_train += "  Train Seg Metrics: Error\n"

             # --- Compute Depth metrics ---
             try:
                 # Check if the specific metric object exists before computing
                 if train_depth_rmse:
                     rmse_train = train_depth_rmse.compute().item()
                     log_msg_train += f"  Train Depth RMSE: {rmse_train:.4f}\n"
                 # (Compute other depth metrics here)
             except Exception as e:
                 logger.error(f"Error computing train depth metrics: {e}", exc_info=True)
                 log_msg_train += "  Train Depth Metrics: Error\n"

        log_msg_train += f"  LR: {current_lr_end:.6f}"
        if rank == 0: logger.info(log_msg_train) # Print combined log message

        # --- VVVVV Log to TensorBoard (with checks) VVVVV ---
        if rank == 0 and writer:
             try: # Use try-except for robustness
                 # Log losses
                 writer.add_scalar('train/epoch_loss_total', avg_epoch_loss_total, epoch)
                 writer.add_scalar('train/epoch_loss_seg', avg_epoch_loss_seg, epoch)
                 writer.add_scalar('train/epoch_loss_depth', avg_epoch_loss_depth, epoch)
                 writer.add_scalar('train/epoch_lr', current_lr_end, epoch)

                 # Log metrics ONLY if they were successfully computed
                 if pixel_acc_train is not None:
                     writer.add_scalar('train/pixel_accuracy', pixel_acc_train, epoch)
                 if mean_iou_train is not None:
                     writer.add_scalar('train/mean_iou', mean_iou_train, epoch)
                 if rmse_train is not None:
                     writer.add_scalar('train/depth_rmse', rmse_train, epoch)
                 # (Add checks for other depth metrics)
             except Exception as tb_e:
                  logger.error(f"Error writing epoch summary to TensorBoard: {tb_e}", exc_info=True)
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        # Step Scheduler
        if scheduler: scheduler.step()

        # --- Validation ---
        # IMPORTANT: Ensure 'validate' function is updated for multi-task
        if not skip_validation and (epoch + 1) % eval_interval == 0:
             if 'validate' in globals():
                 try:
                      logger.info("--- Starting Validation ---")
                      # --- VVVVV VERIFY THIS CALL EXACTLY VVVVV ---
                      validate(
                          model=model,
                          val_loader=val_loader,
                          criterions={'seg': criterion_seg, 'silog': criterion_depth_silog}, # Pass dict of losses
                          loss_weights=loss_weights, # Pass weights
                          epoch=epoch,
                          writer=writer,
                          logger=logger,
                          device=device,
                          work_dir=effective_work_dir,
                          num_seg_classes=num_classes, # Seg specific
                          ignore_index_seg=ignore_idx_from_data, # Seg specific
                          depth_ignore_value=DEPTH_IGNORE_VALUE, # Pass depth ignore value
                          rank=rank
                      )
                      # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                 except Exception as val_e:
                      logger.error(f"Error during validation call: {val_e}", exc_info=True)
                      # Log the Ellipsis error properly if it happens again
                      if isinstance(val_e, TypeError) and "Ellipsis" in str(val_e):
                           logger.error("Validation failed likely due to signature mismatch or internal error.")
                      # Re-raise maybe? Or just continue? For now, log and continue.
             else:
                  logger.error("'validate' function not found in scope.")

        # --- Save Checkpoint ---
        if rank == 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = osp.join(...) # ... (Save checkpoint including scheduler state) ...
            pass

    # --- Final Cleanup ---
    if writer and rank == 0: writer.close()
    if is_ddp: cleanup()
    logger.info(f"--- Training Finished on Rank {rank} ---")


# --- Weight Downloading Helper (moved inside main block for clarity) ---
def ensure_weights(url_or_path, save_dir, rank=0):
    """Downloads weights if URL, checks existence if local path. Runs only on rank 0."""
    if rank != 0: return url_or_path # Other ranks just return the original path/URL

    os.makedirs(save_dir, exist_ok=True)
    if str(url_or_path).startswith(('http:', 'https:')):
        try:
            import wget
        except ImportError:
            print(f"Rank {rank}: Error: 'wget' package not installed. Cannot download weights. Please install or provide a local path.")
            return None

        filename = url_or_path.split('/')[-1].split('?')[0] # Basic filename extraction
        save_path = osp.join(save_dir, filename)

        if not osp.exists(save_path):
            print(f"Rank {rank}: Downloading {filename} to {save_dir}...")
            try:
                wget.download(url_or_path, out=save_path)
                print(f"\nRank {rank}: Download complete! Weights saved to {save_path}")
                return save_path
            except Exception as e:
                print(f"\nRank {rank}: Error downloading {url_or_path}: {e}")
                # Attempt to clean up potentially incomplete file
                if osp.exists(save_path):
                     try: os.remove(save_path)
                     except OSError: pass
                return None
        else:
            print(f"Rank {rank}: Weights file already found at {save_path}")
            return save_path
    else:
        # It's a local path, check if it exists
        local_path = url_or_path
        # Simple check: is it absolute or relative to cwd?
        if not osp.isabs(local_path):
            # Try resolving relative to CWD (common case)
            potential_path_cwd = osp.join(os.getcwd(), local_path)
            if osp.exists(potential_path_cwd):
                 local_path = potential_path_cwd
            # Add checks relative to script dir or config dir if necessary

        if osp.exists(local_path):
            print(f"Rank {rank}: Using local weights/checkpoint path: {local_path}")
            return local_path
        else:
            print(f"Rank {rank}: Local weights/checkpoint path not found: {local_path}")
            return None

# --- Main Execution Block ---
if __name__ == '__main__':
    args = parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(f"--- Loaded configuration from: {args.config} ---")
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        sys.exit(1)

    # --- Determine and Create Work Directory ---
    # Priority: CLI arg > config > default
    if args.work_dir is None:
        config_name = osp.splitext(osp.basename(args.config))[0]
        args.work_dir = cfg.get('training', {}).get('work_dir', osp.join('work_dirs', config_name))
    # Create work_dir here on the main process before spawning workers
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(osp.join(args.work_dir, 'checkpoints'), exist_ok=True) # Ensure checkpoints subdir exists
    print(f"--- Using effective work directory: {args.work_dir} ---")

    # --- Override Config Seed with CLI Arg ---
    if args.seed is not None:
        cfg['training']['seed'] = args.seed
        print(f"--- Overriding config seed with CLI argument: {args.seed} ---")

    # --- Save Final Config ---
    # Save the potentially modified config (e.g., with overridden seed) to work_dir
    try:
        final_config_path = osp.join(args.work_dir, 'final_config.yaml')
        with open(final_config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"--- Saved final effective config to: {final_config_path} ---")
    except Exception as e:
        print(f"Warning: Could not save final config: {e}")


    # --- Pre-download/Load Weights on Rank 0 ---
    # Determine the path/URL for pretrained weights
    # Priority: CLI --load > config 'load_from' > config 'model.pretrained'
    load_source = args.load
    if not load_source: load_source = cfg.get('load_from')
    if not load_source: load_source = cfg.get('model', {}).get('pretrained')

    state_dict_to_pass = None
    checked_load_path = None # The actual path after download/check

    if load_source:
        print(f"--- Attempting to ensure weights '{load_source}' are available (on Rank 0 if DDP) ---")
        # Only Rank 0 should download/check the path initially
        save_dir = cfg.get('model', {}).get('download_dir', osp.join(args.work_dir, 'pretrained')) # Save in work_dir/pretrained by default
        checked_load_path = ensure_weights(load_source, save_dir, rank=args.local_rank)

        if checked_load_path and args.local_rank == 0: # Load state dict only on rank 0
             print(f"--- Pre-loading state dict from: {checked_load_path} on Rank 0 ---")
             try:
                 # Load to CPU first to avoid GPU memory issues if model is large
                 state_dict_to_pass = torch.load(checked_load_path, map_location='cpu')
                 print(f"--- State dict loaded successfully on Rank 0 ---")
             except Exception as e:
                 print(f"Error pre-loading state dict from {checked_load_path} on Rank 0: {e}")
                 state_dict_to_pass = None
                 checked_load_path = None # Mark as failed
        elif args.local_rank == 0:
             print(f"--- Failed to ensure weights are available. Pre-loading skipped. ---")


    # --- Launch Training ---
    world_size = args.gpus
    if world_size > 1:
        # Use torch.multiprocessing.spawn for DDP
        print(f"--- Spawning {world_size} processes for Distributed Data Parallel training ---")
        # Pass the preloaded state_dict (or None) to all workers
        mp.spawn(train_worker,
                 args=(world_size, args, cfg, state_dict_to_pass),
                 nprocs=world_size,
                 join=True)
    else:
        # Run directly in the main process if gpus=1
        print("--- Running training in single process mode (no DDP) ---")
        train_worker(0, 1, args, cfg, state_dict_to_pass)

    print("--- train_denseclip.py finished ---")