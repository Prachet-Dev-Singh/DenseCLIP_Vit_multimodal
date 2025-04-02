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
    TORCHMETRICS_AVAILABLE = True
    print("TorchMetrics found.")
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: TorchMetrics not found. Install (`pip install torchmetrics`) for mIoU/Accuracy calculation.")


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
def validate(model, val_loader, criterion, epoch, writer, logger, device, work_dir, num_val_classes, ignore_val_index):
    """
    Validation function with TorchMetrics and best image visualization.
    Runs only on the primary process (rank 0) in DDP.
    """
    is_primary_process = not dist.is_initialized() or dist.get_rank() == 0
    if not is_primary_process:
        # If not rank 0 in DDP, wait for rank 0 to finish validation if needed, then return
        if dist.is_initialized():
            dist.barrier() # Sync processes after validation
        return

    logger.info(f"--- Starting Validation Epoch: {epoch} ---")
    # Use model.module if wrapped in DDP
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval() # Set model to evaluation mode

    total_loss = 0.0
    num_valid_batches = 0 # Count batches that were successfully processed

    # --- Initialize Metrics (TorchMetrics) ---
    jaccard, accuracy_metric = None, None
    metrics_available_local = False # Use this local flag within the function
    if TORCHMETRICS_AVAILABLE: # Read the global flag set at script start
        try:
            # Ensure task='multiclass' for semantic segmentation
            jaccard = torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=num_val_classes,
                ignore_index=ignore_val_index
            ).to(device)
            accuracy_metric = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_val_classes,
                ignore_index=ignore_val_index,
                average='micro' # Pixel accuracy corresponds to micro average
            ).to(device)
            metrics_available_local = True # Set the LOCAL flag to True
            logger.info("TorchMetrics initialized for validation (mIoU, Pixel Accuracy).")
        except Exception as e:
            logger.error(f"Failed to initialize torchmetrics: {e}. Metrics will be skipped for this validation.")
            # TORCHMETRICS_AVAILABLE = False # <--- REMOVE THIS LINE
            # Do NOT modify the global variable here.
            # metrics_available_local remains False, which is correct.
    else:
            logger.warning("TorchMetrics not available globally. mIoU and Pixel Accuracy calculation will be skipped.")


    # --- For saving the best performing image based on accuracy ---
    best_batch_accuracy = -1.0
    best_image_data = None # Will store (image_tensor, prediction_tensor, target_tensor)

    # --- Validation Loop ---
    num_total_batches = len(val_loader)
    if num_total_batches == 0:
        logger.warning("Validation loader is empty. Skipping validation.")
        if dist.is_initialized(): dist.barrier(); # Sync processes
        return

    val_pbar = tqdm(total=num_total_batches, desc=f"Epoch {epoch} Validate", unit="batch", leave=False)
    output_vis_dir = osp.join(work_dir, "val_vis") # Directory to save visualization
    os.makedirs(output_vis_dir, exist_ok=True)

    # Use torch.no_grad context manager for the loop as well
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            # Check if batch is valid (collate_fn might return None)
            if batch_data is None:
                logger.warning(f"Skipping empty or invalid validation batch {i} (collate_fn returned None).")
                val_pbar.update(1)
                continue

            try:
                # --- Data Handling ---
                # Adapt based on your dataset's actual return format
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    images, targets = batch_data
                # Add elif for dict if your dataset returns dicts
                # elif isinstance(batch_data, dict):
                #     images, targets = batch_data.get('img'), batch_data.get('gt_semantic_seg')
                else:
                    logger.error(f"Unexpected validation batch format at index {i}: {type(batch_data)}. Skipping.")
                    val_pbar.update(1)
                    continue

                if images is None or targets is None:
                    logger.error(f"Validation batch {i} contains None image or target. Skipping.")
                    val_pbar.update(1)
                    continue

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long() # Ensure targets are Long type

                # --- Forward Pass ---
                # Assuming model returns dict in eval, or direct logits
                # Adjust based on your DenseCLIP model's eval output structure
                outputs = model_to_eval(images) # Or model_to_eval(images, return_loss=False) if needed

                # Extract logits (ensure correct key or direct output)
                # Example: assuming output is dict with 'main_output' or raw tensor
                if isinstance(outputs, dict):
                     logits = outputs.get('main_output', outputs.get('logits'))
                elif torch.is_tensor(outputs):
                     logits = outputs
                else:
                    logger.error(f"Unexpected model output type in validation batch {i}: {type(outputs)}. Skipping.")
                    val_pbar.update(1)
                    continue

                if logits is None:
                    logger.error(f"Logits are None in validation batch {i}. Skipping.")
                    val_pbar.update(1)
                    continue

                # --- Resize Logits to Match Target Dimensions ---
                # Crucial for calculating loss and metrics correctly if model output size differs from label size
                gt_h, gt_w = targets.shape[-2:]
                if logits.shape[-2:] != (gt_h, gt_w):
                    align_corners_flag = getattr(model_to_eval, 'align_corners', False) # Check if model has align_corners attr
                    logits_resized = F.interpolate(
                        logits,
                        size=(gt_h, gt_w),
                        mode='bilinear',
                        align_corners=align_corners_flag
                    )
                    # logger.debug(f"Resized logits from {logits.shape} to {logits_resized.shape} to match targets {targets.shape}")
                else:
                    logits_resized = logits

                # --- Calculate Loss ---
                loss = criterion(logits_resized, targets)

                # Check for NaN/Inf loss
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_valid_batches += 1 # Increment only if loss is valid
                else:
                    logger.warning(f"NaN or Inf loss encountered in validation batch {i}. Skipping loss accumulation for this batch.")
                    # Continue processing for metrics if possible, or skip batch entirely?
                    # Let's skip metrics update for this batch too for consistency.
                    val_pbar.update(1)
                    continue

                # --- Update Metrics (if available and loss was valid) ---
                if metrics_available_local:
                    try:
                        # Get predictions (class indices)
                        preds = torch.argmax(logits_resized, dim=1)

                        # Update metrics state
                        jaccard.update(preds, targets)
                        accuracy_metric.update(preds, targets)

                        # --- Check for Best Image in Epoch (based on current batch accuracy) ---
                        # Calculate accuracy for this specific batch
                        # Need to compute temporarily without affecting the main metric state
                        current_batch_acc_metric = torchmetrics.Accuracy(
                             task="multiclass", num_classes=num_val_classes, ignore_index=ignore_val_index, average='micro'
                        ).to(device)
                        current_batch_acc = current_batch_acc_metric(preds, targets).item() * 100 # Percentage

                        if current_batch_acc > best_batch_accuracy:
                             best_batch_accuracy = current_batch_acc
                             # Store the first image, its prediction, and target from this batch
                             # Detach and move to CPU to avoid holding GPU memory
                             best_image_data = (
                                 images[0].detach().cpu(),
                                 preds[0].detach().cpu(),
                                 targets[0].detach().cpu()
                             )
                             # logger.info(f"New best accuracy image found: {best_batch_accuracy:.2f}%")


                    except Exception as metric_update_e:
                        logger.error(f"Error updating metrics for validation batch {i}: {metric_update_e}")

            except Exception as batch_e:
                logger.error(f"Critical error processing validation batch {i}: {batch_e}", exc_info=True)
                # Decide whether to continue or stop validation on critical errors

            finally:
                # Update progress bar regardless of errors in the batch
                val_pbar.update(1)
                if num_valid_batches > 0:
                    avg_loss_so_far = total_loss / num_valid_batches
                    val_pbar.set_postfix(avg_loss=f"{avg_loss_so_far:.4f}")

    val_pbar.close()

    # --- Compute Final Metrics (after loop) ---
    avg_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    log_msg = f'--- Validation Epoch: {epoch} --- Avg Loss: {avg_loss:.4f}'
    metrics_dict_for_csv = {'epoch': epoch, 'avg_loss': f"{avg_loss:.4f}"}

    mean_iou = 0.0
    pixel_acc = 0.0

    if metrics_available_local and num_valid_batches > 0: # Ensure metrics were updated
        try:
            mean_iou = jaccard.compute().item() # Get final mIoU
            pixel_acc = accuracy_metric.compute().item() * 100 # Get final accuracy as percentage

            log_msg += f', Pixel Acc: {pixel_acc:.2f}%, Mean IoU: {mean_iou:.4f}'
            metrics_dict_for_csv.update({'pixel_accuracy': f"{pixel_acc:.2f}", 'mean_iou': f"{mean_iou:.4f}"})

            # Log to TensorBoard
            if writer:
                writer.add_scalar('val/pixel_accuracy', pixel_acc, epoch)
                writer.add_scalar('val/mean_iou', mean_iou, epoch)
                writer.add_scalar('val/epoch_loss', avg_loss, epoch) # Log average epoch loss

        except Exception as metric_compute_e:
            logger.error(f"Error computing final metrics: {metric_compute_e}")
            log_msg += ' --- (Metrics computation failed)'
            metrics_dict_for_csv.update({'pixel_accuracy': "Error", 'mean_iou': "Error"})
        finally:
            # Reset metric states for the next validation run
            jaccard.reset()
            accuracy_metric.reset()
    else:
        if num_valid_batches == 0: log_msg += ' --- (No valid batches processed)'
        else: log_msg += ' --- (Metrics skipped)'
        metrics_dict_for_csv.update({'pixel_accuracy': "N/A", 'mean_iou': "N/A"})

    logger.info(log_msg + ' ---')


    # --- Save Metrics to CSV ---
    csv_path = osp.join(work_dir, 'validation_metrics.csv')
    file_exists = osp.isfile(csv_path)
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            # Define fieldnames - ensure they match keys in metrics_dict_for_csv
            fieldnames = ['epoch', 'avg_loss', 'pixel_accuracy', 'mean_iou']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer_csv.writeheader() # Write header only if file is new
            writer_csv.writerow(metrics_dict_for_csv)
    except Exception as csv_e:
        logger.error(f"Error writing validation metrics to CSV '{csv_path}': {csv_e}")

    # --- Visualize Best Image ---
    if best_image_data:
        img_tensor, pred_tensor, target_tensor = best_image_data
        try:
            # Use the provided visualization function or a simple one
            # Assuming a function `visualize_comparison` exists
            save_path = osp.join(output_vis_dir, f"epoch{epoch}_best_acc_{best_batch_accuracy:.2f}.png")
            visualize_comparison(img_tensor, pred_tensor, target_tensor, save_path, epoch)
            logger.info(f"Saved best validation image visualization to: {save_path}")
        except Exception as vis_e:
            logger.error(f"Error during best image visualization: {vis_e}")


    # --- DDP Synchronization ---
    # Ensure all processes wait here until rank 0 is done validation
    if dist.is_initialized():
        dist.barrier()

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
def train_worker(rank, world_size, args, cfg, state_dict=None): # state_dict for FULL segmentation model passed from main
    """ The main training function executed by each process. """
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
    try:
        train_loader, val_loader, class_names_from_data, ignore_idx_from_data = build_dataloader(cfg, rank, world_size)
        num_classes = len(class_names_from_data)
        logger.info(f"Dataloaders built successfully. Num classes: {num_classes}, Ignore Index: {ignore_idx_from_data}")
    except Exception as e:
        logger.error(f"Failed to build dataloaders: {e}", exc_info=True);
        if is_ddp: cleanup(); return

    # --- Build Model ---
    try:
        model_type = cfg['model']['type']
        model_cfg = cfg['model'].copy()

        # Pop keys not directly passed to constructor or handled elsewhere
        model_cfg.pop('pretrained', None)
        model_cfg.pop('init_cfg', None)
        model_cfg.pop('train_cfg', None)
        model_cfg.pop('test_cfg', None)
        model_cfg.pop('download_dir', None)
        model_cfg.pop('type', None)

        # Extract sub-module configs and specific args
        backbone_cfg = model_cfg.pop('backbone', None)
        text_encoder_cfg = model_cfg.pop('text_encoder', None)
        decode_head_cfg = model_cfg.pop('decode_head', None)
        context_decoder_cfg = model_cfg.pop('context_decoder', None)
        neck_cfg = model_cfg.pop('neck', None)
        auxiliary_head_cfg = model_cfg.pop('auxiliary_head', None)
        identity_head_cfg = model_cfg.pop('identity_head', None)
        clip_path = model_cfg.pop('clip_pretrained', None) # Get CLIP weight path
        explicit_context_length = model_cfg.pop('context_length', 77)

        logger.info(f"Initializing model type: {model_type}")
        if model_type == "DenseCLIP":
            model = DenseCLIP(
                backbone=backbone_cfg,
                text_encoder=text_encoder_cfg,
                decode_head=decode_head_cfg,
                class_names=class_names_from_data,
                context_length=explicit_context_length,
                context_decoder=context_decoder_cfg,
                neck=neck_cfg,
                auxiliary_head=auxiliary_head_cfg,
                identity_head=identity_head_cfg,
                clip_pretrained_path=clip_path, # Pass CLIP path to constructor
                **model_cfg # Pass remaining args from model config
            )
        else:
            raise ValueError(f"Model type '{model_type}' not recognized in train_worker")

        model = model.to(device) # Move model to device BEFORE filtering params for optimizer
        if rank == 0:
             logger.info("Model built and moved to device successfully.")
             # Optional: Log model summary here if needed

    except Exception as e:
        logger.error(f"Failed to build model: {e}", exc_info=True)
        if is_ddp: cleanup(); return

    # --- Load FULL Segmentation Checkpoints (--load/load_from/state_dict) ---
    # This happens AFTER model initialization (including internal CLIP loading)
    # but BEFORE DDP wrapping and optimizer creation.
    load_path = args.load or cfg.get('load_from') # Prioritize CLI args
    if load_path:
        if osp.exists(load_path):
             logger.info(f"Attempting to load FULL model weights from --load/load_from path: {load_path}")
             try:
                 checkpoint = torch.load(load_path, map_location=device)
                 weights_key = None # Determine key for state_dict
                 if 'state_dict' in checkpoint: weights_key = 'state_dict'
                 elif 'model_state_dict' in checkpoint: weights_key = 'model_state_dict'
                 elif 'model' in checkpoint: weights_key = 'model'
                 else: weights_key = None; weights_to_load = checkpoint

                 if weights_key: weights_to_load = checkpoint[weights_key]

                 # Handle 'module.' prefix if loading DDP checkpoint into current non-DDP model
                 if all(key.startswith('module.') for key in weights_to_load):
                      logger.info("Removing 'module.' prefix from --load/load_from checkpoint.")
                      weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}

                 msg = model.load_state_dict(weights_to_load, strict=False)
                 logger.info(f"Loaded FULL model weights from: {load_path}")
                 if rank == 0: logger.info(f"Weight loading details (strict=False): {msg}")

             except Exception as e: logger.error(f"Error loading FULL model weights from {load_path}: {e}", exc_info=True)
        else: logger.warning(f"Specified --load/load_from path does not exist: {load_path}")
    elif state_dict: # Handle state_dict passed from main process (for segmentation ckpt)
         logger.info("Attempting to load pre-fetched FULL model state_dict passed from main process.")
         try:
             weights_to_load = state_dict
             if 'state_dict' in weights_to_load: weights_to_load = weights_to_load['state_dict']
             # ... (other key checks if needed) ...
             if all(key.startswith('module.') for key in weights_to_load):
                 logger.info("Removing 'module.' prefix from passed state_dict.")
                 weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}
             msg = model.load_state_dict(weights_to_load, strict=False)
             logger.info("Loaded passed pre-fetched FULL model weights.")
             if rank == 0: logger.info(f"Passed state_dict loading details (strict=False): {msg}")
         except Exception as e: logger.error(f"Error loading passed state_dict: {e}", exc_info=True)

    # --- Wrap model with DDP (if using multiple GPUs) ---
    # Load weights BEFORE wrapping with DDP
    if is_ddp:
        find_unused = cfg['training'].get('find_unused_parameters', True)
        if find_unused: logger.warning("Using find_unused_parameters=True in DDP. Can slow down training.")
        # Important: Parameter filtering for optimizer needs access to unwrapped model potentially
        # So we wrap AFTER identifying parameters to optimize
        pass # Delay DDP wrapping until after parameter filtering

    # --- Optimizer and Scheduler ---
    opt_cfg = cfg['training'].get('optimizer', {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0001}).copy()
    opt_type = opt_cfg.pop('type', 'AdamW')

    # --- VVVVVVVVVVVVVVVVVV FREEZING LOGIC START VVVVVVVVVVVVVVVVVV ---
    params_to_optimize = []
    params_frozen_count = 0
    params_trained_count = 0

    logger.info("Filtering parameters for optimizer. Freezing 'backbone' and 'text_encoder'.")
    # Iterate over the model *before* potential DDP wrapping
    for name, param in model.named_parameters():
        if name.startswith('backbone.') or name.startswith('text_encoder.'):
            param.requires_grad = False # Freeze parameters
            params_frozen_count += param.numel()
        else:
            param.requires_grad = True # Ensure others require grad
            params_to_optimize.append(param)
            params_trained_count += param.numel()

    logger.info(f"Optimizer will train {params_trained_count:,} parameters.")
    logger.info(f"Froze {params_frozen_count:,} parameters (backbone + text_encoder).")

    if not params_to_optimize:
         logger.error("No parameters found to optimize after freezing backbone and text encoder!")
         # Handle this error appropriately - maybe raise exception or exit
         if is_ddp: cleanup(); return
         else: return

    optimizer_params = params_to_optimize # Use the filtered list for the optimizer
    # --- ^^^^^^^^^^^^^^^^^^^^ FREEZING LOGIC END ^^^^^^^^^^^^^^^^^^^^ ---

    # --- NOW Wrap model with DDP (if using multiple GPUs) ---
    if is_ddp:
        # Pass the model instance that has requires_grad set correctly
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused)
        logger.info(f"Model wrapped with DistributedDataParallel on rank {rank}.")
        # dist.barrier() # Optional barrier

    # --- Build Optimizer ---
    logger.info(f"Building optimizer: {opt_type} with effective config: {opt_cfg}")
    if opt_type.lower() == 'adamw':
         # Pass the filtered list `optimizer_params`
         optimizer = torch.optim.AdamW(optimizer_params, **opt_cfg)
    elif opt_type.lower() == 'sgd':
         optimizer = torch.optim.SGD(optimizer_params, **opt_cfg)
    else: raise ValueError(f"Unsupported optimizer type: {opt_type}")


    # --- Build Scheduler ---
    sched_cfg = cfg['training'].get('scheduler', {'type': 'CosineAnnealingLR', 'T_max': cfg['training']['epochs'], 'eta_min': 1e-6}).copy()
    sched_type = sched_cfg.pop('type', 'CosineAnnealingLR')
    scheduler = None
    if sched_type.lower() == 'cosineannealinglr':
        if 'T_max' not in sched_cfg: sched_cfg['T_max'] = cfg['training']['epochs']
        logger.info(f"Building scheduler: CosineAnnealingLR with effective config: {sched_cfg}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_cfg)
    elif sched_type.lower() == 'steplr':
        logger.info(f"Building scheduler: StepLR with effective config: {sched_cfg}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sched_cfg)
    elif sched_type.lower() == 'poly':
         if 'total_iters' not in sched_cfg: sched_cfg['total_iters'] = cfg['training']['epochs']
         if 'power' not in sched_cfg: sched_cfg['power'] = 0.9
         logger.info(f"Building scheduler: PolynomialLR with effective config: {sched_cfg}")
         scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, **sched_cfg)
    else:
        logger.warning(f"Scheduler type '{sched_type}' not explicitly handled or set to None. Using constant LR.")

    # --- Loss Function ---
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx_from_data)
    logger.info(f"Loss function: CrossEntropyLoss (ignore_index={ignore_idx_from_data})")

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
                    weights_to_load = checkpoint['state_dict']
                    is_ckpt_ddp = all(key.startswith('module.') for key in weights_to_load)
                    is_model_now_ddp = isinstance(model, DDP)
                    if is_model_now_ddp and not is_ckpt_ddp:
                        weights_to_load = {'module.' + k: v for k, v in weights_to_load.items()}
                    elif not is_model_now_ddp and is_ckpt_ddp:
                        weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}
                    # --- VVVVVVVVVVVVVVVVVV MODIFICATION START VVVVVVVVVVVVVVVVVV ---
                    # Load state dict for the *entire* model. Parameters already frozen
                    # will remain frozen (requires_grad=False). Load non-strictly.
                    msg = model.load_state_dict(weights_to_load, strict=False)
                    # --- ^^^^^^^^^^^^^^^^^^^^ MODIFICATION END ^^^^^^^^^^^^^^^^^^^^ ---
                    logger.info(f"Resumed model state from checkpoint.")
                    if rank == 0: logger.info(f"Resume model load details (strict=False): {msg}")
                else: logger.warning("Resume checkpoint missing 'state_dict'.")

                # Resume Optimizer State
                if 'optimizer' in checkpoint:
                    try:
                         # --- VVVVVVVVVVVVVVVVVV MODIFICATION START VVVVVVVVVVVVVVVVVV ---
                         # Load optimizer state. Note: If the set of optimized parameters changed
                         # (e.g., freezing/unfreezing), this might raise errors or behave unexpectedly.
                         # It's usually safe if resuming with the same freezing strategy.
                         optimizer.load_state_dict(checkpoint['optimizer'])
                         logger.info("Resumed optimizer state.")
                         # --- ^^^^^^^^^^^^^^^^^^^^ MODIFICATION END ^^^^^^^^^^^^^^^^^^^^ ---
                    except Exception as opt_e: logger.warning(f"Could not load optimizer state from resume ckpt: {opt_e}", exc_info=True)
                else: logger.warning("Resume checkpoint missing 'optimizer' key.")

                # Resume Scheduler State
                if scheduler and 'scheduler' in checkpoint:
                     try:
                          scheduler.load_state_dict(checkpoint['scheduler'])
                          logger.info("Resumed scheduler state.")
                     except Exception as sch_e: logger.warning(f"Could not load scheduler state from resume ckpt: {sch_e}", exc_info=True)
                elif scheduler: logger.warning("Resume checkpoint missing 'scheduler' key.")

            except Exception as e:
                logger.error(f"Error loading resume checkpoint '{args.resume}': {e}. Starting training from epoch 0.", exc_info=True)
                start_epoch = 0
        else:
            logger.warning(f"Resume checkpoint specified but not found: {args.resume}. Starting training from epoch 0.")

    # --- Training Loop ---
    total_epochs = cfg['training']['epochs']
    eval_interval = cfg['training'].get('eval_interval', 1)
    save_interval = cfg['training'].get('save_interval', 1)
    grad_accum_steps = cfg['training'].get('grad_accum_steps', 1)
    clip_grad_norm_val = cfg['training'].get('clip_grad_norm', None) # Renamed variable
    skip_validation = args.no_validate
    aux_weights = cfg['training'].get('aux_loss_weights', {}) # Get weights for aux losses

    logger.info(f"--- Starting Training --- Total Epochs: {total_epochs}, Start Epoch: {start_epoch}")
    if rank == 0: logger.info(f"Validation {'skipped' if skip_validation else f'every {eval_interval} epochs'}.")
    if rank == 0: logger.info(f"Checkpoints saved every {save_interval} epochs to '{osp.join(effective_work_dir, 'checkpoints')}'")
    if grad_accum_steps > 1: logger.info(f"Using gradient accumulation with {grad_accum_steps} steps.")
    if clip_grad_norm_val: logger.info(f"Using gradient clipping with max_norm={clip_grad_norm_val}.")
    if aux_weights: logger.info(f"Using auxiliary loss weights: {aux_weights}")


    for epoch in range(start_epoch, total_epochs):
        model.train() # Set model to training mode
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
            # logger.debug(f"Set train sampler epoch to {epoch}") # Debug level

        epoch_loss = 0.0
        num_processed_batches = 0
        num_total_batches = len(train_loader)

        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_total_batches, desc=f"Epoch {epoch}/{total_epochs-1} Train", unit="batch")

        # --- VVVVVVVVVVVVVVVV MODIFICATION VVVVVVVVVVVVVVVV ---
        # optimizer.zero_grad() # Zero grad at the beginning ONLY if NOT accumulating
        # For accumulation, zero grad happens after optimizer.step()
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        batch_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
             if batch_data is None:
                 if pbar: pbar.update(1); continue

             data_load_time = time.time() - batch_start_time

             try:
                 # --- Data Handling ---
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: 
                     images, targets = batch_data

                 else:
                     # Handle unexpected batch format
                     logger.error(f"Unexpected train batch format at index {i}: {type(batch_data)}. Skipping.")
                     if pbar:
                         pbar.update(1)
                     continue # Skip to next iteration
                 
                 if images is None or targets is None:
                     # Handle missing data in batch
                     logger.error(f"Train batch {i} contains None image or target. Skipping.")
                     if pbar:
                         pbar.update(1)
                     continue # Skip to next iteration
                 images = images.to(device, non_blocking=True)
                 targets = targets.to(device, non_blocking=True).long()

                 # --- Forward Pass ---
                 # Use torch.set_grad_enabled(True) if parts were frozen? No, forward always enabled.
                 # DDP handles gradient synchronization automatically
                 model_output = model(images, gt_semantic_seg=targets, return_loss=True)

                 # --- Loss Calculation ---
                 main_logits = model_output.get('main_output')
                 if main_logits is None:
                     logger.error(f"Primary logits ('main_output') None in train batch {i}. Skipping.")
                     if pbar: pbar.update(1); continue

                 # Primary loss
                 loss = criterion(main_logits, targets)

                 # Auxiliary losses
                 aux_losses_dict = model_output.get('aux_losses', {})
                 total_aux_loss = torch.tensor(0.0, device=loss.device)

                 for loss_key, aux_logits in aux_losses_dict.items():
                     loss_name = loss_key.replace('_output', '')
                     weight = aux_weights.get(loss_name, 1.0) # Default weight 1.0
                     if torch.is_tensor(aux_logits):
                          aux_loss_i = criterion(aux_logits, targets) * weight
                          total_aux_loss += aux_loss_i
                          # Log individual weighted aux losses
                          if rank == 0 and writer and (i % 100 == 0): # Log every 100 batches
                              writer.add_scalar(f'train_loss/{loss_name}_weighted', aux_loss_i.item(), epoch * num_total_batches + i)
                     else: logger.warning(f"Aux output '{loss_key}' not a tensor.")

                 loss = loss + total_aux_loss # Add weighted aux losses
                 loss = loss / grad_accum_steps # Normalize loss for accumulation

                 # --- Check for NaN/Inf loss ---
                 if torch.isnan(loss) or torch.isinf(loss):
                     logger.error(f"NaN/Inf loss detected in train batch {i} (epoch {epoch}): {loss.item()}. Skipping backward.")
                     # Don't step or zero grad here, just skip backward for this micro-batch
                     if pbar: pbar.update(1); continue

                 # --- Backward Pass ---
                 # Gradients will accumulate here across micro-batches
                 loss.backward()

                 # --- Optimizer Step (with Grad Accum) ---
                 if (i + 1) % grad_accum_steps == 0:
                      # Gradient Clipping (apply before step)
                      if clip_grad_norm_val is not None:
                           model_to_clip = model.module if is_ddp else model
                           torch.nn.utils.clip_grad_norm_(
                               (p for p in model_to_clip.parameters() if p.requires_grad), # Clip only trainable params
                               max_norm=clip_grad_norm_val
                           )

                      optimizer.step() # Update weights
                      optimizer.zero_grad() # Reset gradients AFTER stepping

                 # --- Logging ---
                 batch_loss = (loss.item() * grad_accum_steps) # Log un-normalized loss
                 epoch_loss += batch_loss
                 num_processed_batches += 1

                 if rank == 0:
                      current_lr = optimizer.param_groups[0]['lr'] # Get current LR
                      if writer:
                          step = epoch * num_total_batches + i
                          writer.add_scalar('train/batch_loss', batch_loss, step)
                          writer.add_scalar('train/learning_rate', current_lr, step)

                      if pbar:
                          pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{current_lr:.6f}")

             except Exception as batch_e:
                  logger.error(f"Error processing train batch {i} (epoch {epoch}): {batch_e}", exc_info=True)
             finally:
                  if pbar: pbar.update(1)
                  batch_start_time = time.time()
        # --- End Batch Loop ---

        if pbar: pbar.close()

        # --- Epoch End ---
        # Handle case where num_processed_batches is zero (e.g., all batches failed)
        avg_epoch_loss = epoch_loss / num_processed_batches if num_processed_batches > 0 else 0.0
        current_lr_end = optimizer.param_groups[0]['lr']

        if rank == 0:
             logger.info(f"--- Epoch {epoch}/{total_epochs-1} Finished --- Avg Train Loss: {avg_epoch_loss:.4f} --- LR: {current_lr_end:.6f} ---")
             if writer:
                 writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
                 writer.add_scalar('train/epoch_lr', current_lr_end, epoch)

        # Step the scheduler
        if scheduler:
             scheduler.step()

        # --- Validation ---
        if not skip_validation and (epoch + 1) % eval_interval == 0:
             # Ensure validate function exists and handles its arguments
             try:
                 validate(
                     model, val_loader, criterion, epoch, writer, logger, device,
                     effective_work_dir, num_val_classes=num_classes,
                     ignore_val_index=ignore_idx_from_data
                 )
             except NameError: logger.error("validate function not defined!")
             except Exception as val_e: logger.error(f"Error during validation: {val_e}", exc_info=True)

        # --- Save Checkpoint ---
        if rank == 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = osp.join(effective_work_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            state_to_save = {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if scheduler: state_to_save['scheduler'] = scheduler.state_dict()
            os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
            try:
                torch.save(state_to_save, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e: logger.error(f"Failed to save checkpoint: {e}")

    # --- Final Cleanup ---
    if writer and rank == 0: writer.close(); logger.info("TensorBoard writer closed.")
    if is_ddp: cleanup(); logger.info(f"Rank {rank} destroyed process group.")
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