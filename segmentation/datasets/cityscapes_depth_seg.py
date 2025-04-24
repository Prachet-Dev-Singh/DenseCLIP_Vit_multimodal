# segmentation/datasets/cityscapes_depth_seg.py

import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import warnings

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
# Baseline * Focal_Length (B*f) ~ 0.22m * ~2260 pixels (for 1024x2048) = ~500 pixel*meter
# This value is approximate and might need refinement based on exact camera parameters used
# or can be treated as a scaling factor if absolute metric depth is not critical.
BASELINE_FOCAL_LENGTH = 500.0
DEPTH_IGNORE_VALUE = 0.0      # Use 0.0 depth to indicate invalid pixels
DISPARITY_SCALE = 256.0       # From Cityscapes documentation for disparity map scaling
DISPARITY_OFFSET = 1.0        # From Cityscapes documentation
MIN_DISPARITY_SCALED = 1e-3   # Minimum scaled disparity to consider for depth calculation (avoids division by near-zero)

class CityscapesDepthSegDataset(Dataset):
    """
    Cityscapes dataset for combined semantic segmentation and depth estimation.
    Loads images (_leftImg8bit.png),
    segmentation labels (_gtFine_labelIds.png - remapped),
    and disparity maps (_disparity.png - converted to depth).

    Args:
        root (str): Root directory of the Cityscapes dataset.
        split (str): Dataset split, 'train', 'val', or 'test'.
        transform (callable, optional): A function/transform from Albumentations
            that takes in image and masks (list) and returns a transformed version.
            Normalization and tensor conversion should be part of the transform.
        remap_labels (bool): If True, remap segmentation labels to train IDs (0-18, 255).
        depth_max (float): Maximum depth value (in meters). Values above this
                           or calculated from invalid disparity will be ignored.
    """
    # Standard Cityscapes 19 classes mapping
    ID_TO_TRAIN_ID = np.array([255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4,
                            255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            255, 255, 16, 17, 18], dtype=np.uint8)

    CLASSES = [ # Segmentation classes (19 classes)
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]
    SEG_IGNORE_INDEX = 255 # Ignore index for segmentation task

    _printed_info = {} # Use dict to track printing per split

    def __init__(self,
                 root,
                 split='train',
                 transform=None, # Expects albumentations transform pipeline ending with ToTensorV2
                 remap_labels=True,
                 depth_max=80.0): # Maximum depth value to consider valid (meters)
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.remap_labels = remap_labels
        self.depth_max = depth_max
        self.bf = BASELINE_FOCAL_LENGTH # Baseline * Focal Length constant

        # Define base directories
        self.images_base = osp.join(self.root, 'leftImg8bit', self.split)
        self.labels_base = osp.join(self.root, 'gtFine', self.split)
        self.disparity_base = osp.join(self.root, 'disparity', self.split)

        # Check if directories exist
        if not osp.isdir(self.images_base): raise RuntimeError(f"Image dir not found: {self.images_base}")
        if not osp.isdir(self.labels_base): raise RuntimeError(f"Label dir not found: {self.labels_base}")
        if not osp.isdir(self.disparity_base): raise RuntimeError(f"Disparity dir not found: {self.disparity_base}")

        # Find all corresponding image, label, and disparity files
        self.img_files = []
        self.label_files = []
        self.disp_files = []

        logger.info(f"Scanning for image/label/disparity triplets in {self.root} for split '{self.split}'...")
        for city in sorted(os.listdir(self.images_base)):
            img_dir = osp.join(self.images_base, city)
            label_dir = osp.join(self.labels_base, city)
            disp_dir = osp.join(self.disparity_base, city)
            # Check if corresponding label/disparity dirs exist for the city
            if not osp.isdir(img_dir) or not osp.isdir(label_dir) or not osp.isdir(disp_dir):
                logger.debug(f"Skipping city {city}, missing corresponding label or disparity dir.")
                continue

            for filename in sorted(os.listdir(img_dir)):
                if filename.endswith('_leftImg8bit.png'):
                    base_name = filename.replace('_leftImg8bit.png', '')
                    img_path = osp.join(img_dir, filename)
                    label_name = f"{base_name}_gtFine_labelIds.png"
                    label_path = osp.join(label_dir, label_name)
                    disp_name = f"{base_name}_disparity.png"
                    disp_path = osp.join(disp_dir, disp_name)

                    # Check if all three files exist for this sample
                    if osp.exists(label_path) and osp.exists(disp_path):
                        self.img_files.append(img_path)
                        self.label_files.append(label_path)
                        self.disp_files.append(disp_path)
                    # else: logger.debug(f"Skipping {base_name}, missing label or disparity file.")


        # Final checks and logging
        num_samples = len(self.img_files)
        if num_samples == 0: raise RuntimeError(f"No valid data triplets found for split '{self.split}' in {self.root}")
        if not (num_samples == len(self.label_files) == len(self.disp_files)):
            logger.warning(f"Mismatch in file counts for split '{self.split}'! Imgs: {len(self.img_files)}, Labels: {len(self.label_files)}, Disp: {len(self.disp_files)}")

        # Print info only once per split
        if not self._printed_info.get(self.split, False):
            logger.info(f"Loaded {num_samples} samples for split '{self.split}' from {self.root}")
            self._printed_info[self.split] = True

    def __len__(self):
        """Return the number of samples."""
        return len(self.img_files)

    @classmethod
    def map_labels_fast(cls, label_img_np):
        """Vectorized mapping of Cityscapes label IDs to train IDs."""
        mapped_labels = np.full_like(label_img_np, cls.SEG_IGNORE_INDEX, dtype=np.uint8)
        valid_mask = label_img_np < len(cls.ID_TO_TRAIN_ID) # Ensure index is within bounds
        label_ids_valid = label_img_np[valid_mask]
        mapped_labels[valid_mask] = cls.ID_TO_TRAIN_ID[label_ids_valid]
        return mapped_labels

    def disparity_to_depth(self, disp_map_np):
        """
        Converts disparity map (uint16) to depth map (float32), handling invalid values.
        Returns depth map and a boolean validity mask.
        """
        disp_map_np = disp_map_np.astype(np.float32)
        # Mask where original disparity is valid (non-zero)
        valid_original_disp_mask = disp_map_np > 0

        # Scale disparity according to Cityscapes formula
        disp_scaled = np.zeros_like(disp_map_np)
        disp_scaled[valid_original_disp_mask] = \
            (disp_map_np[valid_original_disp_mask] - DISPARITY_OFFSET) / DISPARITY_SCALE

        # Mask where scaled disparity is usable for depth calculation
        valid_scaled_disp_mask = disp_scaled > MIN_DISPARITY_SCALED

        # Initialize depth map with ignore value
        depth_map = np.full_like(disp_scaled, DEPTH_IGNORE_VALUE, dtype=np.float32)

        # Calculate depth only where scaled disparity is valid
        # Add small epsilon to denominator for safety, although MIN_DISPARITY_SCALED should handle it
        depth_map[valid_scaled_disp_mask] = self.bf / (disp_scaled[valid_scaled_disp_mask] + 1e-6)

        # Final validity mask considers:
        # 1. Original disparity was valid (>0)
        # 2. Calculated depth is positive (>0, already handled by disparity check)
        # 3. Calculated depth is within the specified maximum range
        valid_mask = valid_original_disp_mask & (depth_map <= self.depth_max)

        # Ensure pixels marked as invalid in the final mask have the ignore value in the depth map
        depth_map[~valid_mask] = DEPTH_IGNORE_VALUE

        return depth_map, valid_mask.astype(np.uint8) # Return mask as uint8 for albumentations

    def __getitem__(self, idx):
        """Load image, segmentation label, disparity, convert disparity to depth, apply transforms."""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        disp_path = self.disp_files[idx]

        try:
            # Load using PIL
            image_pil = Image.open(img_path).convert('RGB')
            label_seg_pil = Image.open(label_path)
            label_disp_pil = Image.open(disp_path) # uint16 disparity
        except Exception as e:
            logger.error(f"Error loading file for index {idx}: {e} (Paths: {img_path}, {label_path}, {disp_path})")
            return None, None, None, None # Return None for all items

        try:
            # Convert to NumPy arrays
            image_np = np.array(image_pil)
            label_seg_np = np.array(label_seg_pil)
            label_disp_np = np.array(label_disp_pil)

            # 1. Process Segmentation Label
            if self.remap_labels:
                label_seg_np_mapped = self.map_labels_fast(label_seg_np)
            else:
                label_seg_np_mapped = label_seg_np.astype(np.int64) # Use int64 if not remapping

            # 2. Process Disparity to Depth and get validity mask
            depth_np, valid_mask_np = self.disparity_to_depth(label_disp_np)

            # 3. Apply Augmentations using Albumentations
            if self.transform:
                # Apply transforms to image, segmentation mask, and depth map simultaneously
                # Pass masks as a list for consistent spatial transformation.
                # Depth map needs float32, seg map uint8.
                transformed = self.transform(
                    image=image_np,
                    masks=[label_seg_np_mapped.astype(np.uint8), depth_np.astype(np.float32)]
                )
                # Albumentations with ToTensorV2 at the end returns tensors
                image_tensor = transformed['image'] # Shape [C, H, W]
                label_tensor = transformed['masks'][0].to(torch.long) # Shape [H, W], cast to long
                depth_tensor = transformed['masks'][1].to(torch.float32) # Shape [H, W], cast to float

                # Recreate validity mask *after* spatial transforms using the transformed depth tensor
                # This handles cases where ignore values might appear/disappear due to interpolation
                valid_mask_tensor = (depth_tensor > DEPTH_IGNORE_VALUE).to(torch.bool) # Shape [H, W], cast to bool

            else:
                # Basic tensor conversion if no transform provided
                warnings.warn("No transform provided to CityscapesDepthSegDataset. Using basic ToTensor.")
                # Convert image HWC -> CHW, scale 0-1
                image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float().div(255.0)
                # Convert labels and mask
                label_tensor = torch.tensor(label_seg_np_mapped, dtype=torch.long)
                depth_tensor = torch.tensor(depth_np, dtype=torch.float32)
                valid_mask_tensor = torch.tensor(valid_mask_np, dtype=torch.bool) # Use the numpy mask here

            # Final check for tensor type before returning
            if not isinstance(image_tensor, torch.Tensor) or \
               not isinstance(label_tensor, torch.Tensor) or \
               not isinstance(depth_tensor, torch.Tensor) or \
               not isinstance(valid_mask_tensor, torch.Tensor):
                logger.error(f"Incorrect data type after processing index {idx}. Returning None.")
                return None, None, None, None

            return image_tensor, label_tensor, depth_tensor, valid_mask_tensor

        except Exception as e:
            logger.error(f"Error processing index {idx} ({img_path}): {e}", exc_info=True)
            return None, None, None, None # Return None for all items on error