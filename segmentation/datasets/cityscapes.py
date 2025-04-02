# ~/DenseCLIP/segmentation/datasets/cityscapes.py

import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import warnings

logger = logging.getLogger(__name__)

class CityscapesDataset(Dataset):
    """
    Cityscapes dataset for semantic segmentation using gtFine labelIds.
    """
    # Standard Cityscapes 19 classes mapping
    ID_TO_TRAIN_ID = np.array([255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4,
                            255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            255, 255, 16, 17, 18], dtype=np.uint8)

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]
    PALETTE = None # Add palette later if needed for visualization
    IGNORE_INDEX = 255 # Standard ignore index

    _printed_info = False

    def __init__(self,
                 root,
                 split='train',
                 transform=None, # Expects albumentations transform
                 remap_labels=True):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.remap_labels = remap_labels
        self.images_base = osp.join(self.root, 'leftImg8bit', self.split)
        self.labels_base = osp.join(self.root, 'gtFine', self.split)
        self.img_files = []
        self.label_files = []

        if not osp.isdir(self.images_base): raise RuntimeError(f"Image directory not found: {self.images_base}")
        if not osp.isdir(self.labels_base): raise RuntimeError(f"Label directory not found: {self.labels_base}")

        for city in sorted(os.listdir(self.images_base)):
            img_dir = osp.join(self.images_base, city)
            label_dir = osp.join(self.labels_base, city)
            if not osp.isdir(img_dir) or not osp.isdir(label_dir): continue

            for filename in sorted(os.listdir(img_dir)):
                if filename.endswith('_leftImg8bit.png'):
                    base_name = filename.replace('_leftImg8bit.png', '')
                    label_name = f"{base_name}_gtFine_labelIds.png"
                    label_path = osp.join(label_dir, label_name)

                    if osp.exists(label_path):
                        self.img_files.append(osp.join(img_dir, filename))
                        self.label_files.append(label_path)

        if not self.img_files: raise RuntimeError(f"No image files found in {self.images_base}")
        if not self.label_files: raise RuntimeError(f"No label files found in {self.labels_base}")
        if len(self.img_files) != len(self.label_files):
            logger.warning("Mismatch between number of images and labels found!")

        if not CityscapesDataset._printed_info:
            logger.info(f"Loaded {len(self.img_files)} samples for split '{self.split}' from {self.root}")
            CityscapesDataset._printed_info = True

    def __len__(self):
        return len(self.img_files)

    @classmethod
    def map_labels_fast(cls, label_img_np):
        """Vectorized mapping of Cityscapes label IDs to train IDs."""
        mapped_labels = np.full_like(label_img_np, 255, dtype=np.uint8)
        valid_mask = label_img_np < len(cls.ID_TO_TRAIN_ID)
        mapped_labels[valid_mask] = cls.ID_TO_TRAIN_ID[label_img_np[valid_mask]]
        return mapped_labels

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path)
        except Exception as e:
            logger.error(f"Error loading index {idx} ({img_path}/{label_path}): {e}")
            return None, None

        image_np = np.array(image)
        label_np = np.array(label)

        if self.remap_labels:
            label_np_mapped = self.map_labels_fast(label_np)
        else:
            label_np_mapped = label_np.astype(np.int64)

        if self.transform:
            try:
                transformed = self.transform(image=image_np, mask=label_np_mapped)
                image_transformed = transformed['image']
                label_transformed = transformed['mask']
            except Exception as e:
                logger.error(f"Error applying transform for index {idx}: {e}")
                return None, None
        else:
            warnings.warn("No transform provided to CityscapesDataset. Using basic ToTensor.")
            image_transformed = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
            label_transformed = torch.tensor(label_np_mapped, dtype=torch.long)

        if image_transformed is None or label_transformed is None:
            logger.error(f"Returning None for index {idx} due to previous error.")
            return None, None

        return image_transformed, label_transformed
