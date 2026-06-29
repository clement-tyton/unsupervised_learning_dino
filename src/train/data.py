import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiCropTransform:
    """Generates 2 global + N local crops from a single image."""
 
    def __init__(
        self,
        global_size=224,
        local_size=96,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=6,
    ):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        shared_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
 
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale),
            *shared_aug,
            transforms.ToTensor(),
            normalize,
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale),
            *shared_aug,
            transforms.ToTensor(),
            normalize,
        ])
        self.n_local_crops = n_local_crops
 
    def __call__(self, img):
        """img: PIL Image"""
        crops = [
            self.global_transform(img),  # global view 1  → teacher + student
            self.global_transform(img),  # global view 2  → teacher + student
        ]
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(img))  # local views → student only
        return crops  # list of tensors
 
 
class GeoTIFFDataset(Dataset):
    """
    Loads GeoTIFF files from a flat directory.
    Assumes files are already at the right spatial size.
    Only uses the first 3 bands (RGB). Adjust if needed.
    """
 
    def __init__(self, root_dir, transform=None):
        self.paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        assert len(self.paths) > 0, f"No GeoTIFF files found in {root_dir}"
        self.transform = transform
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        with rasterio.open(self.paths[idx]) as src:
            # Read first 3 bands, shape: (3, H, W)
            arr = src.read([1, 2, 3]).astype(np.float32)
 
        # Normalize to [0, 255] uint8 for PIL compatibility
        for i in range(arr.shape[0]):
            band = arr[i]
            lo, hi = band.min(), band.max()
            if hi > lo:
                arr[i] = (band - lo) / (hi - lo) * 255.0
 
        arr = arr.astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3)
 
        from PIL import Image
        img = Image.fromarray(arr)
 
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)
 
 
def collate_crops(batch):
    """
    batch: list of lists-of-tensors (one per image)
    Returns: list of stacked tensors, one per crop index
    """
    n_crops = len(batch[0])
    return [torch.stack([b[i] for b in batch]) for i in range(n_crops)]
 
 