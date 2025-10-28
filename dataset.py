"""
Dataset and DataLoader for DeepOSWSRM Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from rasterio.windows import Window
import os
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepOSWSRMDataset(Dataset):
    """
    Dataset for DeepOSWSRM training
    
    Handles:
    - Loading Sentinel-1 and Sentinel-2 patches
    - Loading reference water masks
    - Computing water fractions
    - Applying augmentations
    - Simulating cloud cover
    """
    
    def __init__(self, 
                 data_dir,
                 patch_size=64,
                 scale_factor=4,
                 cloud_coverage_range=(0.3, 0.8),
                 augment=True,
                 normalize=True):
        """
        Args:
            data_dir: Root directory containing the data
            patch_size: Size of patches to extract (for coarse resolution)
            scale_factor: Super-resolution scale factor
            cloud_coverage_range: Range of cloud coverage for simulation (min, max)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize the data
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.fine_patch_size = patch_size * scale_factor
        self.cloud_coverage_range = cloud_coverage_range
        self.augment = augment
        self.normalize = normalize
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Prepare sample list
        self.samples = []
        self._prepare_samples()
        
        # Setup augmentation
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            ])
        else:
            self.transform = None
    
    def _prepare_samples(self):
        """Prepare list of valid samples from metadata"""
        for entry in self.metadata:
            if 'water_mask' in entry and os.path.exists(entry['sentinel1']):
                self.samples.append(entry)
        
        print(f"Found {len(self.samples)} valid samples")
    
    def _load_image(self, path, window=None):
        """Load image from file"""
        with rasterio.open(path) as src:
            if window:
                data = src.read(window=window)
            else:
                data = src.read()
            return data.astype(np.float32)
    
    def _compute_fraction(self, fine_mask):
        """
        Compute coarse water fraction from fine resolution mask
        
        Args:
            fine_mask: Fine resolution binary mask [H*scale, W*scale]
        
        Returns:
            Coarse resolution fraction map [H, W] - same size as input
        """
        import cv2
        # The target fraction should be at the same resolution as the input
        # which is (patch_size, patch_size), not downsampled
        h_coarse = self.patch_size
        w_coarse = self.patch_size
        
        # Resize fine mask to coarse resolution and compute average
        fraction = cv2.resize(fine_mask.astype(np.float32), 
                            (w_coarse, h_coarse), 
                            interpolation=cv2.INTER_AREA)
        
        return fraction.astype(np.float32)
    
    def _simulate_clouds(self, image, coverage_rate=None):
        """
        Simulate cloud cover by masking parts of the image
        
        Args:
            image: Image array [C, H, W]
            coverage_rate: Cloud coverage rate (if None, randomly sample)
        
        Returns:
            Masked image and cloud mask
        """
        if coverage_rate is None:
            coverage_rate = np.random.uniform(*self.cloud_coverage_range)
        
        h, w = image.shape[1:]
        cloud_mask = np.zeros((h, w), dtype=np.float32)
        
        # Generate multiple cloud patches
        num_clouds = np.random.randint(3, 8)
        
        for _ in range(num_clouds):
            # Random cloud center
            center_y = np.random.randint(0, h)
            center_x = np.random.randint(0, w)
            
            # Random cloud size
            size_y = np.random.randint(h // 10, h // 3)
            size_x = np.random.randint(w // 10, w // 3)
            
            # Create elliptical cloud
            y_grid, x_grid = np.ogrid[:h, :w]
            cloud = ((y_grid - center_y) ** 2 / (size_y ** 2 + 1e-8) + 
                    (x_grid - center_x) ** 2 / (size_x ** 2 + 1e-8)) <= 1
            
            cloud_mask = np.logical_or(cloud_mask, cloud)
            
            # Check coverage
            if cloud_mask.sum() / cloud_mask.size >= coverage_rate:
                break
        
        # Apply mask to image
        masked_image = image.copy()
        cloud_mask_3d = np.repeat(cloud_mask[np.newaxis, :, :], image.shape[0], axis=0)
        masked_image[cloud_mask_3d > 0] = 0
        
        return masked_image, cloud_mask.astype(np.float32)
    
    def _normalize_sentinel1(self, s1_data):
        """Normalize Sentinel-1 data (VV, VH in dB)"""
        # Convert to dB if not already
        s1_data = 10 * np.log10(np.abs(s1_data) + 1e-8)
        # Clip to reasonable range
        s1_data = np.clip(s1_data, -25, 5)
        # Normalize to [0, 1]
        s1_data = (s1_data + 25) / 30
        return s1_data
    
    def _normalize_sentinel2(self, s2_data):
        """Normalize Sentinel-2 data (reflectance)"""
        # Clip to [0, 1] (already in reflectance)
        s2_data = np.clip(s2_data, 0, 1)
        return s2_data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        sample_info = self.samples[idx]
        
        # Load images
        s1_data = self._load_image(sample_info['sentinel1'])
        s2_data = self._load_image(sample_info['sentinel2'])
        water_mask = self._load_image(sample_info['water_mask'])
        # Debug: check mask shape consistency
        print(f"Water mask shape: {water_mask.shape}, should be: {s1_data.shape[1]*self.scale_factor}×{s1_data.shape[2]*self.scale_factor}")

        
        # Handle single channel mask
        if water_mask.shape[0] == 1:
            water_mask = water_mask[0]
        
        # Get random patch location
        h, w = s1_data.shape[1:]
        
        # Ensure we can extract full patch
        if h < self.patch_size or w < self.patch_size:
            # Pad if needed
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            s1_data = np.pad(s1_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            s2_data = np.pad(s2_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            water_mask = np.pad(water_mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = s1_data.shape[1:]
        
        # Random crop
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        
        s1_patch = s1_data[:, top:top+self.patch_size, left:left+self.patch_size]
        s2_patch = s2_data[:, top:top+self.patch_size, left:left+self.patch_size]
        
        # Extract corresponding fine resolution patch
        # After loading water_mask
        print(f"Water mask shape: {water_mask.shape}, should be: {h*self.scale_factor}×{w*self.scale_factor}")

        # Get dimensions
        h, w = s1_data.shape[1:]
        h_mask, w_mask = water_mask.shape

        # Calculate the scale ratio (should be close to scale_factor)
        actual_scale_h = h_mask / h
        actual_scale_w = w_mask / w

        # Ensure we can extract full patch
        if h < self.patch_size or w < self.patch_size:
            # Pad if needed
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            s1_data = np.pad(s1_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            s2_data = np.pad(s2_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            
            # Also pad water mask proportionally
            pad_h_mask = int(pad_h * actual_scale_h)
            pad_w_mask = int(pad_w * actual_scale_w)
            water_mask = np.pad(water_mask, ((0, pad_h_mask), (0, pad_w_mask)), mode='reflect')
            
            h, w = s1_data.shape[1:]
            h_mask, w_mask = water_mask.shape

        # Random crop at coarse resolution
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        s1_patch = s1_data[:, top:top+self.patch_size, left:left+self.patch_size]
        s2_patch = s2_data[:, top:top+self.patch_size, left:left+self.patch_size]

        # Extract corresponding fine resolution patch using actual scale
        top_fine = int(top * actual_scale_h)
        left_fine = int(left * actual_scale_w)
        fine_patch_h = int(self.patch_size * actual_scale_h)
        fine_patch_w = int(self.patch_size * actual_scale_w)

        water_patch = water_mask[top_fine:top_fine+fine_patch_h, 
                                left_fine:left_fine+fine_patch_w]

        # Ensure water_patch is exactly the right size (may be off by 1-2 pixels due to rounding)
        if water_patch.shape != (self.fine_patch_size, self.fine_patch_size):
            import cv2
            water_patch = cv2.resize(water_patch, (self.fine_patch_size, self.fine_patch_size), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Compute water fraction
        water_fraction = self._compute_fraction(water_patch)
        
        # Apply augmentation (on numpy arrays)
        if self.transform:
            # Augment all together to maintain correspondence
            combined = np.concatenate([
                s1_patch,
                s2_patch,
            ], axis=0)
            
            # Convert to HWC for albumentations
            combined = np.transpose(combined, (1, 2, 0))
            water_patch_hwc = water_patch
            
            import cv2

            # Ensure image and mask have the same H×W before transform
            h, w = combined.shape[:2]
            mh, mw = water_patch_hwc.shape[:2]
            if (h != mh) or (w != mw):
                water_patch_hwc = cv2.resize(water_patch_hwc, (w, h), interpolation=cv2.INTER_NEAREST)

            augmented = self.transform(image=combined, mask=water_patch_hwc)
            combined = augmented['image']
            water_patch = augmented['mask']
            
            # Convert back to CHW
            combined = np.transpose(combined, (2, 0, 1))
            
            # Split back
            s1_patch = combined[:2]
            s2_patch = combined[2:6]
            
            # Recompute fraction after augmentation
            water_fraction = self._compute_fraction(water_patch)
        
        # Simulate clouds on Sentinel-2
        s2_patch, cloud_mask = self._simulate_clouds(s2_patch)
        
        # Normalize
        if self.normalize:
            s1_patch = self._normalize_sentinel1(s1_patch)
            s2_patch = self._normalize_sentinel2(s2_patch)
        
        # Convert to tensors
        s1_tensor = torch.from_numpy(s1_patch).float()
        s2_tensor = torch.from_numpy(s2_patch).float()
        fraction_tensor = torch.from_numpy(water_fraction[np.newaxis, :, :]).float()
        water_tensor = torch.from_numpy(water_patch[np.newaxis, :, :]).long()
        cloud_tensor = torch.from_numpy(cloud_mask[np.newaxis, :, :]).float()
        
        return {
            'sentinel1': s1_tensor,
            'sentinel2': s2_tensor,
            'water_fraction': fraction_tensor,
            'water_map': water_tensor,
            'cloud_mask': cloud_tensor,
            'site_name': sample_info['site_name']
        }


def create_dataloaders(data_dir, 
                      batch_size=8,
                      patch_size=64,
                      scale_factor=4,
                      num_workers=4,
                      train_split=0.8):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        patch_size: Patch size for coarse resolution
        scale_factor: Super-resolution scale factor
        num_workers: Number of worker processes
        train_split: Proportion of data for training
    
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    full_dataset = DeepOSWSRMDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        scale_factor=scale_factor,
        augment=True
    )
    
    # Split into train and validation
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    data_dir = './deeposwsrm_data'
    
    if os.path.exists(data_dir):
        print("Testing dataset...")
        dataset = DeepOSWSRMDataset(
            data_dir=data_dir,
            patch_size=64,
            scale_factor=4,
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        
        print("\nSample shapes:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # Test dataloader
        print("\nTesting dataloader...")
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Get a batch
        batch = next(iter(train_loader))
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run data_download.py first to download data")