import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class SemiconDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, patch_size=128, phase='train', seed=42):
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        self.patch_size = patch_size
        self.phase = phase
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Verify dataset length
        assert len(self.noisy_paths) == len(self.gt_paths), "Mismatch between NoisyLR and GT counts"

    def __len__(self):
        return len(self.noisy_paths)

    def _normalize(self, noisy, gt):
        # Joint normalization strategy: Ensures physical relative intensities are preserved
        max_val = max(noisy.max(), gt.max())
        if max_val == 0:
            return noisy, gt
        return noisy / max_val, gt / max_val

    def __getitem__(self, idx):
        # Load npy files
        noisy_img = np.load(self.noisy_paths[idx]).astype(np.float32)
        gt_img = np.load(self.gt_paths[idx]).astype(np.float32)

        # Normalize
        noisy_img, gt_img = self._normalize(noisy_img, gt_img)

        # Convert to Tensor (C, H, W) where C=1
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)

        # Data augmentation
        if self.phase == 'train':
            # Random horizontal & vertical flips
            if random.random() > 0.5:
                noisy_tensor = torch.flip(noisy_tensor, [2])
                gt_tensor = torch.flip(gt_tensor, [2])
            if random.random() > 0.5:
                noisy_tensor = torch.flip(noisy_tensor, [1])
                gt_tensor = torch.flip(gt_tensor, [1])
            # Random rotations (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                noisy_tensor = torch.rot90(noisy_tensor, k, [1, 2])
                gt_tensor = torch.rot90(gt_tensor, k, [1, 2])

        return noisy_tensor, gt_tensor
