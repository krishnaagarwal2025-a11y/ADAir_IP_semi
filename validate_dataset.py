#!/usr/bin/env python3
"""
AdaIR Dataset Validation Script
================================
Validates image and NumPy array (.npy) datasets to ensure they match AdaIR's
expected directory format and data specifications (input/target pairs, resolution,
channels, data types, and file integrity).

Supported File Types:
- Standard Images: .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp
- NumPy Arrays:    .npy
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Supported image and array extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.npy'}

# System / OS Metadata files to explicitly ignore
IGNORED_SYSTEM_FILES = {'.ds_store', 'thumbs.db', 'desktop.ini', '.gitignore'}


def is_valid_image_file(path: Path) -> bool:
    """Returns True if file is a non-hidden, valid image/npy file, filtering OS metadata."""
    name_lower = path.name.lower()
    if name_lower.startswith('.') or name_lower.startswith('._'):
        return False
    if name_lower in IGNORED_SYSTEM_FILES:
        return False
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


class DatasetValidator:
    def __init__(
        self,
        data_dir: str,
        input_dir_name: str = 'NoisyLR',
        target_dir_name: str = 'GT',
        min_patch_size: int = 128,
        task: str = 'auto',
        verbose: bool = False
    ):
        self.data_dir = Path(data_dir).resolve()
        self.input_dir_name = input_dir_name
        self.target_dir_name = target_dir_name
        self.min_patch_size = min_patch_size
        self.task = task.lower()
        self.verbose = verbose
        
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats = {
            'total_pairs_checked': 0,
            'valid_pairs': 0,
            'corrupted_images': 0,
            'mismatched_dimensions': 0,
            'undersized_images': 0,
            'missing_targets': 0,
            'orphan_targets': 0,
            'channel_warnings': 0,
            'nan_inf_errors': 0,
            'ignored_system_files': 0
        }

    def log_error(self, message: str):
        self.errors.append(message)
        if self.verbose:
            print(f"[ERROR] {message}")

    def log_warning(self, message: str):
        self.warnings.append(message)
        if self.verbose:
            print(f"[WARNING] {message}")

    def log_info(self, message: str):
        print(f"[INFO] {message}")

    def find_subfolder(self, base_path: Path, candidates: List[str]) -> Optional[Path]:
        for candidate in candidates:
            subpath = base_path / candidate if not Path(candidate).is_absolute() else Path(candidate)
            if subpath.exists() and subpath.is_dir():
                return subpath
            if base_path.exists() and base_path.is_dir():
                for item in base_path.iterdir():
                    if item.is_dir() and item.name.lower() == candidate.lower():
                        return item
        return None

    def match_target_file(self, input_path: Path, target_files: Dict[str, Path], task_type: str = 'generic') -> Optional[Path]:
        """Matches target file by exact filename, stem match (e.g. img.npy <-> img.png), or task rule."""
        input_name = input_path.name

        # Exact filename match
        if input_name in target_files:
            return target_files[input_name]

        # Stem match (e.g. 0001.npy <-> 0001.png or 0001.npy <-> 0001.npy)
        stem = input_path.stem
        for t_name, t_path in target_files.items():
            if t_path.stem == stem:
                return t_path

        # Task-specific pattern match: rain-X -> norain-X
        if task_type == 'derain' and 'rain-' in input_name:
            norain_name = input_name.replace('rain-', 'norain-')
            if norain_name in target_files:
                return target_files[norain_name]

        # Task-specific pattern match: dehaze SOTS 0025_0.8_0.1.jpg -> 0025.png
        if task_type == 'dehaze' and '_' in input_name:
            prefix = input_name.split('_')[0]
            for t_name, t_path in target_files.items():
                if t_path.stem == prefix:
                    return t_path

        return None

    def validate_image_or_npy_file(self, img_path: Path) -> Optional[Tuple[Tuple[int, int], str, int]]:
        """
        Validates an image or .npy file's integrity.
        Returns ((Width, Height), data_type_str, channels).
        """
        suffix = img_path.suffix.lower()

        if suffix == '.npy':
            try:
                arr = np.load(img_path)
                
                # Check NaN / Inf
                if np.isnan(arr).any() or np.isinf(arr).any():
                    self.stats['nan_inf_errors'] += 1
                    self.log_error(f"NumPy file contains NaN or Inf values: {img_path}")
                    return None

                dtype_str = str(arr.dtype)

                if arr.ndim == 2:
                    h, w = arr.shape
                    channels = 1
                elif arr.ndim == 3:
                    # Check (C, H, W) vs (H, W, C)
                    if arr.shape[0] in (1, 3, 4) and arr.shape[2] > 4:
                        channels, h, w = arr.shape
                    else:
                        h, w, channels = arr.shape
                else:
                    self.log_error(f"Invalid NumPy array dimensions ({arr.ndim}D) for file: {img_path}")
                    return None

                return (w, h), dtype_str, channels

            except Exception as e:
                self.stats['corrupted_images'] += 1
                self.log_error(f"Corrupt or unreadable .npy file: {img_path} ({e})")
                return None
        else:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    img.load()
                    size = img.size
                    mode = img.mode
                    channels = len(img.getbands())
                    return size, mode, channels
            except Exception as e:
                self.stats['corrupted_images'] += 1
                self.log_error(f"Corrupt or unreadable image file: {img_path} ({e})")
                return None

    def validate_pair_directory(self, input_dir: Path, target_dir: Path, task_type: str = 'generic') -> bool:
        """Validates input and target directory pairs."""
        self.log_info(f"Validating task [{task_type.upper()}] directory:")
        self.log_info(f"  Input dir:  {input_dir}")
        self.log_info(f"  Target dir: {target_dir}")

        input_files = {
            f.name: f for f in input_dir.rglob('*') if is_valid_image_file(f)
        }
        target_files = {
            f.name: f for f in target_dir.rglob('*') if is_valid_image_file(f)
        }

        all_in_items = list(input_dir.rglob('*'))
        ignored_count = sum(1 for f in all_in_items if f.is_file() and not is_valid_image_file(f))
        self.stats['ignored_system_files'] += ignored_count
        if ignored_count > 0:
            self.log_info(f"Ignored {ignored_count} system/non-image metadata file(s) (e.g. .DS_Store, Thumbs.db).")

        if not input_files:
            self.log_error(f"No valid images or .npy files found in input directory: {input_dir}")
            return False

        if not target_files:
            self.log_error(f"No valid images or .npy files found in target directory: {target_dir}")
            return False

        self.log_info(f"Found {len(input_files)} input files and {len(target_files)} target files.")

        matched_targets = set()
        
        for input_name, input_path in input_files.items():
            self.stats['total_pairs_checked'] += 1
            
            target_path = self.match_target_file(input_path, target_files, task_type=task_type)

            if not target_path:
                self.stats['missing_targets'] += 1
                self.log_error(f"Input file '{input_name}' in {input_dir.name} has no matching target in {target_dir.name}")
                continue

            matched_targets.add(target_path)

            input_info = self.validate_image_or_npy_file(input_path)
            if not input_info:
                continue

            target_info = self.validate_image_or_npy_file(target_path)
            if not target_info:
                continue

            (in_w, in_h), in_mode, in_c = input_info
            (tg_w, tg_h), tg_mode, tg_c = target_info

            if (in_w, in_h) != (tg_w, tg_h):
                self.stats['mismatched_dimensions'] += 1
                self.log_error(
                    f"Dimension mismatch for pair '{input_name}': "
                    f"Input ({in_w}x{in_h}) vs Target ({tg_w}x{tg_h})"
                )
                continue

            if in_w < self.min_patch_size or in_h < self.min_patch_size:
                self.stats['undersized_images'] += 1
                self.log_error(
                    f"Pair '{input_name}' resolution ({in_w}x{in_h}) is smaller than "
                    f"minimum patch size requirement ({self.min_patch_size}x{self.min_patch_size})"
                )
                continue

            if in_c not in (1, 3) or tg_c not in (1, 3):
                self.stats['channel_warnings'] += 1
                self.log_warning(
                    f"Pair '{input_name}' channels: Input ({in_c}ch, {in_mode}) / Target ({tg_c}ch, {tg_mode})."
                )

            self.stats['valid_pairs'] += 1

        orphans = set(target_files.values()) - matched_targets
        if orphans:
            self.stats['orphan_targets'] += len(orphans)
            self.log_warning(f"Found {len(orphans)} target files with no corresponding input file.")

        return True

    def validate(self) -> bool:
        """Executes full dataset validation."""
        print("=" * 70)
        print(f"[START] Starting AdaIR Dataset Validation on: {self.data_dir}")
        print(f"        Input Subfolder:  {self.input_dir_name}")
        print(f"        Target Subfolder: {self.target_dir_name}")
        print(f"        Min Patch Size:   {self.min_patch_size}x{self.min_patch_size}")
        print("=" * 70)

        if not self.data_dir.exists():
            self.log_error(f"Dataset path does not exist: {self.data_dir}")
            self.print_summary()
            return False

        task_dirs_found = 0

        input_candidates = [self.input_dir_name, 'NoisyLR', 'input', 'inputs', 'degraded', 'blur', 'rainy', 'low', 'synthetic']
        target_candidates = [self.target_dir_name, 'GT', 'gt', 'target', 'targets', 'clean', 'sharp', 'original']

        direct_input = self.find_subfolder(self.data_dir, input_candidates)
        direct_target = self.find_subfolder(self.data_dir, target_candidates)

        if direct_input and direct_target:
            task_dirs_found += 1
            self.validate_pair_directory(direct_input, direct_target, task_type='generic')
        else:
            for task_key in ['deblur', 'derain', 'dehaze', 'enhance']:
                task_folder = None
                if self.data_dir.exists() and self.data_dir.is_dir():
                    for sub in self.data_dir.iterdir():
                        if sub.is_dir() and sub.name.lower() == task_key:
                            task_folder = sub
                            break
                
                if task_folder:
                    in_dir = self.find_subfolder(task_folder, input_candidates)
                    tg_dir = self.find_subfolder(task_folder, target_candidates)
                    if in_dir and tg_dir:
                        task_dirs_found += 1
                        self.validate_pair_directory(in_dir, tg_dir, task_type=task_key)

        if task_dirs_found == 0:
            train_sub = self.find_subfolder(self.data_dir, ['train', 'Train'])
            if train_sub:
                self.log_info(f"Found subfolder '{train_sub.name}'. Recursing into train directory...")
                self.data_dir = train_sub
                return self.validate()
            
            self.log_error(
                f"Could not locate valid '{self.input_dir_name}' (input) and '{self.target_dir_name}' (target) "
                f"subdirectories in {self.data_dir}.\n"
                f"Expected directory format:\n"
                f"  {self.data_dir}/{self.input_dir_name}/  (degraded images / .npy files)\n"
                f"  {self.data_dir}/{self.target_dir_name}/       (clean images / .npy files)"
            )

        self.print_summary()
        return len(self.errors) == 0

    def print_summary(self):
        print("\n" + "=" * 70)
        print("DATASET VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Pairs Checked:       {self.stats['total_pairs_checked']}")
        print(f"[OK] Valid Image/NPY Pairs:   {self.stats['valid_pairs']}")
        print(f"[FAIL] Missing Target Pairs:  {self.stats['missing_targets']}")
        print(f"[FAIL] Corrupted Files:       {self.stats['corrupted_images']}")
        print(f"[FAIL] NaN/Inf Array Errors:  {self.stats['nan_inf_errors']}")
        print(f"[FAIL] Mismatched Dimensions: {self.stats['mismatched_dimensions']}")
        print(f"[FAIL] Undersized (<{self.min_patch_size}px): {self.stats['undersized_images']}")
        print(f"[WARN] Orphan Target Files:   {self.stats['orphan_targets']}")
        print(f"[WARN] Channel Warnings:      {self.stats['channel_warnings']}")
        print(f"[INFO] Ignored System Files:  {self.stats['ignored_system_files']}")
        print("-" * 70)

        if self.errors:
            print(f"[RESULT] Validation FAILED with {len(self.errors)} error(s):")
            for idx, err in enumerate(self.errors[:10], 1):
                print(f"  {idx}. {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more error(s).")
        else:
            print("[RESULT] Validation PASSED! Dataset (.png/.jpg/.npy) matches AdaIR format.")
        print("=" * 70)

    def save_report(self, output_path: str):
        report_data = {
            'data_dir': str(self.data_dir),
            'input_dir_name': self.input_dir_name,
            'target_dir_name': self.target_dir_name,
            'min_patch_size': self.min_patch_size,
            'is_valid': len(self.errors) == 0,
            'stats': self.stats,
            'errors': self.errors,
            'warnings': self.warnings
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"[INFO] Validation report saved to: {output_path}")


def create_sample_dataset(target_dir: Path):
    """Utility to generate sample NoisyLR / GT image & .npy dataset for testing."""
    print(f"[SETUP] Creating sample NoisyLR & GT dataset (with .npy files) at: {target_dir}")
    train_dir = target_dir / "train"
    input_dir = train_dir / "NoisyLR"
    target_dir_path = train_dir / "GT"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Standard PNG pairs
    for i in range(1, 4):
        img_name = f"sample_{i:03d}.png"
        img = Image.new('RGB', (256, 256), color=(i * 40, 100, 150))
        img.save(input_dir / img_name)
        img.save(target_dir_path / img_name)

    # 2. NumPy .npy pairs
    for i in range(4, 6):
        npy_name = f"sample_{i:03d}.npy"
        arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        np.save(input_dir / npy_name, arr)
        np.save(target_dir_path / npy_name, arr)

    print(f"[SETUP] Created sample image & .npy pairs in NoisyLR/ and GT/ under {train_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate NoisyLR / GT image and .npy dataset structure for AdaIR."
    )
    parser.add_argument(
        '--data_dir', '-d', type=str, default='data/train',
        help="Path to dataset directory to validate (e.g. data/train)."
    )
    parser.add_argument(
        '--input_dir', type=str, default='NoisyLR',
        help="Name or subfolder for input degraded images / .npy files (default: NoisyLR)."
    )
    parser.add_argument(
        '--target_dir', type=str, default='GT',
        help="Name or subfolder for ground-truth clean images / .npy files (default: GT)."
    )
    parser.add_argument(
        '--min_patch_size', '-p', type=int, default=128,
        help="Minimum required image/array width and height (default: 128)."
    )
    parser.add_argument(
        '--task', '-t', type=str, default='auto',
        choices=['auto', 'generic', 'deblur', 'derain', 'dehaze', 'enhance'],
        help="Task type to validate (default: auto)."
    )
    parser.add_argument(
        '--output_report', '-o', type=str, default=None,
        help="Optional path to save JSON validation report."
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Enable verbose output logging."
    )
    parser.add_argument(
        '--create_sample', action='store_true',
        help="Create a sample dataset with .png and .npy files."
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset(Path('data'))
        if args.data_dir == 'data/train' and not Path(args.data_dir).exists():
            args.data_dir = 'data/train'

    validator = DatasetValidator(
        data_dir=args.data_dir,
        input_dir_name=args.input_dir,
        target_dir_name=args.target_dir,
        min_patch_size=args.min_patch_size,
        task=args.task,
        verbose=args.verbose
    )

    success = validator.validate()

    if args.output_report:
        validator.save_report(args.output_report)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
