import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.dataset_utils import AdaIRTrainDataset, crop_img
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
    from pytorch_lightning.callbacks import Callback

# Attempt to import optional metric packages
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    import torchmetrics
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


IGNORED_SYSTEM_FILES = {'.ds_store', 'thumbs.db', 'desktop.ini', '.gitignore'}

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.npy')

def is_valid_image_file(path: Path) -> bool:
    """Returns True if file is a non-hidden, valid image/npy file, filtering OS metadata."""
    name_lower = path.name.lower()
    if name_lower.startswith('.') or name_lower.startswith('._'):
        return False
    if name_lower in IGNORED_SYSTEM_FILES:
        return False
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def load_image_or_npy(path) -> np.ndarray:
    """Loads an image file or a NumPy array (.npy) file into a (H, W, 3) uint8 RGB array."""
    path_str = str(path)
    if path_str.lower().endswith('.npy'):
        arr = np.load(path_str)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[2] > 4:
                arr = arr.transpose(1, 2, 0)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    else:
        return np.array(Image.open(path_str).convert('RGB'))


class GenericPairedValDataset(Dataset):
    """Dataset loader for validation input/target paired images (supporting NoisyLR/GT and .npy files)."""
    def __init__(self, val_dir: str, input_dir_name: str = 'NoisyLR', target_dir_name: str = 'GT'):
        super().__init__()
        self.val_dir = Path(val_dir)
        self.pairs = []

        in_candidates = [input_dir_name, 'NoisyLR', 'input', 'inputs', 'degraded', 'blur', 'rainy', 'low', 'synthetic']
        tg_candidates = [target_dir_name, 'GT', 'gt', 'target', 'targets', 'clean', 'sharp', 'original']

        def find_folder(base: Path, candidates: list) -> Path:
            for cand in candidates:
                p = base / cand
                if p.exists() and p.is_dir():
                    return p
                if base.exists() and base.is_dir():
                    for item in base.iterdir():
                        if item.is_dir() and item.name.lower() == cand.lower():
                            return item
            return base

        input_dir = find_folder(self.val_dir, in_candidates)
        target_dir = find_folder(self.val_dir, tg_candidates)

        if input_dir.exists() and target_dir.exists():
            in_files = {f.name: f for f in input_dir.rglob('*') if is_valid_image_file(f)}
            tg_files = {f.name: f for f in target_dir.rglob('*') if is_valid_image_file(f)}

            for name, in_path in in_files.items():
                tg_path = tg_files.get(name)
                if not tg_path:
                    for tname, tp in tg_files.items():
                        if tp.stem == in_path.stem:
                            tg_path = tp
                            break
                if tg_path:
                    self.pairs.append((in_path, tg_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_path, tg_path = self.pairs[idx]
        in_img = load_image_or_npy(in_path)
        tg_img = load_image_or_npy(tg_path)

        in_img = crop_img(in_img, base=16)
        tg_img = crop_img(tg_img, base=16)

        in_tensor = torch.from_numpy(in_img).permute(2, 0, 1).float() / 255.0
        tg_tensor = torch.from_numpy(tg_img).permute(2, 0, 1).float() / 255.0

        return [in_path.stem], in_tensor, tg_tensor


class ValidationMetricsTracker:
    """Computes PSNR, SSIM, and LPIPS metrics and manages CSV & checkpoint logging."""
    def __init__(self, device: torch.device, ckpt_dir: str, metrics_file: str):
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.metrics_file = Path(metrics_file)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_psnr = -float('inf')
        self.best_lpips = float('inf')

        self.lpips_fn = None
        if HAS_LPIPS:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
                self.lpips_fn.eval()
            except Exception as e:
                print(f"[Warning] Failed to initialize LPIPS: {e}")

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> tuple:
        model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if len(batch) == 3:
                _, degrad_patch, clean_patch = batch
            else:
                degrad_patch, clean_patch = batch

            degrad_patch = degrad_patch.to(self.device)
            clean_patch = clean_patch.to(self.device)

            restored = model(degrad_patch)
            restored = torch.clamp(restored, 0.0, 1.0)

            rec_np = restored.cpu().numpy().transpose(0, 2, 3, 1)
            cln_np = clean_patch.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(rec_np.shape[0]):
                p_val = peak_signal_noise_ratio(cln_np[i], rec_np[i], data_range=1.0)
                try:
                    s_val = structural_similarity(cln_np[i], rec_np[i], data_range=1.0, channel_axis=-1)
                except TypeError:
                    s_val = structural_similarity(cln_np[i], rec_np[i], data_range=1.0, multichannel=True)

                psnr_list.append(p_val)
                ssim_list.append(s_val)

            if self.lpips_fn is not None:
                rec_norm = restored * 2.0 - 1.0
                cln_norm = clean_patch * 2.0 - 1.0
                lp_val = self.lpips_fn(rec_norm, cln_norm).mean().item()
                lpips_list.append(lp_val)

        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

        return avg_psnr, avg_ssim, avg_lpips

    def record_epoch(self, epoch: int, train_loss: float, val_psnr: float, val_ssim: float, val_lpips: float, model: nn.Module):
        file_exists = self.metrics_file.exists()
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips'])
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_psnr:.4f}", f"{val_ssim:.4f}", f"{val_lpips:.4f}"])
            f.flush()

        print(f"\n[Epoch {epoch:03d}] Train Loss: {train_loss:.6f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}")

        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            best_psnr_path = self.ckpt_dir / "best_psnr_model.pth"
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'psnr': val_psnr, 'ssim': val_ssim}, best_psnr_path)
            print(f" [CHECKPOINT] Saved new best PSNR model ({val_psnr:.2f} dB) -> {best_psnr_path.name}")

        if val_lpips > 0 and val_lpips < self.best_lpips:
            self.best_lpips = val_lpips
            best_lpips_path = self.ckpt_dir / "best_lpips_model.pth"
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'lpips': val_lpips, 'psnr': val_psnr}, best_lpips_path)
            print(f" [CHECKPOINT] Saved new best LPIPS model ({val_lpips:.4f}) -> {best_lpips_path.name}")


class AdaIRValidationCallback(Callback):
    """PyTorch Lightning Callback for per-epoch metric evaluation and checkpoint saving."""
    def __init__(self, val_loader: DataLoader, tracker: ValidationMetricsTracker):
        super().__init__()
        self.val_loader = val_loader
        self.tracker = tracker
        self.epoch_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
        self.epoch_losses.append(loss)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        avg_loss = float(np.mean(self.epoch_losses)) if self.epoch_losses else 0.0
        self.epoch_losses.clear()

        device = pl_module.device
        self.tracker.device = device
        val_psnr, val_ssim, val_lpips = self.tracker.evaluate(pl_module.net, self.val_loader)

        self.tracker.record_epoch(epoch, avg_loss, val_psnr, val_ssim, val_lpips, pl_module.net)


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=opt.epochs)
        return [optimizer], [scheduler]


def main():
    print("=" * 70)
    print("🚀 Starting AdaIR Training & Validation Loop")
    print(f"  Input Folder: {opt.input_dir} | Target Folder: {opt.target_dir}")
    print(f"  Epochs: {opt.epochs} | Batch Size: {opt.batch_size} | LR: {opt.lr}")
    print(f"  Checkpoints Dir: {opt.ckpt_dir}")
    print(f"  Metrics CSV:     {opt.metrics_file}")
    print("=" * 70)

    device_str = "gpu" if torch.cuda.is_available() and opt.num_gpus > 0 else "cpu"
    num_devices = opt.num_gpus if device_str == "gpu" else 1

    trainset = AdaIRTrainDataset(opt)
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
        drop_last=True, num_workers=opt.num_workers
    )

    val_dataset = GenericPairedValDataset(opt.val_dir, input_dir_name=opt.input_dir, target_dir_name=opt.target_dir)
    if len(val_dataset) == 0:
        print(f"[Notice] No validation images found in '{opt.val_dir}'. Using subset of training data for validation metrics.")
        val_loader = trainloader
    else:
        val_loader = DataLoader(val_dataset, batch_size=opt.val_batch_size, shuffle=False, num_workers=opt.num_workers)

    tracker = ValidationMetricsTracker(
        device=torch.device('cuda:0' if device_str == 'gpu' else 'cpu'),
        ckpt_dir=opt.ckpt_dir,
        metrics_file=opt.metrics_file
    )
    val_callback = AdaIRValidationCallback(val_loader, tracker)

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    model = AdaIRModel()

    trainer_kwargs = {
        'max_epochs': opt.epochs,
        'accelerator': device_str,
        'devices': num_devices,
        'logger': logger,
        'callbacks': [val_callback]
    }
    if device_str == 'gpu' and num_devices > 1:
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()