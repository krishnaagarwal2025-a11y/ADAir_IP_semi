import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.dataset_utils import AdaIRTrainDataset, crop_img, load_image_or_npy, is_valid_image_file
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

tm_psnr = None
tm_ssim = None
try:
    import torchmetrics
    try:
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio as tm_psnr,
            structural_similarity_index_measure as tm_ssim,
        )
    except ImportError:
        from torchmetrics.functional import (
            peak_signal_noise_ratio as tm_psnr,
            structural_similarity_index_measure as tm_ssim,
        )
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


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

        # Absolute normalization: both input and target must be float32 [0.0, 1.0]
        in_img = np.asarray(in_img, dtype=np.float32)
        tg_img = np.asarray(tg_img, dtype=np.float32)
        if tg_img.max() > 1.0:
            tg_img = tg_img / 255.0
        if in_img.max() > 1.0:
            in_img = in_img / 255.0

        # Deterministic alignment crop only (no random crop, flips, or CutBlur)
        in_img = crop_img(in_img, base=16)
        tg_img = crop_img(tg_img, base=16)

        in_tensor = torch.from_numpy(np.ascontiguousarray(in_img)).permute(2, 0, 1).float()
        tg_tensor = torch.from_numpy(np.ascontiguousarray(tg_img)).permute(2, 0, 1).float()

        return [in_path.stem], in_tensor, tg_tensor


class ValidationMetricsTracker:
    """Computes PSNR, SSIM, and LPIPS metrics and manages CSV & checkpoint logging."""
    def __init__(self, device: torch.device, ckpt_dir: str, metrics_file: str, max_epochs: int = 150):
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.metrics_file = Path(metrics_file)
        self.max_epochs = max_epochs
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_psnr = -float('inf')
        self.best_psnr_epoch = 0
        self.best_lpips = float('inf')
        self.best_lpips_epoch = 0

        self.lpips_fn = None
        if HAS_LPIPS:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
                self.lpips_fn.eval()
            except Exception as e:
                print(f"[Warning] Failed to initialize LPIPS: {e}")

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_loader: DataLoader, max_samples: int = 0) -> tuple:
        """Evaluate PSNR/SSIM/LPIPS on up to max_samples images (0 = all)."""
        model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        n_evaluated = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if max_samples > 0 and n_evaluated >= max_samples:
                break
            if len(batch) == 3:
                _, degrad_patch, clean_patch = batch
            else:
                degrad_patch, clean_patch = batch

            degrad_patch = degrad_patch.to(self.device)
            clean_patch = clean_patch.to(self.device)

            restored = model(degrad_patch)
            if restored.shape[-2:] != clean_patch.shape[-2:]:
                restored = nn.functional.interpolate(restored, size=clean_patch.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.clamp(restored, 0.0, 1.0)
            targets = torch.clamp(clean_patch, 0.0, 1.0)

            rec_np = preds.detach().cpu().numpy().transpose(0, 2, 3, 1)
            cln_np = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(rec_np.shape[0]):
                if max_samples > 0 and n_evaluated >= max_samples:
                    break
                if HAS_TORCHMETRICS and tm_psnr is not None:
                    pred_i = preds[i:i + 1]
                    target_i = targets[i:i + 1]
                    p_val = tm_psnr(pred_i, target_i, data_range=1.0).item()
                    s_val = tm_ssim(pred_i, target_i, data_range=1.0).item()
                else:
                    p_val = peak_signal_noise_ratio(cln_np[i], rec_np[i], data_range=1.0)
                    try:
                        s_val = structural_similarity(cln_np[i], rec_np[i], data_range=1.0, channel_axis=-1)
                    except TypeError:
                        s_val = structural_similarity(cln_np[i], rec_np[i], data_range=1.0, multichannel=True)

                psnr_list.append(p_val)
                ssim_list.append(s_val)
                n_evaluated += 1

            if self.lpips_fn is not None:
                # Only compute LPIPS for images we actually evaluated in this batch
                n_batch = rec_np.shape[0]
                used = min(n_batch, max(0, max_samples - (n_evaluated - n_batch))) if max_samples > 0 else n_batch
                rec_norm = preds[:used] * 2.0 - 1.0
                cln_norm = targets[:used] * 2.0 - 1.0
                lp_val = self.lpips_fn(rec_norm, cln_norm).mean().item()
                lpips_list.append(lp_val)

        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

        return avg_psnr, avg_ssim, avg_lpips

    def record_epoch(self, epoch: int, train_loss: float, val_psnr: float, val_ssim: float, val_lpips: float, model: nn.Module):
        is_best_psnr = False

        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            self.best_psnr_epoch = epoch
            is_best_psnr = True
            best_psnr_path = self.ckpt_dir / "best_psnr_model.pth"
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'psnr': val_psnr, 'ssim': val_ssim}, best_psnr_path)
            print(f"\n>>> [NEW BEST] New Best PSNR: {val_psnr:.2f} dB achieved! Saved best_psnr_model.pth")

        if val_lpips > 0 and val_lpips < self.best_lpips:
            self.best_lpips = val_lpips
            self.best_lpips_epoch = epoch
            best_lpips_path = self.ckpt_dir / "best_lpips_model.pth"
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'lpips': val_lpips, 'psnr': val_psnr}, best_lpips_path)
            print(f"\n>>> [NEW BEST] New Best LPIPS: {val_lpips:.4f} achieved! Saved best_lpips_model.pth")

        best_psnr_val = f"{self.best_psnr:.2f}" if self.best_psnr != -float('inf') else "N/A"
        print(
            f"Epoch [{epoch}/{self.max_epochs}] | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Val LPIPS: {val_lpips:.4f} | "
            f"Best PSNR: {best_psnr_val} dB"
        )

        file_exists = self.metrics_file.exists()
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'is_best_psnr'])
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_psnr:.4f}", f"{val_ssim:.4f}", f"{val_lpips:.4f}", is_best_psnr])
            f.flush()


class AdaIRValidationCallback(Callback):
    """PyTorch Lightning Callback for per-epoch metric evaluation and checkpoint saving."""
    def __init__(self, val_loader: DataLoader, tracker: ValidationMetricsTracker,
                 val_freq: int = 5, val_max_samples: int = 100):
        super().__init__()
        self.val_loader = val_loader
        self.tracker = tracker
        self.val_freq = max(1, val_freq)
        self.val_max_samples = val_max_samples
        self.epoch_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
        self.epoch_losses.append(loss)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        avg_loss = float(np.mean(self.epoch_losses)) if self.epoch_losses else 0.0
        self.epoch_losses.clear()

        # Only run validation every val_freq epochs (always run on final epoch)
        if epoch % self.val_freq != 0 and epoch != trainer.max_epochs:
            return

        device = pl_module.device
        self.tracker.device = device
        val_psnr, val_ssim, val_lpips = self.tracker.evaluate(
            pl_module.net, self.val_loader, max_samples=self.val_max_samples
        )

        self.tracker.record_epoch(epoch, avg_loss, val_psnr, val_ssim, val_lpips, pl_module.net)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 variant), better for image restoration than L1."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: 1.0 - SSIM(pred, target)."""
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        # Create 1D gaussian kernel
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2.0 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1d = gauss.unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        mu1 = nn.functional.conv2d(pred, self.window, padding=pad, groups=self.channels)
        mu2 = nn.functional.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(pred * pred, self.window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = nn.functional.conv2d(target * target, self.window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = nn.functional.conv2d(pred * target, self.window, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.ssim_loss = SSIMLoss(window_size=11, channels=3)
        self.ssim_weight = 0.2

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        if restored.shape[-2:] != clean_patch.shape[-2:]:
            restored = nn.functional.interpolate(restored, size=clean_patch.shape[-2:], mode='bilinear', align_corners=False)
        # Hybrid loss: Charbonnier + 0.2 * (1 - SSIM)
        loss_charb = self.charbonnier(restored, clean_patch)
        loss_ssim = self.ssim_loss(restored, clean_patch)
        loss = loss_charb + self.ssim_weight * loss_ssim
        self.log("train_loss", loss)
        self.log("charb_loss", loss_charb, prog_bar=False)
        self.log("ssim_loss", loss_ssim, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=opt.epochs)
        return [optimizer], [scheduler]


def main():
    print("=" * 70)
    print("[START] Starting AdaIR Training & Validation Loop")
    print(f"  Input Folder: {opt.input_dir} | Target Folder: {opt.target_dir}")
    print(f"  Scale Factor: {opt.scale}x (Target = Input * {opt.scale})")
    print(f"  Epochs: {opt.epochs} | Batch Size: {opt.batch_size} | LR: {opt.lr}")
    print(f"  Checkpoints Dir: {opt.ckpt_dir}")
    print(f"  Metrics CSV:     {opt.metrics_file}")
    if opt.max_samples:
        print(f"  Max Samples:     {opt.max_samples} pairs (subsampled dataset)")
    if opt.resume_from:
        print(f"  Resume From:     {opt.resume_from}")
    print("=" * 70)

    device_str = "gpu" if torch.cuda.is_available() and opt.num_gpus > 0 else "cpu"
    num_devices = opt.num_gpus if device_str == "gpu" else 1

    # Use persistent_workers only when num_workers > 0 (required by PyTorch)
    use_persistent = opt.num_workers > 0

    trainset = AdaIRTrainDataset(opt)
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size,
        pin_memory=True, shuffle=True, drop_last=True,
        num_workers=opt.num_workers, persistent_workers=use_persistent
    )

    val_dataset = GenericPairedValDataset(opt.val_dir, input_dir_name=opt.input_dir, target_dir_name=opt.target_dir)
    # Apply --max_samples to val dataset as well (cap to same N for consistency)
    if opt.max_samples and opt.max_samples > 0 and len(val_dataset) > opt.max_samples:
        val_dataset.pairs = val_dataset.pairs[:opt.max_samples]
        print(f"[Val] Subsampled validation set to {opt.max_samples} pairs (--max_samples={opt.max_samples})")
    if len(val_dataset) == 0:
        print(f"[Notice] No validation images found in '{opt.val_dir}'. Using unaugmented training pairs for validation metrics.")
        val_dataset = AdaIRTrainDataset(opt, augment=False)
        if opt.max_samples and opt.max_samples > 0 and len(val_dataset) > opt.max_samples:
            val_dataset.sample_ids = val_dataset.sample_ids[:opt.max_samples]
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, persistent_workers=use_persistent
    )

    tracker = ValidationMetricsTracker(
        device=torch.device('cuda:0' if device_str == 'gpu' else 'cpu'),
        ckpt_dir=opt.ckpt_dir,
        metrics_file=opt.metrics_file,
        max_epochs=opt.epochs
    )
    val_callback = AdaIRValidationCallback(
        val_loader, tracker,
        val_freq=opt.val_freq,
        val_max_samples=opt.val_max_samples
    )

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    model = AdaIRModel()

    # --- Checkpoint Resume ---
    if opt.resume_from:
        resume_path = Path(opt.resume_from)
        if not resume_path.exists():
            print(f"[WARNING] --resume_from path not found: {resume_path}. Starting from scratch.")
        else:
            ckpt = torch.load(str(resume_path), map_location='cpu')
            # Support both raw state_dicts and our {epoch, state_dict, ...} format
            state_dict = ckpt.get('state_dict', ckpt)
            # Strip Lightning 'net.' prefix if present
            if all(k.startswith('net.') for k in state_dict):
                state_dict = {k[len('net.'):]: v for k, v in state_dict.items()}
            missing, unexpected = model.net.load_state_dict(state_dict, strict=False)
            print(f"[RESUME] Loaded weights from '{resume_path.name}'")
            if missing:
                print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    trainer_kwargs = {
        'max_epochs': opt.epochs,
        'accelerator': device_str,
        'devices': num_devices,
        'logger': logger,
        'callbacks': [val_callback],
        'precision': '16-mixed' if device_str == 'gpu' else '32',  # FP16 Tensor Cores on GPU only
        'limit_train_batches': opt.limit_train_batches,
    }
    if device_str == 'gpu' and num_devices > 1:
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()