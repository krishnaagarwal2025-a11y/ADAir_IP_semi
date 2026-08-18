import os
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.semicon_dataset import SemiconDataset
from models.adair_semicon import AdaIR_SR
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import numpy as np
from utils.metrics import image_metrics

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def evaluate(model, config, device):
    val_noisy_dir = config["dataset"].get("val_noisy_dir")
    val_gt_dir = config["dataset"].get("val_gt_dir") or val_noisy_dir.replace("NoisyLR", "GT")
    noisy_paths = sorted(glob.glob(os.path.join(val_noisy_dir, "*.npy")))
    gt_paths = sorted(glob.glob(os.path.join(val_gt_dir, "*.npy")))

    if not noisy_paths or not gt_paths:
        return None

    totals = {"psnr": 0.0, "ssim": 0.0}
    count = 0
    model.eval()

    with torch.no_grad():
        for noisy_path, gt_path in zip(noisy_paths, gt_paths):
            noisy = np.load(noisy_path).astype(np.float32)
            gt = np.load(gt_path).astype(np.float32)
            max_val = float(max(noisy.max(), gt.max()))
            if max_val <= 0:
                max_val = 1.0

            noisy_tensor = torch.from_numpy(noisy / max_val).unsqueeze(0).unsqueeze(0).to(device)
            pred = torch.clamp(model(noisy_tensor), 0, 1).squeeze().cpu().numpy() * max_val
            values = image_metrics(pred, gt)
            totals["psnr"] += values["psnr"]
            totals["ssim"] += values["ssim"]
            count += 1

    return {name: value / count for name, value in totals.items()}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    train_dataset = SemiconDataset(
        noisy_dir=config['dataset']['train_noisy_dir'],
        gt_dir=config['dataset']['train_gt_dir'],
        patch_size=config['dataset']['patch_size'],
        phase='train'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True, 
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

    # Model
    model = AdaIR_SR(
        pretrained_path=config['model'].get('pretrained_path'),
        in_channels=config['model']['in_channels'],
        upscale=config['model']['upscale']
    ).to(device)

    # Loss
    if config['training']['loss'] == 'CharbonnierLoss':
        criterion = CharbonnierLoss().to(device)
    else:
        criterion = nn.L1Loss().to(device) # fallback

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['T_max'])

    # AMP Scaler
    use_amp = config['training'].get('use_amp', False)
    scaler = GradScaler(enabled=use_amp)

    epochs = config['training']['epochs']
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = float("-inf")
    validate_every = int(config["training"].get("validate_every", 1))

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for noisy_img, gt_img in pbar:
            noisy_img = noisy_img.to(device)
            gt_img = gt_img.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                preds = model(noisy_img)
                loss = criterion(preds, gt_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * noisy_img.size(0)
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.6f}")

        if validate_every > 0 and epoch % validate_every == 0:
            metrics = evaluate(model, config, device)
            if metrics:
                print(f"Validation PSNR: {metrics['psnr']:.4f} SSIM: {metrics['ssim']:.4f}")
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{config['experiment_name']}_best.pth"))
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{config['experiment_name']}.pth"))
                    print(f"Saved new best checkpoint with PSNR {best_psnr:.4f}")

        if epoch % 10 == 0 or epoch == epochs:
            # Save checkpoint
            save_path = os.path.join(save_dir, f"{config['experiment_name']}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            if best_psnr == float("-inf"):
                torch.save(model.state_dict(), os.path.join(save_dir, f"{config['experiment_name']}.pth"))

if __name__ == '__main__':
    train()
