import os
import argparse
import yaml
import glob
import numpy as np
import torch
from models.adair_semicon import AdaIR_SR
from utils.metrics import image_metrics
try:
    import lpips
except ImportError:
    lpips = None

def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to weights")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS
    if lpips is not None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    else:
        print("LPIPS package not found. Run pip install lpips")
        lpips_fn = None

    model = AdaIR_SR(
        pretrained_path=config['model'].get('pretrained_path'),
        in_channels=config['model']['in_channels'],
        upscale=config['model']['upscale']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Metrics
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    
    # Note: Requires GT files to exist in test folder for full validation
    val_noisy_dir = config['dataset']['val_noisy_dir']
    val_gt_dir = config['dataset'].get('val_gt_dir') or val_noisy_dir.replace("NoisyLR", "GT")
    
    noisy_paths = sorted(glob.glob(os.path.join(val_noisy_dir, '*.npy')))
    gt_paths = sorted(glob.glob(os.path.join(val_gt_dir, '*.npy')))
    
    if len(gt_paths) == 0:
        print("Warning: No GT files found for validation. Evaluation needs GT pairs.")
        return

    count = 0
    with torch.no_grad():
        for n_path, g_path in zip(noisy_paths, gt_paths):
            noisy = np.load(n_path).astype(np.float32)
            gt = np.load(g_path).astype(np.float32)

            max_val = max(noisy.max(), gt.max())
            noisy_norm = noisy / max_val if max_val > 0 else noisy
            
            n_tensor = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            pred_tensor = model(n_tensor)
            
            # Clamp directly after inference
            pred_tensor = torch.clamp(pred_tensor, 0, 1)
            pred = pred_tensor.squeeze().cpu().numpy() * max_val
            
            metrics = image_metrics(pred, gt, lpips_fn=lpips_fn, device=device)
            psnr_val = metrics["psnr"]
            ssim_val = metrics["ssim"]
            
            psnr_total += psnr_val
            ssim_total += ssim_val
            
            # LPIPS expects inputs from [-1, 1], shape N C H W
            if lpips_fn:
                lpips_total += metrics["lpips"]
            
            count += 1

    print(f"Validation Results over {count} images:")
    print(f"PSNR:  {psnr_total / count:.4f}")
    print(f"SSIM:  {ssim_total / count:.4f}")
    if lpips_fn:
        print(f"LPIPS: {lpips_total / count:.4f}")

if __name__ == '__main__':
    validate()
