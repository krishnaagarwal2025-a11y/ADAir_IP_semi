import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from models.adair_semicon import AdaIR_SR
import yaml
from utils.metrics import image_metrics

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, help='Path to specific NoisyLR .npy file')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--limit', type=int, default=0, help='Number of validation files to process when --input is not set. 0 means all files.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaIR_SR(
        in_channels=config['model']['in_channels'],
        upscale=config['model']['upscale']
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)

    def process_file(n_path):
        noisy = np.load(n_path).astype(np.float32)
        gt_path = n_path.replace("NoisyLR", "GT")
        has_gt = os.path.exists(gt_path)
        
        gt = np.load(gt_path).astype(np.float32) if has_gt else None
        max_val = noisy.max()
        if has_gt: max_val = max(max_val, gt.max())
            
        n_input = (noisy / max_val) if max_val > 0 else noisy
        n_tensor = torch.from_numpy(n_input).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(n_tensor)
            pred = torch.clamp(pred, 0, 1).squeeze().cpu().numpy() * max_val
            
        stem = os.path.splitext(os.path.basename(n_path))[0]
        np.save(os.path.join(args.output_dir, f'{stem}_restored.npy'), pred.astype(np.float32))

        metrics_text = ''
        if has_gt:
            metrics = image_metrics(pred, gt)
            metrics_text = (
                f" | PSNR: {metrics['psnr']:.4f} "
                f"SSIM: {metrics['ssim']:.4f}"
            )

        # Visualization
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        axes[0].imshow(noisy, cmap='gray')
        axes[0].set_title('Input (Noisy 128x128)')
        
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('Prediction (256x256)')
        
        if has_gt:
            axes[2].imshow(gt, cmap='gray')
            axes[2].set_title('Ground Truth (256x256)')
        for ax in np.ravel(axes):
            ax.axis('off')

        fig.savefig(os.path.join(args.output_dir, f'{stem}_comparison.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {stem}_restored.npy and {stem}_comparison.png{metrics_text}")

    if args.input:
        process_file(args.input)
    else:
        # Process a random file from validation set
        val_noisy_dir = config['dataset']['val_noisy_dir']
        files = glob.glob(os.path.join(val_noisy_dir, '*.npy'))
        if args.limit > 0:
            files = files[:args.limit]
        if files:
            for file_path in files:
                process_file(file_path)
        else:
            print("No test files found to infer.")

if __name__ == '__main__':
    infer()
