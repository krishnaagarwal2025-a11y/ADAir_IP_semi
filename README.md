# AdaIR - Semiconductor Wafer Image Restoration

This repository contains the training and inference pipeline for fine-tuning the AdaIR framework on semiconductor wafer images. The project handles 2x super-resolution and denoising of single-channel (grayscale) wafer patches.

## Project Structure

* **`train.py`**: The core training script implementing Charbonnier loss and AMP (Automatic Mixed Precision) for efficiency.
* **`validate.py`**: Evaluation script computing standard validation metrics (PSNR, SSIM, and optionally LPIPS).
* **`inference.py`**: Inference module for restoring NoisyLR patches and comparing outputs alongside the Ground Truth patches.
* **`datasets/semicon_dataset.py`**: The PyTorch `Dataset` class handling dataset loading, normalization, and basic augmentations.
* **`models/adair_semicon.py`**: The neural network structure that adapts the AdaIR CNN backbone for semantic restoration tasks.
* **`configs/train.yaml`**: The central configuration file driving the entire pipeline (epochs, batch size, learning rates, data paths).

## Usage

1. Configure your dataset paths in `configs/train.yaml`.
2. Run the main training loop. It validates every epoch when `dataset.val_gt_dir` exists and writes the best PSNR checkpoint to `checkpoints/adair_semicon_baseline_best.pth` and `checkpoints/adair_semicon_baseline.pth`:
   ```bash
   python train.py --config configs/train.yaml
   ```
3. Run inference on tested images:
   ```bash
   python inference.py --checkpoint checkpoints/adair_semicon_baseline.pth 
   ```
4. Validate PSNR, SSIM, and LPIPS:
   ```bash
   python validate.py --checkpoint checkpoints/adair_semicon_baseline.pth
   ```

## Web UI (upload & restore)

Launch the Gradio page to upload an image and get the restored output. Upload an optional ground-truth/reference image to compute PSNR, SSIM, and LPIPS:

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860 in your browser. Optional flags:

```bash
python app.py --checkpoint checkpoints/adair_semicon_baseline.pth --port 7860
```

Set `ADAIR_CHECKPOINT` env var to point at your trained weights if the default path is missing.
