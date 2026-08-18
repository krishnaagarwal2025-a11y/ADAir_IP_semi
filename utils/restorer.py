import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image

from models.adair_semicon import AdaIR_SR
from utils.metrics import image_metrics


class ImageRestorer:
    """Load AdaIR and restore noisy low-resolution grayscale images."""

    def __init__(
        self,
        config_path: str = "configs/train.yaml",
        checkpoint_path: Optional[str] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdaIR_SR(
            pretrained_path=self.config["model"].get("pretrained_path"),
            in_channels=self.config["model"]["in_channels"],
            upscale=self.config["model"]["upscale"],
        ).to(self.device)

        ckpt = checkpoint_path or os.environ.get("ADAIR_CHECKPOINT")
        if not ckpt:
            ckpt = os.path.join(
                self.config["training"]["save_dir"],
                f"{self.config['experiment_name']}.pth",
            )

        self.checkpoint_path = ckpt
        self.checkpoint_loaded = False
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device)
            )
            self.checkpoint_loaded = True

        self.model.eval()
        self.lpips_fn = None
        try:
            import lpips

            self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()
        except Exception:
            self.lpips_fn = None

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                return Image.fromarray(image, mode="L")
            return Image.fromarray(image).convert("L")
        if image.mode != "L":
            return image.convert("L")
        return image

    @staticmethod
    def _to_display_array(gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32)
        max_val = float(gray.max())
        if max_val > 0:
            gray = gray / max_val
        return (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    def _restore_array(self, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        noisy = np.asarray(noisy, dtype=np.float32)
        if noisy.ndim == 3:
            noisy = np.squeeze(noisy)
        if noisy.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale .npy array, got shape {noisy.shape}.")

        max_val = float(noisy.max())
        if max_val <= 0:
            raise ValueError("Input array appears to be empty (all zeros).")

        normalized = noisy / max_val
        tensor = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            pred = self.model(tensor)
            pred = torch.clamp(pred, 0, 1).squeeze().cpu().numpy() * max_val

        input_display = self._to_display_array(noisy)
        restored_display = self._to_display_array(pred)

        upscale = self.config["model"]["upscale"]
        status = (
            f"Input: {noisy.shape[1]}x{noisy.shape[0]} -> "
            f"Output: {pred.shape[1]}x{pred.shape[0]} ({upscale}x upscale). "
            f"Device: {self.device}."
        )
        if self.checkpoint_loaded:
            status += f" Checkpoint: {self.checkpoint_path}"
        else:
            status += (
                " Warning: no trained checkpoint found - output uses an untrained model. "
                f"Place weights at `{self.checkpoint_path}` or set ADAIR_CHECKPOINT."
            )

        return input_display, restored_display, pred.astype(np.float32), status

    def restore(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Run restoration on an uploaded image.

        Returns (input_display, restored_display, status_message) as uint8 arrays.
        """
        pil = self._to_pil(image)
        noisy = np.array(pil, dtype=np.float32)
        input_display, restored_display, _, status = self._restore_array(noisy)
        return input_display, restored_display, status

    def restore_with_metrics(
        self,
        image: Union[Image.Image, np.ndarray],
        reference: Optional[Union[Image.Image, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, object]]:
        input_display, restored_display, status = self.restore(image)

        params: Dict[str, object] = {
            "device": str(self.device),
            "checkpoint": self.checkpoint_path if self.checkpoint_loaded else "not loaded",
            "checkpoint_loaded": self.checkpoint_loaded,
            "upscale": self.config["model"]["upscale"],
            "input_size": f"{input_display.shape[1]}x{input_display.shape[0]}",
            "output_size": f"{restored_display.shape[1]}x{restored_display.shape[0]}",
            "psnr": None,
            "ssim": None,
            "lpips": None,
        }

        if reference is None:
            params["metrics_note"] = "Upload a ground-truth/reference image to compute PSNR, SSIM, and LPIPS."
            return input_display, restored_display, status, params

        ref_pil = self._to_pil(reference)
        reference_array = np.array(ref_pil, dtype=np.float32)
        if reference_array.shape != restored_display.shape:
            ref_pil = ref_pil.resize(
                (restored_display.shape[1], restored_display.shape[0]),
                Image.Resampling.BICUBIC,
            )
            reference_array = np.array(ref_pil, dtype=np.float32)
            params["reference_resized"] = True
        else:
            params["reference_resized"] = False

        metrics = image_metrics(
            restored_display.astype(np.float32),
            reference_array,
            lpips_fn=self.lpips_fn,
            device=self.device,
        )
        params.update(metrics)
        if self.lpips_fn is None:
            params["lpips_note"] = "Install lpips to enable LPIPS: pip install lpips"

        return input_display, restored_display, status, params

    def restore_npy(
        self,
        npy_path: str,
        reference_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, object]]:
        noisy = np.load(npy_path).astype(np.float32)
        input_display, restored_display, restored_array, status = self._restore_array(noisy)

        params: Dict[str, object] = {
            "device": str(self.device),
            "checkpoint": self.checkpoint_path if self.checkpoint_loaded else "not loaded",
            "checkpoint_loaded": self.checkpoint_loaded,
            "upscale": self.config["model"]["upscale"],
            "input_file": os.path.basename(npy_path),
            "input_shape": list(noisy.shape),
            "output_shape": list(restored_array.shape),
            "input_dtype": str(noisy.dtype),
            "output_dtype": str(restored_array.dtype),
            "input_min": float(noisy.min()),
            "input_max": float(noisy.max()),
            "output_min": float(restored_array.min()),
            "output_max": float(restored_array.max()),
            "psnr": None,
            "ssim": None,
            "lpips": None,
        }

        if not reference_path:
            params["metrics_note"] = "Upload a matching ground-truth .npy file to compute PSNR, SSIM, and LPIPS."
            return input_display, restored_display, restored_array, status, params

        reference = np.load(reference_path).astype(np.float32)
        if reference.ndim == 3:
            reference = np.squeeze(reference)
        metrics = image_metrics(
            restored_array,
            reference,
            lpips_fn=self.lpips_fn,
            device=self.device,
        )
        params.update(metrics)
        params["reference_file"] = os.path.basename(reference_path)
        params["reference_shape"] = list(reference.shape)
        if self.lpips_fn is None:
            params["lpips_note"] = "Install lpips to enable LPIPS: pip install lpips"

        return input_display, restored_display, restored_array, status, params
