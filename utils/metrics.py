from typing import Dict, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _as_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    return image


def image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    lpips_fn: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Optional[float]]:
    """Compute restoration metrics for two same-sized grayscale images."""
    prediction = _as_float_image(prediction)
    target = _as_float_image(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target must have the same shape, got "
            f"{prediction.shape} and {target.shape}."
        )

    data_range = float(max(prediction.max(), target.max()) - min(prediction.min(), target.min()))
    if data_range <= 0:
        data_range = 1.0

    values: Dict[str, Optional[float]] = {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=data_range)),
        "ssim": float(structural_similarity(target, prediction, data_range=data_range)),
        "lpips": None,
    }

    if lpips_fn is None:
        return values

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denom = float(max(prediction.max(), target.max()))
    if denom <= 0:
        denom = 1.0

    pred_tensor = torch.from_numpy(prediction / denom).float().unsqueeze(0).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target / denom).float().unsqueeze(0).unsqueeze(0).to(device)
    pred_tensor = pred_tensor.repeat(1, 3, 1, 1) * 2 - 1
    target_tensor = target_tensor.repeat(1, 3, 1, 1) * 2 - 1

    with torch.no_grad():
        values["lpips"] = float(lpips_fn(pred_tensor, target_tensor).item())

    return values
