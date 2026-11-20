import cv2
import numpy as np
from src.preprocessing import to_grayscale


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
    """
    Linear intensity transformation: g(x, y) = alpha * f(x, y) + beta
    alpha > 1 increases contrast; beta adjusts brightness.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def contrast_stretching(image: np.ndarray) -> np.ndarray:
    """Expand pixel intensity values across the full [0, 255] dynamic range."""
    gray = to_grayscale(image)
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val == min_val:
        return gray.copy()
    stretched = ((gray.astype(np.float32) - min_val) / (max_val - min_val)) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Redistribute pixel intensities using cumulative distribution function (CDF)."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    # Apply equalization on Luminance channel in YCrCb space to preserve colours
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if len(image.shape) == 2:
        return clahe.apply(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
