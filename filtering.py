import cv2
import numpy as np


def mean_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply spatial averaging (Box) smoothing filter."""
    k = max(3, kernel_size | 1)
    return cv2.blur(image, (k, k))


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Apply Gaussian smoothing filter to suppress high-frequency Gaussian noise."""
    k = max(3, kernel_size | 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply non-linear Median filter effective against salt-and-pepper impulse noise."""
    k = max(3, kernel_size | 1)
    return cv2.medianBlur(image, k)


def sharpen_filter(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Sharpen edges using Unsharp Masking."""
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
