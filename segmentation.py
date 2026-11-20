import cv2
import numpy as np
from src.preprocessing import to_grayscale


def global_thresholding(image: np.ndarray, thresh_val: int = 127) -> np.ndarray:
    """Segment foreground and background using a fixed global intensity threshold."""
    gray = to_grayscale(image)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return mask


def otsu_thresholding(image: np.ndarray) -> tuple:
    """Automatically determine optimal threshold using Otsu's intra-class variance minimization."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    optimal_thresh, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(optimal_thresh), mask


def adaptive_thresholding(image: np.ndarray, block_size: int = 15, c_val: int = 4) -> np.ndarray:
    """Local adaptive Gaussian thresholding for images with uneven illumination."""
    gray = to_grayscale(image)
    b = max(3, block_size | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c_val
    )
