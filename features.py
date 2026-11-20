import cv2
import numpy as np
from src.preprocessing import to_grayscale


def sobel_edges(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using horizontal and vertical Sobel kernels."""
    gray = to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip((magnitude / (np.max(magnitude) + 1e-8)) * 255.0, 0, 255)
    return magnitude.astype(np.uint8)


def prewitt_edges(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Prewitt convolution masks."""
    gray = to_grayscale(image).astype(np.float32)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kernel_x)
    gy = cv2.filter2D(gray, -1, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.clip((magnitude / (np.max(magnitude) + 1e-8)) * 255.0, 0, 255)
    return magnitude.astype(np.uint8)


def canny_edges(image: np.ndarray, low_thresh: int = 60, high_thresh: int = 150) -> np.ndarray:
    """Multi-stage Canny edge detector with non-maximum suppression and hysteresis."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(blurred, low_thresh, high_thresh)


def detect_harris_corners(image: np.ndarray, threshold_ratio: float = 0.02) -> tuple:
    """Detect structural corners using Harris corner response."""
    gray = np.float32(to_grayscale(image))
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    corner_mask = dst > threshold_ratio * dst.max()
    vis[corner_mask] = [255, 0, 0]
    corner_count = int(np.sum(corner_mask))
    return vis, corner_count


def extract_statistical_features(image: np.ndarray) -> dict:
    """Extract quantitative intensity, edge density, and corner descriptors."""
    gray = to_grayscale(image)
    edges = canny_edges(image)
    _, corner_count = detect_harris_corners(image)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

    return {
        "mean_intensity": round(float(np.mean(gray)), 3),
        "std_intensity": round(float(np.std(gray)), 3),
        "shannon_entropy": round(entropy, 3),
        "edge_pixel_ratio_%": round(float(np.mean(edges > 0) * 100.0), 3),
        "detected_corner_pixels": corner_count,
    }
