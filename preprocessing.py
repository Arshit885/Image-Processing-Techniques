import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and convert from BGR to RGB."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, width: int = 512, height: int = 512) -> np.ndarray:
    """Resize image to uniform spatial dimensions."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to 8-bit single-channel Grayscale (0-255)."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_binary(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert an image into a Binary image (0 and 255)."""
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel intensities to [0.0, 1.0] float32 range."""
    return image.astype(np.float32) / 255.0


def get_image_metadata(image: np.ndarray) -> dict:
    """Return basic spatial and intensity properties of f(x, y)."""
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_intensity": int(np.min(image)),
        "max_intensity": int(np.max(image)),
        "mean_intensity": round(float(np.mean(image)), 2),
    }
