import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.preprocessing import to_grayscale


def compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE) between reference and processed image."""
    ref_g = to_grayscale(reference).astype(np.float64)
    proc_g = to_grayscale(processed).astype(np.float64)
    return round(float(np.mean((ref_g - proc_g) ** 2)), 4)


def compute_psnr(reference: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR in dB)."""
    ref_g = to_grayscale(reference)
    proc_g = to_grayscale(processed)
    if np.array_equal(ref_g, proc_g):
        return float("inf")
    return round(float(peak_signal_noise_ratio(ref_g, proc_g, data_range=255)), 4)


def compute_ssim(reference: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Structural Similarity Index Measure (SSIM in [-1, 1])."""
    ref_g = to_grayscale(reference)
    proc_g = to_grayscale(processed)
    return round(float(structural_similarity(ref_g, proc_g, data_range=255)), 4)


def evaluate_all_metrics(reference: np.ndarray, processed: np.ndarray) -> dict:
    """Return MSE, PSNR, and SSIM evaluation dictionary."""
    return {
        "MSE": compute_mse(reference, processed),
        "PSNR_dB": compute_psnr(reference, processed),
        "SSIM": compute_ssim(reference, processed),
    }
