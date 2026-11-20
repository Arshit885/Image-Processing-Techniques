"""
End-to-End Digital Image Processing Pipeline
Student: Arshit | Roll No.: 2420678 | B.Tech (AI & DS)
Chandigarh Engineering College, Jhanjeri, Mohali
"""

import os
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import load_image, resize_image, to_grayscale, to_binary, get_image_metadata
from src.enhancement import contrast_stretching, histogram_equalization, clahe_enhancement
from src.filtering import mean_filter, gaussian_filter, median_filter, sharpen_filter
from src.segmentation import otsu_thresholding, adaptive_thresholding
from src.features import sobel_edges, prewitt_edges, canny_edges, detect_harris_corners, extract_statistical_features
from src.metrics import evaluate_all_metrics


def save_rgb_or_gray(path: str, img):
    if len(img.shape) == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)


def run_pipeline(input_path: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Image Acquisition & Preprocessing
    raw_img = load_image(input_path)
    img = resize_image(raw_img, 512, 512)
    gray = to_grayscale(img)
    binary = to_binary(img)
    metadata = get_image_metadata(img)

    # Stage 2: Image Enhancement
    stretched = contrast_stretching(img)
    hist_eq = histogram_equalization(img)
    clahe_img = clahe_enhancement(img)

    # Stage 3: Image Filtering
    mean_img = mean_filter(img, kernel_size=5)
    gauss_img = gaussian_filter(img, kernel_size=5, sigma=1.2)
    median_img = median_filter(img, kernel_size=5)
    sharp_img = sharpen_filter(gauss_img, strength=1.0)

    # Stage 4: Image Segmentation
    otsu_val, otsu_mask = otsu_thresholding(gauss_img)
    adaptive_mask = adaptive_thresholding(gauss_img)

    # Stage 5: Edge Detection & Feature Extraction
    sobel_img = sobel_edges(gauss_img)
    prewitt_img = prewitt_edges(gauss_img)
    canny_img = canny_edges(gauss_img)
    corners_img, corner_count = detect_harris_corners(gauss_img)
    features = extract_statistical_features(gauss_img)

    # Stage 6: Quantitative Evaluation
    metrics_summary = {
        "Gaussian_Smoothing_vs_Raw": evaluate_all_metrics(img, gauss_img),
        "Median_Filtering_vs_Raw": evaluate_all_metrics(img, median_img),
        "Histogram_Equalization_vs_Raw": evaluate_all_metrics(img, hist_eq),
        "Sharpened_vs_Raw": evaluate_all_metrics(img, sharp_img),
    }

    # Save individual processed outputs
    outputs_map = {
        "01_original.png": img,
        "02_grayscale.png": gray,
        "03_binary.png": binary,
        "04_contrast_stretched.png": stretched,
        "05_histogram_equalized.png": hist_eq,
        "06_clahe_enhanced.png": clahe_img,
        "07_gaussian_smoothed.png": gauss_img,
        "08_median_filtered.png": median_img,
        "09_sharpened.png": sharp_img,
        "10_otsu_segmentation.png": otsu_mask,
        "11_adaptive_segmentation.png": adaptive_mask,
        "12_sobel_edges.png": sobel_img,
        "13_prewitt_edges.png": prewitt_img,
        "14_canny_edges.png": canny_img,
        "15_harris_corners.png": corners_img,
    }
    for fname, out_img in outputs_map.items():
        save_rgb_or_gray(os.path.join(output_dir, fname), out_img)

    # Generate 3x4 Summary Comparison Grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    grid_items = [
        ("1. Original Image", img, None),
        ("2. Grayscale", gray, "gray"),
        ("3. Binary Image", binary, "gray"),
        ("4. Contrast Stretched", stretched, "gray"),
        ("5. Histogram Equalization", hist_eq, None),
        ("6. Gaussian Smoothing", gauss_img, None),
        ("7. Median Filter", median_img, None),
        ("8. Unsharp Sharpening", sharp_img, None),
        (f"9. Otsu Segmentation (T={otsu_val:.0f})", otsu_mask, "gray"),
        ("10. Sobel Edges", sobel_img, "gray"),
        ("11. Canny Edges", canny_img, "gray"),
        (f"12. Harris Corners ({corner_count} px)", corners_img, None),
    ]
    for ax, (title, data, cmap) in zip(axes.flat, grid_items):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Digital Image Processing Pipeline Summary — Arshit (Roll No: 2420678)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pipeline_summary_grid.png"), dpi=200)
    plt.close(fig)

    # Save Evaluation Report JSON
    report_data = {
        "student": "Arshit",
        "roll_no": "2420678",
        "course": "B.Tech (AI & DS)",
        "image_metadata": metadata,
        "otsu_optimal_threshold": otsu_val,
        "extracted_features": features,
        "evaluation_metrics": metrics_summary,
    }
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print("Pipeline executed successfully! Outputs saved to:", output_dir)
    print(json.dumps(report_data, indent=2))


if __name__ == "__main__":
    default_sample = os.path.join("data", "sample_images", "sample_scene.png")
    run_pipeline(default_sample)
