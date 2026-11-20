"""
Interactive Streamlit Web Application for Image Processing Techniques
Student: Arshit | Roll No.: 2420678 | B.Tech (AI & DS)
Chandigarh Engineering College, Jhanjeri, Mohali
"""

import os
import numpy as np
from PIL import Image
import streamlit as st

from src.preprocessing import resize_image, to_grayscale, to_binary, get_image_metadata
from src.enhancement import adjust_brightness_contrast, contrast_stretching, histogram_equalization, clahe_enhancement
from src.filtering import mean_filter, gaussian_filter, median_filter, sharpen_filter
from src.segmentation import global_thresholding, otsu_thresholding, adaptive_thresholding
from src.features import sobel_edges, prewitt_edges, canny_edges, detect_harris_corners, extract_statistical_features
from src.metrics import evaluate_all_metrics

st.set_page_config(page_title="Image Processing Techniques — Arshit", layout="wide")

st.title("Digital Image Processing Studio")
st.caption("Prepared by **Arshit** (Roll No: **2420678**) | B.Tech (AI & DS) | CEC, Jhanjeri, Mohali")

st.sidebar.header("1. Input Image")
uploaded = st.sidebar.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
sample_choice = st.sidebar.selectbox(
    "Or choose a built-in sample image:",
    ["sample_scene.png", "low_contrast_scan.png"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    raw_img = np.array(pil_img)
else:
    sample_path = os.path.join("data", "sample_images", sample_choice)
    raw_img = np.array(Image.open(sample_path).convert("RGB"))

img = resize_image(raw_img, 512, 512)

st.sidebar.header("2. Select Processing Stage")
stage = st.sidebar.radio(
    "Pipeline Stage",
    [
        "Preprocessing & Colour Space",
        "Image Enhancement",
        "Spatial Filtering",
        "Image Segmentation",
        "Edge Detection & Feature Extraction",
    ],
)

processed = img.copy()
info_note = ""

if stage == "Preprocessing & Colour Space":
    op = st.sidebar.selectbox("Technique", ["Original RGB", "Grayscale (8-bit)", "Binary Thresholding"])
    if op == "Grayscale (8-bit)":
        processed = to_grayscale(img)
    elif op == "Binary Thresholding":
        t = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        processed = to_binary(img, t)

elif stage == "Image Enhancement":
    op = st.sidebar.selectbox(
        "Technique",
        ["Brightness & Contrast", "Contrast Stretching", "Histogram Equalization", "CLAHE (Adaptive)"],
    )
    if op == "Brightness & Contrast":
        alpha = st.sidebar.slider("Contrast (alpha)", 0.5, 2.5, 1.2, 0.1)
        beta = st.sidebar.slider("Brightness (beta)", -80, 80, 15, 5)
        processed = adjust_brightness_contrast(img, alpha, beta)
    elif op == "Contrast Stretching":
        processed = contrast_stretching(img)
    elif op == "Histogram Equalization":
        processed = histogram_equalization(img)
    elif op == "CLAHE (Adaptive)":
        clip = st.sidebar.slider("Clip Limit", 1.0, 6.0, 2.0, 0.5)
        processed = clahe_enhancement(img, clip_limit=clip)

elif stage == "Spatial Filtering":
    op = st.sidebar.selectbox(
        "Technique",
        ["Mean (Box) Smoothing", "Gaussian Smoothing", "Median Filter", "Unsharp Mask Sharpening"],
    )
    if op == "Mean (Box) Smoothing":
        k = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
        processed = mean_filter(img, k)
    elif op == "Gaussian Smoothing":
        k = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
        sigma = st.sidebar.slider("Sigma", 0.5, 4.0, 1.2, 0.1)
        processed = gaussian_filter(img, k, sigma)
    elif op == "Median Filter":
        k = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
        processed = median_filter(img, k)
    elif op == "Unsharp Mask Sharpening":
        strength = st.sidebar.slider("Sharpening Strength", 0.2, 2.5, 1.0, 0.1)
        processed = sharpen_filter(img, strength)

elif stage == "Image Segmentation":
    op = st.sidebar.selectbox("Technique", ["Global Thresholding", "Otsu's Automatic Thresholding", "Adaptive Gaussian"])
    if op == "Global Thresholding":
        t = st.sidebar.slider("Threshold Value", 0, 255, 127)
        processed = global_thresholding(img, t)
    elif op == "Otsu's Automatic Thresholding":
        opt_t, processed = otsu_thresholding(img)
        info_note = f"Otsu Optimal Threshold Computed: **{opt_t:.1f}**"
    elif op == "Adaptive Gaussian":
        b = st.sidebar.slider("Block Size", 3, 31, 15, 2)
        processed = adaptive_thresholding(img, b)

elif stage == "Edge Detection & Feature Extraction":
    op = st.sidebar.selectbox("Technique", ["Canny Edge Detector", "Sobel Gradient", "Prewitt Gradient", "Harris Corner Detection"])
    if op == "Canny Edge Detector":
        low = st.sidebar.slider("Low Threshold", 10, 150, 60)
        high = st.sidebar.slider("High Threshold", 50, 250, 150)
        processed = canny_edges(img, low, high)
    elif op == "Sobel Gradient":
        processed = sobel_edges(img)
    elif op == "Prewitt Gradient":
        processed = prewitt_edges(img)
    elif op == "Harris Corner Detection":
        processed, corners = detect_harris_corners(img)
        info_note = f"Detected Corner Pixels: **{corners}**"

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Image")
    st.image(img, use_container_width=True)
with col2:
    st.subheader("Processed Output")
    st.image(processed, use_container_width=True)

if info_note:
    st.info(info_note)

st.divider()
m_col1, m_col2, m_col3 = st.columns(3)
metrics = evaluate_all_metrics(img, processed)
with m_col1:
    st.subheader("Image Metadata")
    st.json(get_image_metadata(img))
with m_col2:
    st.subheader("Quality Metrics (vs Input)")
    st.json(metrics)
with m_col3:
    st.subheader("Statistical Features")
    st.json(extract_statistical_features(processed))
