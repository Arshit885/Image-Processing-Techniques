# Image Processing Techniques — Digital Image Processing Pipeline

**Student Name:** Arshit  
**University Roll No.:** 2420678  
**Course:** B.Tech — Artificial Intelligence & Data Science (AI & DS)  
**Department:** CSE-APEX, Chandigarh Engineering College, Jhanjeri, Mohali  
**Affiliated University:** I.K. Gujral Punjab Technical University, Jalandhar  

---

## Project Overview
This repository implements an end-to-end **Digital Image Processing Pipeline** in Python aligned with the academic project report *"Image Processing Techniques"*. It covers the core stages of image processing:

1. **Image Acquisition & Representation (`src/preprocessing.py`)**: Loading digital images as 2D/3D NumPy pixel arrays $f(x, y)$, resizing, normalization, and colour-space conversion (RGB, Grayscale, Binary).
2. **Image Enhancement (`src/enhancement.py`)**: Brightness/contrast adjustment, Contrast Stretching, and Histogram Equalization.
3. **Image Filtering (`src/filtering.py`)**: Mean (Box) smoothing, Gaussian smoothing, Median noise removal, and Laplacian/Unsharp-mask sharpening.
4. **Image Segmentation (`src/segmentation.py`)**: Global thresholding, Otsu's automatic thresholding, and adaptive thresholding.
5. **Edge Detection & Feature Extraction (`src/features.py`)**: Prewitt & Sobel gradient operators, Canny edge detection, corner detection, and statistical texture/intensity feature extraction.
6. **Quantitative Evaluation (`src/metrics.py`)**: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
7. **Interactive Web Dashboard (`app.py`)**: A clean Streamlit web UI to upload any image, apply techniques interactively, compare results side-by-side, and view evaluation metrics.

---

## Folder Structure
```text
Image-Processing-Techniques-Project/
├── app.py                      # Interactive Streamlit Web Application
├── main.py                     # CLI Pipeline Runner & Batch Output Generator
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── .gitignore                  # Git Ignore Rules
├── data/
│   └── sample_images/          # Sample test images
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Stage 1: Acquisition, Resizing, Grayscale, Binary
│   ├── enhancement.py          # Stage 2: Contrast Stretching, Histogram Equalization
│   ├── filtering.py            # Stage 3: Mean, Gaussian, Median, Sharpening Filters
│   ├── segmentation.py         # Stage 4: Global & Otsu Thresholding
│   ├── features.py             # Stage 5: Sobel, Prewitt, Canny Edge & Feature Extraction
│   └── metrics.py              # Stage 6: MSE, PSNR, SSIM Evaluation Metrics
└── outputs/                    # Generated output images & comparison grids
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Image-Processing-Techniques-Project.git
cd Image-Processing-Techniques-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Command-Line Pipeline (Generates All Outputs)
```bash
python main.py
```

### 4. Launch the Interactive Web App (For Viva / Demo)
```bash
streamlit run app.py
```

---

## References
1. Rafael C. Gonzalez and Richard E. Woods, *Digital Image Processing*, 4th Edition, Pearson.
2. OpenCV Documentation: https://docs.opencv.org/
3. scikit-image Documentation: https://scikit-image.org/docs/stable/
4. NumPy Documentation: https://numpy.org/doc/
