
# 🎨 Image Processing and Compression Tool

## ✨ Overview

This project is a powerful, interactive tool built using **Streamlit** for real-time image manipulation, analysis, and compression. It provides a user-friendly interface to apply various digital image processing techniques, visualize results, and compare advanced compression algorithms.

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

You must have **Python 3.x** installed on your system.

### 1\. Clone the Repository

Open your terminal and clone the project repository:

```bash
git clone https://github.com/yourusername/image-processing-compression-tool.git
cd image-processing-compression-tool
```

### 2\. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3\. Usage

Run the Streamlit application from the terminal:

```bash
streamlit run app.py
```

The application will automatically open in your web browser.

-----

## 💡 How to Use

1.  **Upload** an image using the file uploader in the sidebar.
2.  Navigate between the different **tabs** to access various processing categories.
3.  **Adjust sliders and settings** interactively. The processed image and relevant metrics (like PSNR) will update in real-time.
4.  View the **original vs. processed** images side-by-side.
5.  Use the **Advanced** tab to download the final processed image.

-----

## 📋 Operations & Functionalities

The tool offers a wide range of features categorized for easy access:

| Category | Operations | Description |
| :--- | :--- | :--- |
| **Basic Analysis** | Grayscale, Binary, PSNR/MSE | Core color space conversions and quality assessment metrics. |
| **Transformations** | Translation, Scaling, Rotation, Shear | Geometric modifications to the image structure. |
| **Interpolation** | Nearest Neighbor, Bilinear, Bicubic | Techniques for resampling image data during scaling. |
| **Histogram** | Compute, Equalize, Visualize | Analysis of pixel distribution and equalization for contrast enhancement. |
| **Filters** | Gaussian, Median, Laplacian, Sobel, Gradient | Spatial filtering for noise reduction, smoothing, and edge detection. |
| **Image Operations** | Invert, Blend, Brightness/Contrast | Adjustments to pixel intensity and combining images. |
| **Cropping / Export** | ROI selection, Save Locally | Defining a Region of Interest and downloading the final output. |
| **Compression** | Huffman, RLE, LZW, DCT, Wavelet, Arithmetic | Advanced data reduction techniques for efficient storage. |

-----

## 📂 Project Structure

```
image-processing-compression-tool/
│
├── app.py                 # Main Streamlit application and UI logic
├── README.md              # Project documentation (this file)
├── requirements.txt       # List of Python dependencies
├── utils/                 # Module containing core image processing and compression logic
│   ├── image_ops.py
│   └── compression_algs.py
└── assets/                # Optional: Example input images or icons
```

-----

## ⏭️ Future Enhancements

The following features are planned for future development:

  * Implement advanced coding techniques: **Golomb–Rice coding** and **Bit-plane coding**.
  * Add **real-time interactive ROI selection** using mouse clicks/drag, replacing the current slider-based method.
  * Develop advanced image blending capabilities using **masking**.
  * Introduce **batch processing** functionality for handling multiple images.
  * Explore **GPU acceleration** (e.g., via CUDA/CuPy) for faster filtering and transformations.

-----

## 📜 License

This project is licensed under the **MIT License**. See the repository for details.

-----

## 👩‍💻 Author

**Mirna Nageh Botros**

  * **Role:** AI | ML Engineer | Data Scientist
  * **Email:** mirnanagehb.w@gmail.com
  * **LinkedIn:** [linkedin.com/in/mirnanageh](https://www.linkedin.com/in/mirna-nageh-botros/))
