# 🖼️ Image Processing & Compression Tool

A **Python-based interactive GUI application** for performing a wide range of **image processing operations and compression techniques**. Built using **Streamlit**, **OpenCV**, **NumPy**, **PyWavelets**, and other core libraries.

---

## **Table of Contents**

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Operations & Functionalities](#operations--functionalities)  
- [Future Enhancements](#future-enhancements)  
- [License](#license)

---

## **Features**

1. **Image Reading & Display**  
   - Upload images in PNG, JPEG, BMP formats.  
   - Display image information: resolution, color space, channels, file size.

2. **Basic Operations**  
   - Grayscale conversion.  
   - Binary image conversion with **mean threshold** or **Otsu's method**.  
   - PSNR and MSE calculation for quality assessment.

3. **Affine Transformations**  
   - Translation (x, y).  
   - Scaling (x, y).  
   - Rotation.  
   - Shear (x, y).  
   - Interactive sliders for parameter adjustments.

4. **Interpolation / Resolution Enhancement**  
   - Resize using **Nearest Neighbor**, **Bilinear**, and **Bicubic** interpolation.

5. **Histogram Analysis**  
   - Compute and display grayscale histograms.  
   - Histogram equalization for contrast enhancement.  
   - PSNR/MSE comparison with original image.

6. **Filters**  
   - Low-pass: Gaussian (adjustable kernel/sigma), Median.  
   - High-pass: Laplacian, Sobel, Gradient filters.

7. **Image Operations**  
   - Inversion / Negative image.  
   - Image blending.  
   - Brightness and contrast adjustment.  

8. **Cropping & Export**  
   - Interactive cropping with preview.  
   - Save processed images locally.

9. **Compression Techniques**  
   - **Huffman coding**  
   - **Run-Length Encoding (RLE)**  
   - **LZW coding**  
   - **Discrete Cosine Transform (DCT)**  
   - **Wavelet coding**  
   - **Arithmetic coding** (simulation)  
   - Compression ratio calculation.

10. **User Interface**  
    - Clean, organized tabs: Basic, Transformations, Histogram, Filters, Compression, Advanced, Image Ops.  
    - Interactive sliders, buttons, and checkboxes for dynamic processing.  
    - Side-by-side display of original vs processed images.

---

## **Requirements**

- Python 3.9+  
- Streamlit  
- OpenCV (`opencv-python`)  
- NumPy  
- Pillow (`PIL`)  
- PyWavelets (`pywt`)  
- scikit-image (`skimage`)  
- Matplotlib  

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/image-processing-compression-tool.git
cd image-processing-compression-tool
