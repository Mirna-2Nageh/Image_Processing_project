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
cd image-processing-compression-tool```
Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run app.py


Upload an image from the sidebar.

Use the tabs to apply transformations, filters, histogram equalization, and compression.

Adjust sliders and settings interactively.

View original vs processed images in real-time.

Download processed images using the "Advanced" tab.

Project Structure
image-processing-compression-tool/
│
├── app.py                 # Main Streamlit app
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── utils/                 # Helper functions (image ops, compression)
└── assets/                # Optional: example images or icons

Operations & Functionalities
Category	Operations
Basic	Grayscale, Binary, PSNR/MSE
Transformations	Translation, Scaling, Rotation, Shear
Interpolation	Nearest, Bilinear, Bicubic
Histogram	Compute, Equalize, Visualize
Filters	Gaussian, Median, Laplacian, Sobel, Gradient
Image Operations	Invert, Blend, Brightness/Contrast
Cropping / Export	ROI selection, save locally
Compression Techniques	Huffman, RLE, LZW, DCT, Wavelet, Arithmetic coding
Future Enhancements

Implement Golomb–Rice coding and Bit-plane coding.

Real-time interactive ROI selection using mouse (instead of sliders).

Advanced image blending with masks.

Batch processing of multiple images.

GPU acceleration for faster filtering and transformations.

License

This project is licensed under the MIT License.

Screenshots

(Add screenshots here showing original vs processed images, filters, histogram, compression results.)

Author

Mirna Nageh Botros
AI | ML Engineer | Data Scientist
Email: mirnanagehb.w@gmail.com


LinkedIn: linkedin.com/in/mirnanageh
