Image Processing & Compression Tool
📌 Project Overview

This project is an interactive image processing and compression application designed to help users perform a variety of image transformations, enhancements, analysis, and compression techniques. Users can upload images, apply operations dynamically, and visualize results in real time.

The project combines image reading, geometric transformations, filtering, histogram analysis, and both lossless and lossy compression methods in a single, user-friendly interface.

🖥️ Features
Image Operations

Upload and display images (with resolution, size, and type information)

Grayscale conversion

Binary (threshold-based) conversion

Histogram computation and equalization

Affine transformations:

Translation

Scaling

Rotation

X/Y Shearing

Image cropping

Image interpolation:

Nearest-neighbor

Bilinear

Bicubic

Filtering

Low-pass filters:

Gaussian (19×19, σ=3)

Median (7×7)

High-pass filters:

Laplacian

Sobel

Gradient (first derivatives)

Image Compression Techniques

Lossless:

Huffman Coding

Golomb–Rice Coding

Arithmetic Coding

LZW Coding

Run-Length Encoding (RLE)

Symbol-Based Coding

Bit-Plane Coding

Predictive Coding

Lossy:

Block Transform (DCT)

Wavelet Coding

Interactive UI

Dynamic display of processed images

Buttons and sliders for controlling operations

Side-by-side comparison of original vs processed image

Visualization of histograms and compression performance

Creative Enhancements

Real-time filter adjustment using sliders

Compression ratio and quality metrics display (PSNR, MSE)

Multi-operation pipeline (chain transformations)

Export processed images, histograms, and compression results

📂 Project Structure
project_root/
│
├── src/                   # Source code files
│   ├── main.py            # Main application launcher
│   ├── image_processing.py# Image processing functions
│   ├── filters.py         # Filtering algorithms
│   ├── compression.py     # Compression algorithms
│   └── utils.py           # Helper functions
│
├── data/                  # Sample images and processed outputs
│   ├── raw/               # Original images
│   └── processed/         # Processed results
│
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

⚙️ Requirements

Python 3.8+

Packages:

pip install numpy opencv-python matplotlib scikit-image pillow


Optional (for advanced GUI):

PyQt5 or Tkinter (built-in)

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/image-processing-tool.git
cd image-processing-tool


Install dependencies:

pip install -r requirements.txt


Launch the application:

python src/main.py


Upload an image and use the buttons to apply operations.

📊 Usage Example

Step 1: Click “Upload Image” to load an image.

Step 2: Click “Grayscale” to convert the image.

Step 3: Apply filters or transformations dynamically using the corresponding buttons.

Step 4: Compare original vs processed images side by side.

Step 5: Save processed images or compression results for further analysis.

📝 Notes

Binary conversion uses average intensity thresholding by default.

Affine transformations are matrix-based, enabling precise geometric modifications.

Histogram equalization improves contrast for low-light images.

Compression techniques include both lossless and lossy methods with performance metrics.

📈 Future Enhancements

Real-time live camera processing

AI-based super-resolution for interpolation

Advanced visualization of compression artifacts

Multi-format export (JPEG, PNG, BMP, TIFF)

Batch processing of multiple images

📚 References

Gonzalez, R. C., & Woods, R. E. Digital Image Processing, 4th Edition, 2018.

Jain, A. K. Fundamentals of Digital Image Processing, 1989.

OpenCV Documentation: https://docs.opencv.org/

Scikit-Image Documentation: https://scikit-image.org/# Image_Processing_project
