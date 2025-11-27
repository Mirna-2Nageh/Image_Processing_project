import streamlit as st
import numpy as np
from PIL import Image
import io
import heapq
import math
from collections import Counter

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Advanced Image Processing Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 12px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .section-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    h1, h2, h3, h4 {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'image_info' not in st.session_state:
    st.session_state.image_info = {}

# ==================== CUSTOM IMPLEMENTATIONS ====================

def get_image_info(image_array):
    """Get image information"""
    height, width = image_array.shape[:2]
    channels = 1 if len(image_array.shape) == 2 else image_array.shape[2]
    size_kb = (image_array.nbytes) / 1024
    
    return {
        'resolution': f"{width} × {height}",
        'size': f"{size_kb:.2f} KB",
        'type': f"{'Grayscale' if channels == 1 else 'RGB'} ({channels} channel{'s' if channels > 1 else ''})",
        'width': width,
        'height': height,
        'channels': channels
    }

# ============ GRAYSCALE CONVERSION (CUSTOM) ============
def custom_rgb_to_grayscale(image):
    """Convert RGB to grayscale using custom implementation"""
    if len(image.shape) == 2:
        return image
    
    height, width = image.shape[:2]
    grayscale = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            grayscale[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
    
    return grayscale

# ============ BINARY CONVERSION (CUSTOM) ============
def custom_binary_threshold(image, threshold_type="otsu"):
    """Convert to binary using different threshold methods"""
    gray = custom_rgb_to_grayscale(image)
    
    if threshold_type == "mean":
        # Use mean as threshold
        threshold = np.mean(gray)
    elif threshold_type == "otsu":
        # Otsu's method
        hist = [0] * 256
        for row in gray:
            for pixel in row:
                hist[pixel] += 1
        
        total_pixels = gray.shape[0] * gray.shape[1]
        sum_total = sum(i * hist[i] for i in range(256))
        
        sum_bg = 0
        weight_bg = 0
        max_variance = 0
        threshold = 128
        
        for i in range(256):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
                
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0:
                break
                
            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold = i
    else:  # Fixed threshold
        threshold = 128
    
    # Apply threshold
    binary = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            binary[i, j] = 255 if gray[i, j] >= threshold else 0
    
    # Evaluate threshold optimality
    num_white = np.sum(binary == 255)
    num_black = np.sum(binary == 0)
    total_pixels = num_white + num_black
    
    if total_pixels > 0:
        white_ratio = num_white / total_pixels
        black_ratio = num_black / total_pixels
        balance_ratio = min(white_ratio, black_ratio) / max(white_ratio, black_ratio)
    else:
        balance_ratio = 0
    
    is_optimal = balance_ratio > 0.2  # Good balance between black and white
    
    evaluation = {
        'threshold': threshold,
        'white_pixels': num_white,
        'black_pixels': num_black,
        'balance_ratio': balance_ratio,
        'is_optimal': is_optimal,
        'message': f"Threshold: {threshold:.1f}, Balance: {balance_ratio:.3f} ({'Optimal' if is_optimal else 'Suboptimal'})"
    }
    
    return binary, evaluation

# ============ AFFINE TRANSFORMATIONS (CUSTOM) ============
def custom_translation(image, tx, ty):
    """Apply translation (custom implementation)"""
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        translated = np.zeros_like(image)
    else:
        translated = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            new_i, new_j = i + ty, j + tx
            if 0 <= new_i < height and 0 <= new_j < width:
                translated[new_i, new_j] = image[i, j]
    
    return translated

def custom_scaling(image, scale_x, scale_y):
    """Apply scaling (custom implementation)"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_y), int(width * scale_x)
    
    if len(image.shape) == 3:
        scaled = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        scaled = np.zeros((new_height, new_width), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i / scale_y)
            orig_j = int(j / scale_x)
            if 0 <= orig_i < height and 0 <= orig_j < width:
                scaled[i, j] = image[orig_i, orig_j]
    
    return scaled

def custom_rotation(image, angle_degrees):
    """Apply rotation (custom implementation)"""
    angle = np.radians(angle_degrees)
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    cos_a, sin_a = abs(np.cos(angle)), abs(np.sin(angle))
    new_height = int(height * cos_a + width * sin_a)
    new_width = int(width * cos_a + height * sin_a)
    
    if len(image.shape) == 3:
        rotated = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        rotated = np.zeros((new_height, new_width), dtype=np.uint8)
    
    cx, cy = width / 2, height / 2
    new_cx, new_cy = new_width / 2, new_height / 2
    
    for i in range(new_height):
        for j in range(new_width):
            # Translate to origin
            x = j - new_cx
            y = i - new_cy
            
            # Apply inverse rotation
            orig_x = x * np.cos(-angle) - y * np.sin(-angle) + cx
            orig_y = x * np.sin(-angle) + y * np.cos(-angle) + cy
            
            orig_i, orig_j = int(orig_y), int(orig_x)
            if 0 <= orig_i < height and 0 <= orig_j < width:
                rotated[i, j] = image[orig_i, orig_j]
    
    return rotated

def custom_shear_x(image, shear_factor):
    """Apply X-direction shear (custom)"""
    height, width = image.shape[:2]
    new_width = width + int(height * abs(shear_factor))
    
    if len(image.shape) == 3:
        sheared = np.zeros((height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        sheared = np.zeros((height, new_width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            new_j = j + int(i * shear_factor)
            if 0 <= new_j < new_width:
                sheared[i, new_j] = image[i, j]
    
    return sheared

def custom_shear_y(image, shear_factor):
    """Apply Y-direction shear (custom)"""
    height, width = image.shape[:2]
    new_height = height + int(width * abs(shear_factor))
    
    if len(image.shape) == 3:
        sheared = np.zeros((new_height, width, image.shape[2]), dtype=np.uint8)
    else:
        sheared = np.zeros((new_height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            new_i = i + int(j * shear_factor)
            if 0 <= new_i < new_height:
                sheared[new_i, j] = image[i, j]
    
    return sheared

# ============ INTERPOLATION METHODS (CUSTOM) ============
def custom_nearest_neighbor(image, scale_factor):
    """Nearest neighbor interpolation (custom)"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i / scale_factor)
            orig_j = int(j / scale_factor)
            orig_i = min(orig_i, height - 1)
            orig_j = min(orig_j, width - 1)
            resized[i, j] = image[orig_i, orig_j]
    
    return resized

def custom_bilinear(image, scale_factor):
    """Bilinear interpolation (custom)"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        channels = image.shape[2]
    else:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
        channels = 1
        image = image[:, :, np.newaxis]
    
    for i in range(new_height):
        for j in range(new_width):
            # Map to original coordinates
            orig_i = i / scale_factor
            orig_j = j / scale_factor
            
            # Get four nearest pixels
            i1, j1 = int(orig_i), int(orig_j)
            i2, j2 = min(i1 + 1, height - 1), min(j1 + 1, width - 1)
            
            # Calculate weights
            di, dj = orig_i - i1, orig_j - j1
            
            for c in range(channels):
                # Bilinear interpolation
                v1 = image[i1, j1, c] * (1 - di) + image[i2, j1, c] * di
                v2 = image[i1, j2, c] * (1 - di) + image[i2, j2, c] * di
                value = v1 * (1 - dj) + v2 * dj
                
                if channels == 1:
                    resized[i, j] = int(value)
                else:
                    resized[i, j, c] = int(value)
    
    return resized if channels > 1 else resized[:, :, 0]

def custom_bicubic(image, scale_factor):
    """Bicubic interpolation (custom)"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        channels = image.shape[2]
    else:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
        channels = 1
        image = image[:, :, np.newaxis]
    
    def cubic_weight(x):
        """Cubic interpolation kernel"""
        x = abs(x)
        if x <= 1:
            return 1.5 * x**3 - 2.5 * x**2 + 1
        elif x < 2:
            return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
        return 0
    
    for i in range(new_height):
        for j in range(new_width):
            orig_i = i / scale_factor
            orig_j = j / scale_factor
            
            i0 = int(orig_i)
            j0 = int(orig_j)
            
            for c in range(channels):
                value = 0
                total_weight = 0
                
                # Sample 4x4 neighborhood
                for di in range(-1, 3):
                    for dj in range(-1, 3):
                        ni, nj = i0 + di, j0 + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            weight = cubic_weight(orig_i - ni) * cubic_weight(orig_j - nj)
                            value += image[ni, nj, c] * weight
                            total_weight += weight
                
                if total_weight > 0:
                    value = value / total_weight
                
                value = max(0, min(255, int(value)))
                
                if channels == 1:
                    resized[i, j] = value
                else:
                    resized[i, j, c] = value
    
    return resized if channels > 1 else resized[:, :, 0]

def custom_crop(image, x1, y1, x2, y2):
    """Crop image (custom)"""
    return image[y1:y2, x1:x2]

# ============ HISTOGRAM ANALYSIS (CUSTOM) ============
def custom_histogram(image):
    """Calculate histogram (custom)"""
    gray = custom_rgb_to_grayscale(image)
    hist = [0] * 256
    
    for row in gray:
        for pixel in row:
            hist[pixel] += 1
    
    return np.array(hist)

def assess_histogram(hist):
    """Assess histogram quality"""
    total = sum(hist)
    if total == 0:
        return {
            'mean': 0,
            'std_dev': 0,
            'non_zero_bins': 0,
            'is_good': False,
            'reasoning': "No data in histogram"
        }
    
    # Calculate mean and standard deviation
    mean = sum(i * hist[i] for i in range(256)) / total
    variance = sum(hist[i] * (i - mean)**2 for i in range(256)) / total
    std_dev = np.sqrt(variance)
    
    # Check distribution
    non_zero_bins = sum(1 for h in hist if h > 0)
    
    # Assess quality
    is_good = (std_dev > 40 and non_zero_bins > 150 and 
               mean > 50 and mean < 200 and 
               max(hist) < total * 0.3)
    
    reasoning = []
    if std_dev > 40:
        reasoning.append("good contrast")
    else:
        reasoning.append("low contrast")
    
    if non_zero_bins > 150:
        reasoning.append("good tonal range")
    else:
        reasoning.append("limited tonal range")
    
    if 50 < mean < 200:
        reasoning.append("balanced brightness")
    else:
        reasoning.append("extreme brightness")
    
    assessment = {
        'mean': mean,
        'std_dev': std_dev,
        'non_zero_bins': non_zero_bins,
        'is_good': is_good,
        'reasoning': "Image has " + ", ".join(reasoning)
    }
    
    return assessment

def custom_histogram_equalization(image):
    """Histogram equalization (custom)"""
    gray = custom_rgb_to_grayscale(image)
    hist = custom_histogram(image)
    
    # Calculate CDF
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    # Find minimum non-zero CDF value
    cdf_min = min(c for c in cdf if c > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    # Create lookup table
    lut = [0] * 256
    for i in range(256):
        if total_pixels - cdf_min > 0:
            lut[i] = int(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
        else:
            lut[i] = i
    
    # Apply equalization
    equalized = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            equalized[i, j] = lut[gray[i, j]]
    
    return equalized

# ============ FILTERS (CUSTOM) ============
def custom_gaussian_filter(image, kernel_size=19, sigma=3):
    """Gaussian filter (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    # Create Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution
    pad = kernel_size // 2
    height, width = gray.shape
    filtered = np.zeros_like(gray)
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            value = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    value += gray[i - pad + ki, j - pad + kj] * kernel[ki, kj]
            filtered[i, j] = int(value)
    
    return filtered

def custom_median_filter(image, kernel_size=7):
    """Median filter (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    pad = kernel_size // 2
    height, width = gray.shape
    filtered = np.zeros_like(gray)
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Collect neighborhood values
            window = []
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    window.append(gray[i - pad + ki, j - pad + kj])
            
            # Find median manually
            window.sort()
            median = window[len(window) // 2]
            filtered[i, j] = median
    
    return filtered

def custom_laplacian_filter(image):
    """Laplacian filter (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    # Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    
    height, width = gray.shape
    filtered = np.zeros_like(gray)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            value = 0
            for ki in range(3):
                for kj in range(3):
                    value += gray[i - 1 + ki, j - 1 + kj] * kernel[ki, kj]
            filtered[i, j] = min(255, max(0, abs(value)))
    
    return filtered

def custom_sobel_filter(image):
    """Sobel filter (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    height, width = gray.shape
    filtered = np.zeros_like(gray)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = 0
            gy = 0
            for ki in range(3):
                for kj in range(3):
                    pixel = gray[i - 1 + ki, j - 1 + kj]
                    gx += pixel * sobel_x[ki, kj]
                    gy += pixel * sobel_y[ki, kj]
            
            magnitude = int(np.sqrt(gx**2 + gy**2))
            filtered[i, j] = min(255, magnitude)
    
    return filtered

def custom_gradient_filter(image):
    """Gradient filter - first derivatives (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    height, width = gray.shape
    filtered = np.zeros_like(gray)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Simple gradient using differences
            dx = int(gray[i, j+1]) - int(gray[i, j-1])
            dy = int(gray[i+1, j]) - int(gray[i-1, j])
            
            magnitude = int(np.sqrt(dx**2 + dy**2))
            filtered[i, j] = min(255, magnitude)
    
    return filtered

# ============ COMPRESSION TECHNIQUES (CUSTOM) ============

# 1. HUFFMAN CODING
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(image):
    """Huffman coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten()
    
    # Calculate frequencies
    freq_dict = {}
    for pixel in data:
        freq_dict[pixel] = freq_dict.get(pixel, 0) + 1
    
    # Build Huffman tree
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    # Calculate average code length
    def get_codes(node, code='', codes={}):
        if node.symbol is not None:
            codes[node.symbol] = code
            return codes
        if node.left:
            get_codes(node.left, code + '0', codes)
        if node.right:
            get_codes(node.right, code + '1', codes)
        return codes
    
    codes = get_codes(heap[0])
    
    # Calculate compression ratio
    original_bits = len(data) * 8
    compressed_bits = sum(len(codes[pixel]) * freq_dict[pixel] for pixel in freq_dict)
    
    return {
        'original_size': original_bits / 8,
        'compressed_size': compressed_bits / 8,
        'ratio': (1 - compressed_bits / original_bits) * 100 if original_bits > 0 else 0,
        'method': 'Huffman Coding'
    }

# 2. RUN-LENGTH ENCODING (RLE)
def rle_encoding(image):
    """Run-Length Encoding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten()
    
    encoding = []
    i = 0
    while i < len(data):
        count = 1
        while i + count < len(data) and data[i] == data[i + count] and count < 255:
            count += 1
        encoding.append((data[i], count))
        i += count
    
    original_size = len(data)
    compressed_size = len(encoding) * 2  # (value, count) pairs
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'method': 'Run-Length Encoding'
    }

# 3. LZW CODING
def lzw_encoding(image):
    """LZW coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten().tolist()
    
    if not data:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'LZW Coding'
        }
    
    # Initialize dictionary
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    
    result = []
    current = bytes([data[0]])
    
    for i in range(1, min(len(data), 10000)):  # Increased limit
        symbol = bytes([data[i]])
        combined = current + symbol
        
        if combined in dictionary:
            current = combined
        else:
            result.append(dictionary[current])
            if dict_size < 4096:  # Limit dictionary size
                dictionary[combined] = dict_size
                dict_size += 1
            current = symbol
    
    if current in dictionary:
        result.append(dictionary[current])
    
    original_size = len(data)
    compressed_size = len(result) * 2  # Assuming 16-bit codes
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'method': 'LZW Coding'
    }

# 4. GOLOMB-RICE CODING
def golomb_rice_encoding(image):
    """Golomb-Rice coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten()
    
    if len(data) == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Golomb-Rice Coding'
        }
    
    # Calculate differences (predictive)
    differences = [data[0]]
    for i in range(1, len(data)):
        differences.append(int(data[i]) - int(data[i-1]))
    
    # Estimate parameter m
    avg_diff = sum(abs(d) for d in differences) / len(differences)
    m = max(1, int(avg_diff))
    
    # Estimate code length
    total_bits = 0
    sample_size = min(1000, len(differences))
    for diff in differences[:sample_size]:
        quotient = abs(diff) // m
        total_bits += quotient + 1 + math.ceil(math.log2(m))
    
    estimated_compressed = (total_bits / sample_size) * len(differences) / 8
    
    return {
        'original_size': len(data),
        'compressed_size': estimated_compressed,
        'ratio': (1 - estimated_compressed / len(data)) * 100 if len(data) > 0 else 0,
        'method': 'Golomb-Rice Coding'
    }

# 5. ARITHMETIC CODING
def arithmetic_encoding(image):
    """Arithmetic coding simulation"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten()
    
    if len(data) == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Arithmetic Coding'
        }
    
    # Calculate symbol probabilities
    freq = {}
    for pixel in data:
        freq[pixel] = freq.get(pixel, 0) + 1
    
    total = len(data)
    
    # Calculate entropy
    entropy = 0
    for f in freq.values():
        p = f / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Theoretical compressed size
    original_bits = len(data) * 8
    compressed_bits = len(data) * entropy
    
    return {
        'original_size': original_bits / 8,
        'compressed_size': compressed_bits / 8,
        'ratio': (1 - compressed_bits / original_bits) * 100 if original_bits > 0 else 0,
        'method': 'Arithmetic Coding'
    }

# 6. SYMBOL-BASED CODING
def symbol_based_encoding(image):
    """Symbol-based coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    data = gray.flatten()
    
    if len(data) == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Symbol-Based Coding'
        }
    
    # Count unique symbols
    unique_symbols = len(set(data))
    
    # Bits needed per symbol
    bits_per_symbol = math.ceil(math.log2(unique_symbols)) if unique_symbols > 0 else 0
    
    original_bits = len(data) * 8
    compressed_bits = len(data) * bits_per_symbol
    
    return {
        'original_size': original_bits / 8,
        'compressed_size': compressed_bits / 8,
        'ratio': (1 - compressed_bits / original_bits) * 100 if original_bits > 0 else 0,
        'method': 'Symbol-Based Coding'
    }

# 7. BIT-PLANE CODING
def bitplane_encoding(image):
    """Bit-plane coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    if gray.size == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Bit-Plane Coding'
        }
    
    # Extract bit planes
    bitplanes = []
    for bit in range(8):
        plane = (gray >> bit) & 1
        bitplanes.append(plane)
    
    # Estimate compression by RLE on each plane
    total_compressed = 0
    for plane in bitplanes:
        data = plane.flatten()
        if len(data) == 0:
            continue
        runs = 1
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                runs += 1
        total_compressed += runs * 2  # (value, count)
    
    original_size = gray.size
    
    return {
        'original_size': original_size,
        'compressed_size': total_compressed / 8,
        'ratio': (1 - total_compressed / (original_size * 8)) * 100 if original_size > 0 else 0,
        'method': 'Bit-Plane Coding'
    }

# 8. DCT BLOCK TRANSFORM
def dct_encoding(image):
    """DCT Block Transform (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    if gray.size == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'DCT Block Transform'
        }
    
    # Simple DCT on 8x8 blocks
    def dct2d(block):
        N = 8
        dct_block = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                sum_val = 0
                for x in range(N):
                    for y in range(N):
                        sum_val += block[x, y] * np.cos((2*x+1)*u*np.pi/(2*N)) * np.cos((2*y+1)*v*np.pi/(2*N))
                alpha_u = 1/np.sqrt(N) if u == 0 else np.sqrt(2/N)
                alpha_v = 1/np.sqrt(N) if v == 0 else np.sqrt(2/N)
                dct_block[u, v] = alpha_u * alpha_v * sum_val
        return dct_block
    
    # Process sample blocks
    total_significant = 0
    total_coeffs = 0
    sample_blocks = 0
    
    for i in range(0, gray.shape[0]-7, 8):
        for j in range(0, gray.shape[1]-7, 8):
            if sample_blocks >= 10:  # Limit samples for performance
                break
            block = gray[i:i+8, j:j+8].astype(float)
            dct_block = dct2d(block)
            
            # Count significant coefficients (above threshold)
            threshold = 10
            significant = np.sum(np.abs(dct_block) > threshold)
            total_significant += significant
            total_coeffs += 64
            sample_blocks += 1
    
    if total_coeffs > 0:
        compression_ratio = total_significant / total_coeffs
    else:
        compression_ratio = 0
    
    original_size = gray.size
    compressed_size = original_size * compression_ratio
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'method': 'DCT Block Transform'
    }

# 9. PREDICTIVE CODING
def predictive_coding(image):
    """Predictive coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    if gray.size == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Predictive Coding'
        }
    
    # Simple predictive coding using previous pixel
    differences = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if i == 0 and j == 0:
                differences.append(gray[i, j])
            else:
                if j == 0:
                    predicted = gray[i-1, j]
                else:
                    predicted = gray[i, j-1]
                differences.append(int(gray[i, j]) - int(predicted))
    
    # Calculate entropy of differences
    freq = {}
    for diff in differences:
        freq[diff] = freq.get(diff, 0) + 1
    
    total = len(differences)
    entropy = 0
    for f in freq.values():
        p = f / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    original_bits = gray.size * 8
    compressed_bits = gray.size * entropy
    
    return {
        'original_size': original_bits / 8,
        'compressed_size': compressed_bits / 8,
        'ratio': (1 - compressed_bits / original_bits) * 100 if original_bits > 0 else 0,
        'method': 'Predictive Coding'
    }

# 10. WAVELET CODING (Simplified)
def wavelet_coding(image):
    """Simplified Wavelet coding (custom)"""
    gray = custom_rgb_to_grayscale(image)
    
    if gray.size == 0:
        return {
            'original_size': 0,
            'compressed_size': 0,
            'ratio': 0,
            'method': 'Wavelet Coding'
        }
    
    # Simple Haar wavelet approximation
    def haar_transform(block):
        """Simple Haar wavelet transform"""
        N = block.shape[0]
        transformed = np.zeros_like(block, dtype=float)
        
        # Horizontal transform
        for i in range(N):
            for j in range(0, N, 2):
                if j + 1 < N:
                    avg = (block[i, j] + block[i, j+1]) / 2
                    diff = (block[i, j] - block[i, j+1]) / 2
                    transformed[i, j//2] = avg
                    transformed[i, j//2 + N//2] = diff
        
        # Vertical transform
        result = np.zeros_like(transformed)
        for j in range(N):
            for i in range(0, N, 2):
                if i + 1 < N:
                    avg = (transformed[i, j] + transformed[i+1, j]) / 2
                    diff = (transformed[i, j] - transformed[i+1, j]) / 2
                    result[i//2, j] = avg
                    result[i//2 + N//2, j] = diff
        
        return result
    
    # Process sample blocks
    total_significant = 0
    total_coeffs = 0
    sample_blocks = 0
    
    for i in range(0, gray.shape[0]-7, 8):
        for j in range(0, gray.shape[1]-7, 8):
            if sample_blocks >= 5:  # Limit samples for performance
                break
            block = gray[i:i+8, j:j+8].astype(float)
            wavelet_block = haar_transform(block)
            
            # Count significant coefficients (above threshold)
            threshold = 5
            significant = np.sum(np.abs(wavelet_block) > threshold)
            total_significant += significant
            total_coeffs += 64
            sample_blocks += 1
    
    if total_coeffs > 0:
        compression_ratio = total_significant / total_coeffs
    else:
        compression_ratio = 0
    
    original_size = gray.size
    compressed_size = original_size * compression_ratio
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'method': 'Wavelet Coding'
    }

# ==================== STREAMLIT UI ====================

st.title("🖼️ Advanced Image Processing Laboratory")

# ==================== SIDEBAR - IMAGE UPLOAD ====================
st.sidebar.header("📁 Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and store the image
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        
        # Store both original and processed images
        st.session_state.original_image = image_array
        st.session_state.processed_image = image_array.copy()
        st.session_state.image_uploaded = True
        st.session_state.image_info = get_image_info(image_array)
        
        st.sidebar.success("✅ Image uploaded successfully!")
        
        # Display image information
        st.sidebar.header("📊 Image Information")
        info = st.session_state.image_info
        st.sidebar.write(f"**Resolution:** {info['resolution']}")
        st.sidebar.write(f"**Size:** {info['size']}")
        st.sidebar.write(f"**Type:** {info['type']}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading image: {e}")
else:
    st.session_state.image_uploaded = False

# ==================== MAIN CONTENT - TABS ====================
if st.session_state.image_uploaded:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🖼️ Basic Operations",
        "🔄 Geometric Transforms", 
        "📈 Histogram Analysis",
        "🔍 Image Filters",
        "🗜️ Compression",
        "🎯 Advanced Features"
    ])

    # ==================== TAB 1: BASIC OPERATIONS ====================
    with tab1:
        st.header("Basic Image Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("🎨 Color Conversion")
            
            if st.button("Convert to Grayscale", key="grayscale_btn"):
                try:
                    gray = custom_rgb_to_grayscale(st.session_state.processed_image)
                    st.session_state.processed_image = gray
                    st.success("✅ Successfully converted to grayscale!")
                except Exception as e:
                    st.error(f"❌ Error in grayscale conversion: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("✂️ Image Cropping")
            
            if st.session_state.processed_image is not None:
                height, width = st.session_state.processed_image.shape[:2]
                col_x1, col_x2 = st.columns(2)
                with col_x1:
                    x1 = st.slider("X Start", 0, width-1, 0, key="x1_slider")
                    y1 = st.slider("Y Start", 0, height-1, 0, key="y1_slider")
                with col_x2:
                    x2 = st.slider("X End", x1+1, width, width, key="x2_slider")
                    y2 = st.slider("Y End", y1+1, height, height, key="y2_slider")
                
                if st.button("Crop Image", key="crop_btn"):
                    try:
                        cropped = custom_crop(st.session_state.processed_image, x1, y1, x2, y2)
                        st.session_state.processed_image = cropped
                        st.success("✅ Image cropped successfully!")
                    except Exception as e:
                        st.error(f"❌ Error in cropping: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("⚫ Binary Conversion")
            
            threshold_method = st.selectbox(
                "Threshold Method:",
                ["otsu", "mean", "fixed"],
                key="threshold_select"
            )
            
            if st.button("Convert to Binary", key="binary_btn"):
                try:
                    binary, evaluation = custom_binary_threshold(
                        st.session_state.processed_image, 
                        threshold_method
                    )
                    st.session_state.processed_image = binary
                    st.success("✅ Binary conversion complete!")
                    
                    # Display evaluation
                    st.info(f"📊 {evaluation['message']}")
                    col_eval1, col_eval2 = st.columns(2)
                    with col_eval1:
                        st.metric("Threshold", f"{evaluation['threshold']:.1f}")
                    with col_eval2:
                        st.metric("Balance Ratio", f"{evaluation['balance_ratio']:.3f}")
                        
                except Exception as e:
                    st.error(f"❌ Error in binary conversion: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 2: GEOMETRIC TRANSFORMS ====================
    with tab2:
        st.header("Geometric Transformations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("➡️ Translation")
            tx = st.slider("Translate X", -100, 100, 0, key="tx_slider")
            ty = st.slider("Translate Y", -100, 100, 0, key="ty_slider")
            
            if st.button("Apply Translation", key="translate_btn"):
                try:
                    st.session_state.processed_image = custom_translation(st.session_state.processed_image, tx, ty)
                    st.success("✅ Translation applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in translation: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("📏 Scaling")
            scale = st.slider("Scale Factor", 0.1, 3.0, 1.0, key="scale_slider")
            
            if st.button("Apply Scaling", key="scale_btn"):
                try:
                    st.session_state.processed_image = custom_scaling(st.session_state.processed_image, scale, scale)
                    st.success("✅ Scaling applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in scaling: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("🔄 Rotation")
            angle = st.slider("Rotation Angle", -180, 180, 0, key="angle_slider")
            
            if st.button("Apply Rotation", key="rotate_btn"):
                try:
                    st.session_state.processed_image = custom_rotation(st.session_state.processed_image, angle)
                    st.success("✅ Rotation applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in rotation: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("📐 Shearing")
            shear = st.slider("Shear Factor", -1.0, 1.0, 0.0, key="shear_slider")
            
            col_shear1, col_shear2 = st.columns(2)
            with col_shear1:
                if st.button("Shear X", key="shear_x_btn"):
                    try:
                        st.session_state.processed_image = custom_shear_x(st.session_state.processed_image, shear)
                        st.success("✅ X-shear applied successfully!")
                    except Exception as e:
                        st.error(f"❌ Error in X-shear: {e}")
            with col_shear2:
                if st.button("Shear Y", key="shear_y_btn"):
                    try:
                        st.session_state.processed_image = custom_shear_y(st.session_state.processed_image, shear)
                        st.success("✅ Y-shear applied successfully!")
                    except Exception as e:
                        st.error(f"❌ Error in Y-shear: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 3: HISTOGRAM ANALYSIS ====================
    with tab3:
        st.header("Histogram Analysis & Equalization")
        
        if st.session_state.image_uploaded:
            try:
                # Calculate histogram
                hist = custom_histogram(st.session_state.processed_image)
                assessment = assess_histogram(hist)

                col_hist1, col_hist2 = st.columns([2, 1])
                
                with col_hist1:
                    st.subheader("📊 Image Histogram")
                    st.bar_chart(hist)
                
                with col_hist2:
                    st.subheader("📈 Histogram Assessment")
                    
                    # Display metrics
                    st.metric("Mean Intensity", f"{assessment['mean']:.2f}")
                    st.metric("Standard Deviation", f"{assessment['std_dev']:.2f}")
                    st.metric("Non-zero Bins", assessment['non_zero_bins'])
                    
                    # Display assessment
                    if assessment['is_good']:
                        st.success(f"✅ {assessment['reasoning']}")
                    else:
                        st.warning(f"⚠️ {assessment['reasoning']}")

                # Histogram equalization
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.subheader("⚖️ Histogram Equalization")
                
                if st.button("Apply Histogram Equalization", key="hist_eq_btn"):
                    try:
                        eq = custom_histogram_equalization(st.session_state.processed_image)
                        st.session_state.processed_image = eq
                        st.success("✅ Histogram equalization applied successfully!")
                    except Exception as e:
                        st.error(f"❌ Error in histogram equalization: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error in histogram analysis: {e}")

    # ==================== TAB 4: IMAGE FILTERS ====================
    with tab4:
        st.header("Image Filtering Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("🌫️ Low-Pass Filters")
            
            if st.button("Gaussian Filter (19×19, σ=3)", key="gaussian_btn"):
                try:
                    st.session_state.processed_image = custom_gaussian_filter(st.session_state.processed_image)
                    st.success("✅ Gaussian filter applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in Gaussian filter: {e}")

            if st.button("Median Filter (7×7)", key="median_btn"):
                try:
                    st.session_state.processed_image = custom_median_filter(st.session_state.processed_image)
                    st.success("✅ Median filter applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in median filter: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("🔍 High-Pass Filters")
            
            if st.button("Laplacian Filter", key="laplacian_btn"):
                try:
                    st.session_state.processed_image = custom_laplacian_filter(st.session_state.processed_image)
                    st.success("✅ Laplacian filter applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in Laplacian filter: {e}")

            if st.button("Sobel Filter", key="sobel_btn"):
                try:
                    st.session_state.processed_image = custom_sobel_filter(st.session_state.processed_image)
                    st.success("✅ Sobel filter applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in Sobel filter: {e}")

            if st.button("Gradient Filter", key="gradient_btn"):
                try:
                    st.session_state.processed_image = custom_gradient_filter(st.session_state.processed_image)
                    st.success("✅ Gradient filter applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in gradient filter: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 5: COMPRESSION ====================
    with tab5:
        st.header("Image Compression Techniques")
        
        if st.session_state.image_uploaded:
            compression_methods = [
                ("Huffman Coding", huffman_encoding),
                ("Run-Length Encoding", rle_encoding),
                ("LZW Coding", lzw_encoding),
                ("Golomb-Rice Coding", golomb_rice_encoding),
                ("Arithmetic Coding", arithmetic_encoding),
                ("Symbol-Based Coding", symbol_based_encoding),
                ("Bit-Plane Coding", bitplane_encoding),
                ("DCT Block Transform", dct_encoding),
                ("Predictive Coding", predictive_coding),
                ("Wavelet Coding", wavelet_coding)
            ]
            
            # Display compression methods in a grid
            cols = st.columns(2)
            for idx, (method_name, method_func) in enumerate(compression_methods):
                with cols[idx % 2]:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    if st.button(f"🚀 {method_name}", key=f"compress_{idx}"):
                        with st.spinner(f"🔄 Applying {method_name}..."):
                            try:
                                result = method_func(st.session_state.processed_image)
                                
                                st.subheader(f"📊 {result['method']} Results")
                                
                                # Display results in metrics
                                col_res1, col_res2, col_res3 = st.columns(3)
                                with col_res1:
                                    st.metric("Original Size", f"{result['original_size']:.2f} bytes")
                                with col_res2:
                                    st.metric("Compressed Size", f"{result['compressed_size']:.2f} bytes")
                                with col_res3:
                                    st.metric("Compression Ratio", f"{result['ratio']:.2f}%")
                                    
                            except Exception as e:
                                st.error(f"❌ Error in {method_name}: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 6: ADVANCED FEATURES ====================
    with tab6:
        st.header("🎯 Advanced Image Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("🔄 Image Interpolation")
            
            scale_method = st.selectbox(
                "Interpolation Method:",
                ["Nearest Neighbor", "Bilinear", "Bicubic"],
                key="interp_select"
            )
            
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.5, key="interp_scale")
            
            if st.button("Apply Interpolation", key="interp_btn"):
                try:
                    if scale_method == "Nearest Neighbor":
                        resized = custom_nearest_neighbor(st.session_state.processed_image, scale_factor)
                    elif scale_method == "Bilinear":
                        resized = custom_bilinear(st.session_state.processed_image, scale_factor)
                    else:  # Bicubic
                        resized = custom_bicubic(st.session_state.processed_image, scale_factor)
                    
                    st.session_state.processed_image = resized
                    st.success(f"✅ {scale_method} interpolation applied successfully!")
                except Exception as e:
                    st.error(f"❌ Error in interpolation: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("📊 Real-time Analysis")
            
            if st.button("Generate Analysis Report", key="analysis_btn"):
                try:
                    # Generate comprehensive analysis
                    gray = custom_rgb_to_grayscale(st.session_state.processed_image)
                    hist = custom_histogram(st.session_state.processed_image)
                    assessment = assess_histogram(hist)
                    
                    st.success("📈 Analysis Report Generated!")
                    
                    col_rep1, col_rep2 = st.columns(2)
                    with col_rep1:
                        st.metric("Image Dimensions", f"{gray.shape[1]}×{gray.shape[0]}")
                        st.metric("Total Pixels", f"{gray.size:,}")
                        st.metric("Mean Intensity", f"{np.mean(gray):.2f}")
                    with col_rep2:
                        st.metric("Contrast (Std Dev)", f"{np.std(gray):.2f}")
                        st.metric("Dynamic Range", f"{np.max(gray) - np.min(gray)}")
                        st.metric("Histogram Quality", "Good" if assessment['is_good'] else "Needs Enhancement")
                        
                except Exception as e:
                    st.error(f"❌ Error in analysis: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== IMAGE DISPLAY ====================
    st.markdown("---")
    st.header("🖼️ Image Comparison")
    
    col_disp1, col_disp2 = st.columns(2)
    
    with col_disp1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_column_width=True)
        
    with col_disp2:
        st.subheader("Processed Image")
        st.image(st.session_state.processed_image, use_column_width=True)

else:
    # Welcome screen when no image is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1>🖼️ Welcome to Advanced Image Processing Lab</h1>
        <p style='font-size: 18px;'>Upload an image from the sidebar to get started with advanced image processing operations!</p>
        <div style='margin-top: 30px;'>
            <h3>🎯 Features Included:</h3>
            <ul style='text-align: left; display: inline-block;'>
                <li>Basic Image Operations (Grayscale, Binary Conversion)</li>
                <li>Geometric Transformations (Translation, Rotation, Scaling, Shearing)</li>
                <li>Histogram Analysis & Equalization</li>
                <li>Image Filtering (Gaussian, Median, Laplacian, Sobel, Gradient)</li>
                <li>10+ Compression Techniques</li>
                <li>Advanced Interpolation Methods</li>
                <li>Real-time Image Analysis</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR - IMAGE MANAGEMENT ====================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Image Management")

# Reset button
if st.sidebar.button("🔄 Reset to Original", key="reset_btn"):
    if st.session_state.original_image is not None:
        st.session_state.processed_image = st.session_state.original_image.copy()
        st.sidebar.success("✅ Image reset to original!")
        st.rerun()
    else:
        st.sidebar.error("❌ No original image available!")

# Download button
if st.session_state.processed_image is not None:
    try:
        # Convert numpy array to PIL Image
        if len(st.session_state.processed_image.shape) == 2:
            img = Image.fromarray(st.session_state.processed_image, mode='L')
        else:
            img = Image.fromarray(st.session_state.processed_image)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        st.sidebar.download_button(
            label="📥 Download Processed Image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png",
            key="download_btn"
        )
    except Exception as e:
        st.sidebar.error(f"❌ Error creating download: {e}")

# ==================== ABOUT SECTION ====================
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Advanced Image Processing Laboratory**

This application demonstrates comprehensive computer vision algorithms with custom implementations:

- **All operations implemented from scratch**
- **No reliance on built-in high-level image processing functions**
- **Educational focus with real-time visualization**

**Features:**
- Basic & Advanced Image Operations
- Geometric Transformations  
- Histogram Analysis
- Image Filtering (Low-pass & High-pass)
- 10+ Compression Techniques
- Professional GUI Design

*Built with Streamlit & Custom Algorithms*
""")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>"
    "🖼️ Advanced Image Processing Laboratory | Custom Implementation | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)