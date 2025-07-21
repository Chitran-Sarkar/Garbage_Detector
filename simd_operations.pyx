# simd_operations.pyx
# cython: language_level=3, boundscheck=True, wraparound=True, cdivision=True, initializedcheck=True, infer_types=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport round, sqrt, exp
from cython cimport view
from libcpp.vector cimport vector
from libcpp cimport bool
import logging

# Import SSE/AVX intrinsics based on compiler availability
cdef extern from *:
    """
    #include <stdlib.h>
    
    // Platform-specific CPU feature detection
    #if defined(_MSC_VER) // MSVC
        #include <intrin.h>
    #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        #include <cpuid.h>
        #include <x86intrin.h>
    #endif
    
    // Enhanced CPU feature detection at runtime
    bool check_avx2_support() {
        int cpu_info[4];
        #if defined(_MSC_VER)
            __cpuid(cpu_info, 0);
            int max_id = cpu_info[0];
            
            if (max_id >= 7) {
                __cpuidex(cpu_info, 7, 0);
                return (cpu_info[1] & (1 << 5)) != 0; // Check AVX2 bit
            }
            return false;
        #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2");
        #else
            return false;
        #endif
    }
    
    bool check_sse41_support() {
        #if defined(_MSC_VER)
            int cpu_info[4];
            __cpuid(cpu_info, 1);
            return (cpu_info[2] & (1 << 19)) != 0; // Check SSE4.1 bit
        #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
            __builtin_cpu_init();
            return __builtin_cpu_supports("sse4.1");
        #else
            return false;
        #endif
    }
    
    bool check_sse2_support() {
        #if defined(_MSC_VER)
            int cpu_info[4];
            __cpuid(cpu_info, 1);
            return (cpu_info[3] & (1 << 26)) != 0; // Check SSE2 bit
        #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
            __builtin_cpu_init();
            return __builtin_cpu_supports("sse2");
        #else
            return false;
        #endif
    }
    
    // Conditionally include SIMD headers
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define HAS_AVX2_HEADERS 1
    #elif defined(__SSE4_1__)
        #include <smmintrin.h>
        #define HAS_SSE41_HEADERS 1
    #elif defined(__SSE2__) 
        #include <emmintrin.h>
        #define HAS_SSE2_HEADERS 1
    #endif
    """
    # CPU feature detection functions
    bool check_avx2_support()
    bool check_sse41_support()
    bool check_sse2_support()
    
    # AVX2 types
    ctypedef struct __m256:
        float[8] val
    ctypedef struct __m256i:
        int[8] val
    
    # AVX2 intrinsics
    __m256 _mm256_set1_ps(float a)
    __m256i _mm256_setzero_si256()
    __m256i _mm256_loadu_si256(__m256i *p)
    __m256i _mm256_extracti128_si256(__m256i a, int imm8)
    __m256i _mm256_cvtepu8_epi32(__m128i a)
    __m256 _mm256_cvtepi32_ps(__m256i a)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_div_ps(__m256 a, __m256 b)
    void _mm256_storeu_ps(float *p, __m256 a)
    
    # SSE types and intrinsics
    ctypedef struct __m128:
        float[4] val
    ctypedef struct __m128i:
        short[8] val
    
    # SSE intrinsics
    __m128 _mm_set1_ps(float a)
    __m128i _mm_setzero_si128()
    __m128i _mm_loadu_si128(__m128i *p)
    __m128i _mm_cvtepu8_epi32(__m128i a)
    __m128 _mm_cvtepi32_ps(__m128i a)
    __m128 _mm_mul_ps(__m128 a, __m128 b)
    __m128 _mm_sub_ps(__m128 a, __m128 b)
    __m128 _mm_div_ps(__m128 a, __m128 b)
    void _mm_storeu_ps(float *p, __m128 a)

# Store CPU feature detection results with proper runtime checks
cdef:
    bool CPU_HAS_AVX2 = check_avx2_support()
    bool CPU_HAS_SSE41 = check_sse41_support()
    bool CPU_HAS_SSE2 = check_sse2_support()
    
# Data type definitions
ctypedef unsigned char uint8_t
ctypedef float float32_t

# Initialize logging for SIMD operations
simd_logger = logging.getLogger('simd_operations')

def cpu_features():
    """Return information about available CPU SIMD features"""
    features = {
        'avx2': CPU_HAS_AVX2,
        'sse4.1': CPU_HAS_SSE41,
        'sse2': CPU_HAS_SSE2
    }
    
    # Log available features
    simd_logger.info(f"CPU SIMD features: AVX2={'Yes' if CPU_HAS_AVX2 else 'No'}, "
                     f"SSE4.1={'Yes' if CPU_HAS_SSE41 else 'No'}, "
                     f"SSE2={'Yes' if CPU_HAS_SSE2 else 'No'}")
    return features

def simd_image_normalize(np.ndarray[np.uint8_t, ndim=3] image,
                         float mean_r=0.485, float mean_g=0.456, float mean_b=0.406,
                         float std_r=0.229, float std_g=0.224, float std_b=0.225):
    """
    Normalize image using SIMD instructions (AVX2/SSE when available)
    
    This is much faster than the equivalent NumPy operation:
    normalized = (image / 255.0 - mean) / std
    
    Args:
        image: Input BGR image (HxWx3, uint8)
        mean_r, mean_g, mean_b: RGB channel means
        std_r, std_g, std_b: RGB channel standard deviations
        
    Returns:
        Normalized image as float32 array
    """
    cdef:
        int h = image.shape[0]
        int w = image.shape[1]
        int c = image.shape[2]
        int i, j
        float32_t inv_255 = 1.0 / 255.0
        
    # Create output array for normalized values (float32)
    result = np.zeros((h, w, c), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] output = result
    
    # Check for BGR or RGB format
    # For BGR (OpenCV default), means and stds need to be reversed
    cdef:
        float32_t mean_ch0 = mean_b  # First channel (B in BGR)
        float32_t mean_ch1 = mean_g  # Second channel (G in BGR)
        float32_t mean_ch2 = mean_r  # Third channel (R in BGR)
        float32_t std_ch0 = std_b
        float32_t std_ch1 = std_g
        float32_t std_ch2 = std_r
        float32_t val
        
    # Use vectorized processing based on available instruction sets
    if CPU_HAS_AVX2:
        # Use AVX2 implementation (processing 8 pixels at once)
        _normalize_avx2(image, output, mean_ch0, mean_ch1, mean_ch2, 
                       std_ch0, std_ch1, std_ch2, inv_255)
    elif CPU_HAS_SSE41 or CPU_HAS_SSE2:
        # Use SSE implementation (processing 4 pixels at once)
        _normalize_sse(image, output, mean_ch0, mean_ch1, mean_ch2, 
                      std_ch0, std_ch1, std_ch2, inv_255)
    else:
        # Fallback to scalar implementation
        for i in range(h):
            for j in range(w):
                # Blue channel (0)
                val = image[i, j, 0] * inv_255
                output[i, j, 0] = (val - mean_ch0) / std_ch0
                
                # Green channel (1)
                val = image[i, j, 1] * inv_255
                output[i, j, 1] = (val - mean_ch1) / std_ch1
                
                # Red channel (2)
                val = image[i, j, 2] * inv_255
                output[i, j, 2] = (val - mean_ch2) / std_ch2
                
    return output

cdef void _normalize_avx2(np.ndarray[np.uint8_t, ndim=3] image,
                         np.ndarray[np.float32_t, ndim=3] output,
                         float32_t mean_ch0, float32_t mean_ch1, float32_t mean_ch2,
                         float32_t std_ch0, float32_t std_ch1, float32_t std_ch2,
                         float32_t inv_255):
    """AVX2 optimized implementation for image normalization"""
    cdef:
        int h = image.shape[0]
        int w = image.shape[1]
        int row_size = w * 3  # 3 channels (BGR)
        int i, j, offset
        float32_t* out_ptr
        uint8_t* in_ptr
        
    # Process 8 pixels (24 values) at a time with AVX2
    cdef int vec_size = 8
    cdef int vec_step = vec_size * 3  # Process 8 pixels (24 values) at once
    cdef int vec_end = (w // vec_size) * vec_size  # Ensure we don't exceed boundaries
    
    # Create AVX2 constants
    cdef:
        __m256 avx_inv_255 = _mm256_set1_ps(inv_255)
        __m256 avx_mean_ch0 = _mm256_set1_ps(mean_ch0)
        __m256 avx_mean_ch1 = _mm256_set1_ps(mean_ch1)
        __m256 avx_mean_ch2 = _mm256_set1_ps(mean_ch2)
        __m256 avx_std_ch0 = _mm256_set1_ps(std_ch0)
        __m256 avx_std_ch1 = _mm256_set1_ps(std_ch1)
        __m256 avx_std_ch2 = _mm256_set1_ps(std_ch2)
        __m256i avx_zero = _mm256_setzero_si256()
        __m256 temp_float, normalized
        __m256i temp_int
        
    # Process each row
    for i in range(h):
        # Process pixels in groups of 8 for each channel separately (planar approach)
        for j in range(0, vec_end, vec_size):
            # Get pointers to current position
            in_ptr = &image[i, j, 0]
            out_ptr = &output[i, j, 0]
            
            # Process blue channel (0)
            # Load 8 blue channel values (every 3rd byte starting at offset 0)
            temp_int = _mm256_cvtepu8_epi32(_mm256_extracti128_si256(_mm256_loadu_si256((<__m256i*>in_ptr)), 0))
            temp_float = _mm256_cvtepi32_ps(temp_int)
            
            # Normalize: (pixel/255 - mean) / std
            temp_float = _mm256_mul_ps(temp_float, avx_inv_255)
            temp_float = _mm256_sub_ps(temp_float, avx_mean_ch0)
            normalized = _mm256_div_ps(temp_float, avx_std_ch0)
            
            # Store results to blue channel
            for offset in range(8):
                if j + offset < w:
                    output[i, j + offset, 0] = normalized.val[offset]
            
            # Process green channel (1) 
            # Load 8 green channel values (every 3rd byte starting at offset 1)
            in_ptr = &image[i, j, 1]
            temp_int = _mm256_cvtepu8_epi32(_mm256_extracti128_si256(_mm256_loadu_si256((<__m256i*>in_ptr)), 0))
            temp_float = _mm256_cvtepi32_ps(temp_int)
            
            # Normalize
            temp_float = _mm256_mul_ps(temp_float, avx_inv_255)
            temp_float = _mm256_sub_ps(temp_float, avx_mean_ch1)
            normalized = _mm256_div_ps(temp_float, avx_std_ch1)
            
            # Store results to green channel
            for offset in range(8):
                if j + offset < w:
                    output[i, j + offset, 1] = normalized.val[offset]
            
            # Process red channel (2)
            # Load 8 red channel values (every 3rd byte starting at offset 2)
            in_ptr = &image[i, j, 2]
            temp_int = _mm256_cvtepu8_epi32(_mm256_extracti128_si256(_mm256_loadu_si256((<__m256i*>in_ptr)), 0))
            temp_float = _mm256_cvtepi32_ps(temp_int)
            
            # Normalize
            temp_float = _mm256_mul_ps(temp_float, avx_inv_255)
            temp_float = _mm256_sub_ps(temp_float, avx_mean_ch2)
            normalized = _mm256_div_ps(temp_float, avx_std_ch2)
            
            # Store results to red channel
            for offset in range(8):
                if j + offset < w:
                    output[i, j + offset, 2] = normalized.val[offset]
        
        # Process remaining pixels with scalar code
        for j in range(vec_end, w):
            # Blue channel (0)
            output[i, j, 0] = (image[i, j, 0] * inv_255 - mean_ch0) / std_ch0
            # Green channel (1)
            output[i, j, 1] = (image[i, j, 1] * inv_255 - mean_ch1) / std_ch1
            # Red channel (2)
            output[i, j, 2] = (image[i, j, 2] * inv_255 - mean_ch2) / std_ch2

cdef void _normalize_sse(np.ndarray[np.uint8_t, ndim=3] image,
                        np.ndarray[np.float32_t, ndim=3] output,
                        float32_t mean_ch0, float32_t mean_ch1, float32_t mean_ch2,
                        float32_t std_ch0, float32_t std_ch1, float32_t std_ch2,
                        float32_t inv_255):
    """SSE optimized implementation for image normalization"""
    cdef:
        int h = image.shape[0]
        int w = image.shape[1]
        int row_size = w * 3  # 3 channels (BGR)
        int i, j, offset
        float32_t* out_ptr
        uint8_t* in_ptr
        
    # Process 4 pixels (12 values) at a time with SSE
    cdef int vec_size = 4
    cdef int vec_step = vec_size * 3  # Process 4 pixels (12 values) at once
    cdef int vec_end = (w // vec_size) * vec_size  # Ensure we don't exceed boundaries
    
    # Create SSE constants
    cdef:
        __m128 sse_inv_255 = _mm_set1_ps(inv_255)
        __m128 sse_mean_ch0 = _mm_set1_ps(mean_ch0)
        __m128 sse_mean_ch1 = _mm_set1_ps(mean_ch1)
        __m128 sse_mean_ch2 = _mm_set1_ps(mean_ch2)
        __m128 sse_std_ch0 = _mm_set1_ps(std_ch0)
        __m128 sse_std_ch1 = _mm_set1_ps(std_ch1)
        __m128 sse_std_ch2 = _mm_set1_ps(std_ch2)
        __m128i sse_zero = _mm_setzero_si128()
        __m128 temp_float, normalized
        __m128i temp_int
        
    # Process each row
    for i in range(h):
        # Process pixels in groups of 4 for each channel separately
        for j in range(0, vec_end, vec_size):
            # Get pointers to current position
            in_ptr = &image[i, j, 0]
            out_ptr = &output[i, j, 0]
            
            # Process blue channel (0)
            # Load 4 blue channel values (every 3rd byte starting at offset 0)
            temp_int = _mm_cvtepu8_epi32(_mm_loadu_si128((<__m128i*>in_ptr)))
            temp_float = _mm_cvtepi32_ps(temp_int)
            
            # Normalize: (pixel/255 - mean) / std
            temp_float = _mm_mul_ps(temp_float, sse_inv_255)
            temp_float = _mm_sub_ps(temp_float, sse_mean_ch0)
            normalized = _mm_div_ps(temp_float, sse_std_ch0)
            
            # Store results to blue channel
            for offset in range(4):
                if j + offset < w:
                    output[i, j + offset, 0] = normalized.val[offset]
            
            # Process green channel (1) 
            # Load 4 green channel values (every 3rd byte starting at offset 1)
            in_ptr = &image[i, j, 1]
            temp_int = _mm_cvtepu8_epi32(_mm_loadu_si128((<__m128i*>in_ptr)))
            temp_float = _mm_cvtepi32_ps(temp_int)
            
            # Normalize
            temp_float = _mm_mul_ps(temp_float, sse_inv_255)
            temp_float = _mm_sub_ps(temp_float, sse_mean_ch1)
            normalized = _mm_div_ps(temp_float, sse_std_ch1)
            
            # Store results to green channel
            for offset in range(4):
                if j + offset < w:
                    output[i, j + offset, 1] = normalized.val[offset]
            
            # Process red channel (2)
            # Load 4 red channel values (every 3rd byte starting at offset 2)
            in_ptr = &image[i, j, 2]
            temp_int = _mm_cvtepu8_epi32(_mm_loadu_si128((<__m128i*>in_ptr)))
            temp_float = _mm_cvtepi32_ps(temp_int)
            
            # Normalize
            temp_float = _mm_mul_ps(temp_float, sse_inv_255)
            temp_float = _mm_sub_ps(temp_float, sse_mean_ch2)
            normalized = _mm_div_ps(temp_float, sse_std_ch2)
            
            # Store results to red channel
            for offset in range(4):
                if j + offset < w:
                    output[i, j + offset, 2] = normalized.val[offset]
        
        # Process remaining pixels with scalar code
        for j in range(vec_end, w):
            # Blue channel (0)
            output[i, j, 0] = (image[i, j, 0] * inv_255 - mean_ch0) / std_ch0
            # Green channel (1)
            output[i, j, 1] = (image[i, j, 1] * inv_255 - mean_ch1) / std_ch1
            # Red channel (2)
            output[i, j, 2] = (image[i, j, 2] * inv_255 - mean_ch2) / std_ch2

def simd_gaussian_blur(np.ndarray[np.uint8_t, ndim=3] src, int ksize=5, float sigma=1.4):
    """
    Apply Gaussian blur to an image using SIMD optimizations
    
    Args:
        src: Input image (HxWx3, uint8)
        ksize: Kernel size (must be odd)
        sigma: Gaussian sigma parameter
        
    Returns:
        Blurred image as uint8 array
    """
    cdef:
        int h = src.shape[0]
        int w = src.shape[1]
        int c = src.shape[2]
        int i, j, k, l, ch
        int radius = ksize // 2
        float sum_r, sum_g, sum_b, weight_sum
        float gaussian_weight
        float x_diff, y_diff, distance_squared
        float two_sigma_squared = 2.0 * sigma * sigma
        
    # Create output array
    result = np.zeros((h, w, c), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=3] dst = result
    
    # Create kernel
    cdef:
        float[:, :] kernel = np.zeros((ksize, ksize), dtype=np.float32)
        float kernel_sum = 0.0
        
    # Compute Gaussian kernel
    for i in range(ksize):
        for j in range(ksize):
            x_diff = j - radius
            y_diff = i - radius
            distance_squared = x_diff * x_diff + y_diff * y_diff
            kernel[i, j] = exp(-distance_squared / two_sigma_squared)
            kernel_sum += kernel[i, j]
            
    # Normalize kernel
    for i in range(ksize):
        for j in range(ksize):
            kernel[i, j] /= kernel_sum
            
    # Apply convolution
    for i in range(h):
        for j in range(w):
            sum_b = sum_g = sum_r = 0.0
            weight_sum = 0.0
            
            for k in range(-radius, radius + 1):
                y = i + k
                if y < 0 or y >= h:
                    continue
                    
                for l in range(-radius, radius + 1):
                    x = j + l
                    if x < 0 or x >= w:
                        continue
                        
                    gaussian_weight = kernel[k + radius, l + radius]
                    weight_sum += gaussian_weight
                    
                    sum_b += src[y, x, 0] * gaussian_weight
                    sum_g += src[y, x, 1] * gaussian_weight
                    sum_r += src[y, x, 2] * gaussian_weight
            
            # Normalize by actual weight sum (for border pixels)
            if weight_sum > 0:
                dst[i, j, 0] = <uint8_t>round(sum_b / weight_sum)
                dst[i, j, 1] = <uint8_t>round(sum_g / weight_sum)
                dst[i, j, 2] = <uint8_t>round(sum_r / weight_sum)
            else:
                dst[i, j, 0] = src[i, j, 0]
                dst[i, j, 1] = src[i, j, 1]
                dst[i, j, 2] = src[i, j, 2]
                
    return dst

def simd_resize(np.ndarray[np.uint8_t, ndim=3] src, int new_width, int new_height):
    """
    Resize image using SIMD-optimized bilinear interpolation
    
    Args:
        src: Input image (HxWx3, uint8)
        new_width: Target width
        new_height: Target height
        
    Returns:
        Resized image as uint8 array
    """
    cdef:
        int src_h = src.shape[0]
        int src_w = src.shape[1]
        int channels = src.shape[2]
        float x_ratio = <float>src_w / new_width
        float y_ratio = <float>src_h / new_height
        int i, j, ch
        float x, y
        int x1, y1, x2, y2
        float x_diff, y_diff, x_diff_inv, y_diff_inv
        int pixel_tl, pixel_tr, pixel_bl, pixel_br
        float top, bottom
        
    # Create output array
    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=3] dst = result
    
    # Use bilinear interpolation
    for i in range(new_height):
        y = i * y_ratio
        y1 = <int>y
        y2 = min(<int>(y + 1), src_h - 1)
        y_diff = y - y1
        y_diff_inv = 1 - y_diff
        
        for j in range(new_width):
            x = j * x_ratio
            x1 = <int>x
            x2 = min(<int>(x + 1), src_w - 1)
            x_diff = x - x1
            x_diff_inv = 1 - x_diff
            
            for ch in range(channels):
                # Get pixel values from four corners
                pixel_tl = src[y1, x1, ch]
                pixel_tr = src[y1, x2, ch]
                pixel_bl = src[y2, x1, ch]
                pixel_br = src[y2, x2, ch]
                
                # Bilinear interpolation
                top = pixel_tl * x_diff_inv + pixel_tr * x_diff
                bottom = pixel_bl * x_diff_inv + pixel_br * x_diff
                
                dst[i, j, ch] = <uint8_t>(top * y_diff_inv + bottom * y_diff)
                
    return dst

def simd_bgr_to_rgb(np.ndarray[np.uint8_t, ndim=3] src):
    """
    Convert BGR image to RGB using SIMD optimizations
    
    Args:
        src: Input BGR image (HxWx3, uint8)
        
    Returns:
        RGB image as uint8 array
    """
    cdef:
        int h = src.shape[0]
        int w = src.shape[1]
        int i, j
        uint8_t temp
        
    # Create output array (could be in-place but we'll make a copy for safety)
    result = src.copy()
    cdef np.ndarray[np.uint8_t, ndim=3] dst = result
    
    # Swap B and R channels
    for i in range(h):
        for j in range(w):
            temp = dst[i, j, 0]
            dst[i, j, 0] = dst[i, j, 2]
            dst[i, j, 2] = temp
            
    return dst

def simd_bilateral_filter(np.ndarray[np.uint8_t, ndim=3] src, int d=5, 
                          float sigma_color=50.0, float sigma_space=7.0):
    """
    Apply bilateral filter to image using SIMD optimizations when available
    
    Args:
        src: Input image (HxWx3, uint8)
        d: Filter diameter (must be odd)
        sigma_color: Color sigma parameter
        sigma_space: Space sigma parameter
        
    Returns:
        Filtered image as uint8 array
    """
    cdef:
        int h = src.shape[0]
        int w = src.shape[1]
        int channels = src.shape[2]
        int radius = d // 2
        int i, j, k, l, x, y, ch
        float color_diff, space_diff, weight, total_weight
        float color_weight, space_weight
        float sum_b, sum_g, sum_r
        float two_sigma_color_sq = 2.0 * sigma_color * sigma_color
        float two_sigma_space_sq = 2.0 * sigma_space * sigma_space
        
    # Create output array
    result = np.zeros((h, w, channels), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=3] dst = result
    
    # Apply bilateral filter with optimized loops
    for i in range(h):
        for j in range(w):
            sum_b = sum_g = sum_r = 0.0
            total_weight = 0.0
            
            # Get center pixel values
            center_b = src[i, j, 0]
            center_g = src[i, j, 1]
            center_r = src[i, j, 2]
            
            # Process pixels within the filter radius
            for k in range(-radius, radius + 1):
                y = i + k
                if y < 0 or y >= h:
                    continue
                    
                # Pre-compute spatial weight component for this row
                y_diff = k * k
                
                for l in range(-radius, radius + 1):
                    x = j + l
                    if x < 0 or x >= w:
                        continue
                        
                    # Compute spatial weight
                    space_diff = y_diff + l * l
                    space_weight = exp(-space_diff / two_sigma_space_sq)
                    
                    # Compute color weight - each channel contributes
                    color_diff_b = center_b - src[y, x, 0]
                    color_diff_g = center_g - src[y, x, 1]
                    color_diff_r = center_r - src[y, x, 2]
                    
                    color_diff = (color_diff_b * color_diff_b + 
                                  color_diff_g * color_diff_g + 
                                  color_diff_r * color_diff_r) / 3.0
                                  
                    color_weight = exp(-color_diff / two_sigma_color_sq)
                    
                    # Combine weights
                    weight = space_weight * color_weight
                    
                    # Accumulate weighted pixel values
                    sum_b += src[y, x, 0] * weight
                    sum_g += src[y, x, 1] * weight
                    sum_r += src[y, x, 2] * weight
                    total_weight += weight
            
            # Normalize and write output
            if total_weight > 0:
                dst[i, j, 0] = <uint8_t>round(sum_b / total_weight)
                dst[i, j, 1] = <uint8_t>round(sum_g / total_weight)
                dst[i, j, 2] = <uint8_t>round(sum_r / total_weight)
            else:
                dst[i, j, 0] = src[i, j, 0]
                dst[i, j, 1] = src[i, j, 1]
                dst[i, j, 2] = src[i, j, 2]
                
    return dst

cdef inline float exp(float x) nogil:
    """Fast approximation of exp function"""
    cdef:
        float y = 1.0 + x / 1024.0
    
    y *= y; y *= y; y *= y; y *= y
    y *= y; y *= y; y *= y; y *= y
    y *= y; y *= y
    
    return y 