# fast_operations.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport round

ctypedef unsigned char uint8_t

def alpha_blend(np.ndarray[np.uint8_t, ndim=3] background, 
                np.ndarray[np.uint8_t, ndim=3] foreground,
                np.ndarray[np.float32_t, ndim=2] alpha,
                int x, int y):
    """
    Fast alpha blending operation using Cython
    
    Args:
        background: Background image (HxWx3)
        foreground: Foreground image (hxwx3)
        alpha: Alpha channel (hxw)
        x, y: Position to place foreground on background
    
    Returns:
        None (modifies background in-place)
    """
    cdef:
        int h = foreground.shape[0]
        int w = foreground.shape[1]
        int bg_h = background.shape[0]
        int bg_w = background.shape[1]
        int i, j, ch
        int roi_y_end = min(y + h, bg_h)
        int roi_x_end = min(x + w, bg_w)
        int roi_h = roi_y_end - y
        int roi_w = roi_x_end - x
        float a, inv_a
    
    # Bounds checking
    if x < 0 or y < 0 or x >= bg_w or y >= bg_h:
        return
    
    # Process each pixel
    for i in range(roi_h):
        for j in range(roi_w):
            a = alpha[i, j]
            inv_a = 1.0 - a
            for ch in range(3):  # RGB channels
                background[y+i, x+j, ch] = <uint8_t>(
                    round(foreground[i, j, ch] * a + background[y+i, x+j, ch] * inv_a)
                )

def fast_resize(np.ndarray[np.uint8_t, ndim=3] src, int new_w, int new_h):
    """
    Fast image resize using nearest neighbor interpolation
    
    Args:
        src: Source image (HxWx3 or HxWx4)
        new_w: New width
        new_h: New height
    
    Returns:
        Resized image
    """
    cdef:
        int src_h = src.shape[0]
        int src_w = src.shape[1]
        int channels = src.shape[2]
        int i, j, ch
        float ratio_h = src_h / <float>new_h
        float ratio_w = src_w / <float>new_w
        int src_i, src_j
    
    # Create destination array
    cdef np.ndarray[np.uint8_t, ndim=3] dst = np.zeros((new_h, new_w, channels), dtype=np.uint8)
    
    # Process each pixel
    for i in range(new_h):
        src_i = <int>(i * ratio_h)
        if src_i >= src_h:
            src_i = src_h - 1
            
        for j in range(new_w):
            src_j = <int>(j * ratio_w)
            if src_j >= src_w:
                src_j = src_w - 1
                
            for ch in range(channels):
                dst[i, j, ch] = src[src_i, src_j, ch]
    
    return dst

def fast_crop(np.ndarray[np.uint8_t, ndim=3] src, int x, int y, int w, int h):
    """
    Fast image cropping with bounds checking
    
    Args:
        src: Source image
        x, y: Top-left corner of crop region
        w, h: Width and height of crop region
    
    Returns:
        Cropped image
    """
    cdef:
        int src_h = src.shape[0]
        int src_w = src.shape[1]
        int channels = src.shape[2]
        int crop_x = max(0, x)
        int crop_y = max(0, y)
        int crop_w = min(w, src_w - crop_x)
        int crop_h = min(h, src_h - crop_y)
    
    # Check if crop region is valid
    if crop_w <= 0 or crop_h <= 0:
        return np.zeros((1, 1, channels), dtype=np.uint8)
    
    # Create destination array
    cdef np.ndarray[np.uint8_t, ndim=3] dst = np.zeros((crop_h, crop_w, channels), dtype=np.uint8)
    
    # Copy data
    dst = src[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :].copy()
    
    return dst

def fast_copy_to(np.ndarray[np.uint8_t, ndim=3] dst, 
                np.ndarray[np.uint8_t, ndim=3] src, 
                int x, int y):
    """
    Fast copy of src into dst at position (x,y) with bounds checking
    
    Args:
        dst: Destination image
        src: Source image
        x, y: Position to place source on destination
    
    Returns:
        None (modifies dst in-place)
    """
    cdef:
        int src_h = src.shape[0]
        int src_w = src.shape[1]
        int dst_h = dst.shape[0]
        int dst_w = dst.shape[1]
        int channels = min(src.shape[2], dst.shape[2])
        int roi_x = max(0, x)
        int roi_y = max(0, y)
        int roi_w = min(src_w, dst_w - roi_x)
        int roi_h = min(src_h, dst_h - roi_y)
        int src_offset_x = roi_x - x
        int src_offset_y = roi_y - y
    
    # Check if ROI is valid
    if roi_w <= 0 or roi_h <= 0 or channels <= 0:
        return
    
    # Copy with bounds checking
    dst[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :channels] = \
        src[src_offset_y:src_offset_y+roi_h, src_offset_x:src_offset_x+roi_w, :channels] 