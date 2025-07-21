#ui-render.py
import cv2
import cvzone
import logging
import numpy as np
from config import CAMERA_FEED_POS, CAMERA_FEED_SIZE, WASTE_POS, ARROW_POS, BIN_POS, DISPLAY_LABELS

# Check if OpenCV UMat is available for hardware acceleration
HAS_OPENCV_UMAT = hasattr(cv2, 'UMat')

# Check if OpenCV CUDA is available
try:
    HAS_OPENCV_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except:
    HAS_OPENCV_CUDA = False


def draw_ui(img, img_bg, class_id, img_waste_list, img_arrow, img_bins_list, class_mapping,
            scale_x, scale_y, screen_width, screen_height, overlay_text=None):
    """
    Overlays the feed image onto the background and draws waste, arrow, and bin images based on classification.
    
    Args:
        img: Camera feed image
        img_bg: Background image
        class_id: Current classification ID
        img_waste_list: List of waste item images
        img_arrow: Arrow image
        img_bins_list: List of bin images
        class_mapping: Dictionary mapping class IDs to bin indices
        scale_x: Horizontal scaling factor
        scale_y: Vertical scaling factor
        screen_width: Screen width
        screen_height: Screen height
        overlay_text: Optional text to overlay on the screen
        
    Returns:
        Rendered UI image
    """
    try:
        # Make a copy of img_bg to avoid modifying the original
        if img_bg is None:
            logging.error("Background image is None, creating a blank background")
            img_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        else:
            img_bg = img_bg.copy()
            
        # Handle missing camera feed
        if img is None:
            logging.warning("Camera feed is None, creating a blank feed")
            img = np.zeros((CAMERA_FEED_SIZE[1], CAMERA_FEED_SIZE[0], 3), dtype=np.uint8)
            cv2.putText(img, "No Camera Feed", (50, CAMERA_FEED_SIZE[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Camera feed position and size scaling.
        cam_x = int(CAMERA_FEED_POS[0] * scale_x)
        cam_y = int(CAMERA_FEED_POS[1] * scale_y)
        cam_w = int(CAMERA_FEED_SIZE[0] * scale_x)
        cam_h = int(CAMERA_FEED_SIZE[1] * scale_y)

        # Resize background and camera feed.
        img_bg = cv2.resize(img_bg, (screen_width, screen_height))
        
        # Ensure the img dimensions are valid before resizing
        if img.shape[0] > 0 and img.shape[1] > 0:
            img_resized = cv2.resize(img, (cam_w, cam_h))
        else:
            logging.error("Invalid camera feed dimensions")
            img_resized = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        # If a valid waste object is detected.
        if class_id > 0 and img_waste_list:  # Make sure class_id is valid and waste list exists
            # Ensure class_id is within range of img_waste_list
            waste_idx = class_id - 1
            if 0 <= waste_idx < len(img_waste_list):
                waste_img = img_waste_list[waste_idx]
                if waste_img is not None:
                    try:
                        waste_img = cv2.resize(waste_img, (int(230 * scale_x), int(306 * scale_y)))
                        img_bg = cvzone.overlayPNG(img_bg, waste_img,
                                                (int(WASTE_POS[0] * scale_x), int(WASTE_POS[1] * scale_y)))
                    except Exception as e:
                        logging.error(f"Error overlaying waste image: {e}")
                        
            # Draw arrow if it exists
            if img_arrow is not None:
                try:
                    arrow_img = cv2.resize(img_arrow, (int(90 * scale_x), int(90 * scale_y)))
                    img_bg = cvzone.overlayPNG(img_bg, arrow_img,
                                            (int(ARROW_POS[0] * scale_x), int(ARROW_POS[1] * scale_y)))
                except Exception as e:
                    logging.error(f"Error overlaying arrow image: {e}")
                    
            # Draw bin if valid bin_index exists
            bin_index = class_mapping.get(class_id)
            if bin_index is not None and img_bins_list and 0 <= bin_index < len(img_bins_list):
                bin_img = img_bins_list[bin_index]
                if bin_img is not None:
                    try:
                        bin_img = cv2.resize(bin_img, (int(250 * scale_x), int(327 * scale_y)))
                        img_bg = cvzone.overlayPNG(img_bg, bin_img,
                                                (int(BIN_POS[0] * scale_x), int(BIN_POS[1] * scale_y)))
                    except Exception as e:
                        logging.error(f"Error overlaying bin image: {e}")
                        
        # Optionally display labels
        if DISPLAY_LABELS:
            # Display the class name and confidence
            class_names = ["Nothing"] + [f"Class {i}" for i in range(1, 8)]
            if 0 <= class_id < len(class_names):
                class_text = class_names[class_id]
                cv2.putText(img_bg, f"Detected: {class_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw overlay text if provided.
        if overlay_text:
            try:
                text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = 30
                text_y = 50
                box_coords = ((text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10))
                cv2.rectangle(img_bg, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                cv2.putText(img_bg, overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception as e:
                logging.error(f"Error drawing overlay text: {e}")

        # Place the camera feed onto the background.
        try:
            img_bg[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = img_resized
        except Exception as e:
            logging.error(f"Error placing camera feed on background: {e} - sizes: bg={img_bg.shape}, cam={img_resized.shape}")
            # Try a safer approach if the above fails
            roi = img_bg[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
            if roi.shape == img_resized.shape:
                np.copyto(roi, img_resized)
            
        return img_bg
        
    except Exception as e:
        logging.error(f"Error in draw_ui: {e}")
        # Return a simple error screen in case of failure
        error_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(error_img, "UI Rendering Error", (screen_width//4, screen_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        return error_img


def draw_dropdown(frame, current_camera_index, selected_text, camera_list, pos, size, dropdown_expanded,
                  hover_index=-1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Draws a dropdown menu for camera selection.

    - The selected box shows the chosen device.
    - When expanded, options are shown and highlight on hover.
    """
    try:
        x, y = pos
        width, height = size

        # Main selected box.
        selected_bg_color = (211, 211, 211)  # light grey
        selected_text_color = (0, 0, 0)  # black
        cv2.rectangle(frame, (x, y), (x + width, y + height), selected_bg_color, -1)
        txt_size, _ = cv2.getTextSize(selected_text, font, 0.7, 2)
        txt_y = y + (height + txt_size[1]) // 2
        cv2.putText(frame, selected_text, (x + 10, txt_y), font, 0.7, selected_text_color, 2)

        if dropdown_expanded and camera_list:
            filtered_options = [(cam_idx, cam_name) for cam_idx, cam_name in camera_list if cam_idx != current_camera_index]
            for i, (cam_idx, cam_name) in enumerate(filtered_options):
                option_top = y + height * (i + 1)
                option_bg_color = (50, 50, 50)  # dark grey
                if i == hover_index:
                    option_bg_color = (70, 70, 70)  # hover highlight
                cv2.rectangle(frame, (x, option_top), (x + width, option_top + height), option_bg_color, -1)
                txt_size, _ = cv2.getTextSize(cam_name, font, 0.7, 2)
                txt_y_option = option_top + (height + txt_size[1]) // 2
                cv2.putText(frame, cam_name, (x + 10, txt_y_option), font, 0.7, (255, 255, 255), 2)
        return frame
    except Exception as e:
        logging.error(f"Error in draw_dropdown: {e}")
        return frame


def draw_method_dropdown(frame, current_method, method_list, pos, size, dropdown_expanded,
                   hover_index=-1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Draws a dropdown menu for aggregation method selection.

    - The selected box shows the current method.
    - When expanded, options are shown and highlight on hover.
    """
    try:
        x, y = pos
        width, height = size

        # Main selected box.
        selected_bg_color = (20, 150, 120)  # teal
        selected_text_color = (255, 255, 255)  # white
        cv2.rectangle(frame, (x, y), (x + width, y + height), selected_bg_color, -1)
        txt_size, _ = cv2.getTextSize(current_method, font, 0.7, 2)
        txt_y = y + (height + txt_size[1]) // 2
        cv2.putText(frame, current_method, (x + 10, txt_y), font, 0.7, selected_text_color, 2)

        if dropdown_expanded and method_list:
            filtered_options = [(method_id, method_name) for method_id, method_name in method_list if method_name != current_method]
            for i, (method_id, method_name) in enumerate(filtered_options):
                option_top = y + height * (i + 1)
                option_bg_color = (20, 100, 80)  # darker teal for options
                if i == hover_index:
                    option_bg_color = (40, 180, 140)  # lighter teal for hover
                cv2.rectangle(frame, (x, option_top), (x + width, option_top + height), option_bg_color, -1)
                txt_size, _ = cv2.getTextSize(method_name, font, 0.7, 2)
                txt_y_option = option_top + (height + txt_size[1]) // 2
                cv2.putText(frame, method_name, (x + 10, txt_y_option), font, 0.7, (255, 255, 255), 2)
        return frame
    except Exception as e:
        logging.error(f"Error in draw_method_dropdown: {e}")
        return frame


def draw_ui_optimized(img, img_bg, class_id, img_waste_list, img_arrow, img_bins_list, class_mapping,
                     scale_x, scale_y, screen_width, screen_height, overlay_text=None, memory_manager=None):
    """
    Optimized version of draw_ui using vectorized operations and hardware acceleration when available.
    
    Args:
        img: Camera feed image
        img_bg: Background image
        class_id: Current classification ID
        img_waste_list: List of waste item images
        img_arrow: Arrow image
        img_bins_list: List of bin images
        class_mapping: Dictionary mapping class IDs to bin indices
        scale_x: Horizontal scaling factor
        scale_y: Vertical scaling factor
        screen_width: Screen width
        screen_height: Screen height
        overlay_text: Optional text to overlay on the screen
        memory_manager: Optional memory manager for buffer allocation
        
    Returns:
        Rendered UI image
    """
    try:
        # Use memory manager if provided
        if memory_manager:
            # Get preallocated buffer for output image
            output_img = memory_manager.get_buffer((screen_height, screen_width, 3), np.uint8, "ui_output")
        else:
            # Create new output buffer
            output_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
        # Prepare background
        if img_bg is None:
            # Just use the zeroed buffer as is
            pass
        else:
            # Use hardware acceleration for resize if available
            if HAS_OPENCV_CUDA:
                try:
                    # Upload to GPU
                    gpu_img_bg = cv2.cuda_GpuMat()
                    gpu_img_bg.upload(img_bg)
                    
                    # Resize on GPU
                    gpu_bg_resized = cv2.cuda.resize(gpu_img_bg, (screen_width, screen_height))
                    
                    # Download result
                    bg_resized = gpu_bg_resized.download()
                    
                    # Copy to output
                    np.copyto(output_img, bg_resized)
                except Exception:
                    # Fall back to CPU resize
                    bg_resized = cv2.resize(img_bg, (screen_width, screen_height))
                    np.copyto(output_img, bg_resized)
            elif HAS_OPENCV_UMAT:
                try:
                    # Use OpenCL acceleration
                    umat_bg = cv2.UMat(img_bg)
                    bg_resized = cv2.resize(umat_bg, (screen_width, screen_height))
                    bg_resized = bg_resized.get()  # Get from UMat
                    np.copyto(output_img, bg_resized)
                except Exception:
                    # Fall back to CPU resize
                    bg_resized = cv2.resize(img_bg, (screen_width, screen_height))
                    np.copyto(output_img, bg_resized)
            else:
                # Standard CPU resize with optimized copy
                bg_resized = cv2.resize(img_bg, (screen_width, screen_height))
                np.copyto(output_img, bg_resized)
        
        # Camera feed position and size scaling
        cam_x = int(CAMERA_FEED_POS[0] * scale_x)
        cam_y = int(CAMERA_FEED_POS[1] * scale_y)
        cam_w = int(CAMERA_FEED_SIZE[0] * scale_x)
        cam_h = int(CAMERA_FEED_SIZE[1] * scale_y)
        
        # Prepare camera feed
        if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            # Create blank camera feed
            camera_roi = output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
            camera_roi.fill(0)  # Fill with black
            
            # Add "No Camera Feed" text
            text = "No Camera Feed"
            cv2.putText(camera_roi, text, 
                      (camera_roi.shape[1]//2 - 80, camera_roi.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Resize camera feed with hardware acceleration if available
            if HAS_OPENCV_CUDA:
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(img)
                    gpu_img_resized = cv2.cuda.resize(gpu_img, (cam_w, cam_h))
                    img_resized = gpu_img_resized.download()
                    
                    # Copy to output using direct memory copy
                    camera_roi = output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
                    np.copyto(camera_roi, img_resized)
                except Exception:
                    # Fall back to CPU
                    img_resized = cv2.resize(img, (cam_w, cam_h))
                    output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = img_resized
            elif HAS_OPENCV_UMAT:
                try:
                    umat_img = cv2.UMat(img)
                    img_resized = cv2.resize(umat_img, (cam_w, cam_h)).get()
                    output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = img_resized
                except Exception:
                    # Fall back to CPU
                    img_resized = cv2.resize(img, (cam_w, cam_h))
                    output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = img_resized
            else:
                # CPU resize
                img_resized = cv2.resize(img, (cam_w, cam_h))
                output_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = img_resized
        
        # If a valid waste object is detected
        if class_id > 0 and img_waste_list:
            waste_idx = class_id - 1
            if 0 <= waste_idx < len(img_waste_list):
                waste_img = img_waste_list[waste_idx]
                if waste_img is not None:
                    try:
                        # Optimized PNG overlay using alpha compositing
                        waste_w, waste_h = int(230 * scale_x), int(306 * scale_y)
                        waste_x, waste_y = int(WASTE_POS[0] * scale_x), int(WASTE_POS[1] * scale_y)
                        
                        # Resize waste image
                        waste_resized = cv2.resize(waste_img, (waste_w, waste_h))
                        
                        # Check if we have alpha channel (4th channel)
                        if waste_resized.shape[2] == 4:
                            # Extract alpha channel
                            alpha = waste_resized[:, :, 3] / 255.0
                            
                            # Extract BGR channels
                            waste_bgr = waste_resized[:, :, :3]
                            
                            # Get ROI from output image
                            roi_y_end = min(waste_y + waste_h, output_img.shape[0])
                            roi_x_end = min(waste_x + waste_w, output_img.shape[1])
                            roi_h = roi_y_end - waste_y
                            roi_w = roi_x_end - waste_x
                            
                            # Adjust waste image size if ROI is smaller
                            waste_bgr = waste_bgr[:roi_h, :roi_w]
                            alpha = alpha[:roi_h, :roi_w]
                            
                            # Get ROI from output image
                            roi = output_img[waste_y:roi_y_end, waste_x:roi_x_end]
                            
                            # Ensure shapes match
                            if roi.shape[:2] == waste_bgr.shape[:2]:
                                # Vectorized alpha compositing
                                alpha = np.expand_dims(alpha, axis=2)  # 2D -> 3D
                                roi[:] = (waste_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                    except Exception as e:
                        logging.error(f"Optimized waste overlay error: {e}")
                
                # Draw arrow if it exists
                if img_arrow is not None:
                    try:
                        arrow_w, arrow_h = int(90 * scale_x), int(90 * scale_y)
                        arrow_x, arrow_y = int(ARROW_POS[0] * scale_x), int(ARROW_POS[1] * scale_y)
                        
                        # Resize arrow image
                        arrow_resized = cv2.resize(img_arrow, (arrow_w, arrow_h))
                        
                        # Check if we have alpha channel
                        if arrow_resized.shape[2] == 4:
                            # Extract alpha channel
                            alpha = arrow_resized[:, :, 3] / 255.0
                            
                            # Extract BGR channels
                            arrow_bgr = arrow_resized[:, :, :3]
                            
                            # Get ROI from output image
                            roi_y_end = min(arrow_y + arrow_h, output_img.shape[0])
                            roi_x_end = min(arrow_x + arrow_w, output_img.shape[1])
                            roi_h = roi_y_end - arrow_y
                            roi_w = roi_x_end - arrow_x
                            
                            # Adjust arrow image size if ROI is smaller
                            arrow_bgr = arrow_bgr[:roi_h, :roi_w]
                            alpha = alpha[:roi_h, :roi_w]
                            
                            # Get ROI from output image
                            roi = output_img[arrow_y:roi_y_end, arrow_x:roi_x_end]
                            
                            # Ensure shapes match
                            if roi.shape[:2] == arrow_bgr.shape[:2]:
                                # Vectorized alpha compositing
                                alpha = np.expand_dims(alpha, axis=2)  # 2D -> 3D
                                roi[:] = (arrow_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                    except Exception as e:
                        logging.error(f"Optimized arrow overlay error: {e}")
                
                # Draw bin if valid bin_index exists
                bin_index = class_mapping.get(class_id)
                if bin_index is not None and img_bins_list and 0 <= bin_index < len(img_bins_list):
                    bin_img = img_bins_list[bin_index]
                    if bin_img is not None:
                        try:
                            bin_w, bin_h = int(250 * scale_x), int(327 * scale_y)
                            bin_x, bin_y = int(BIN_POS[0] * scale_x), int(BIN_POS[1] * scale_y)
                            
                            # Resize bin image
                            bin_resized = cv2.resize(bin_img, (bin_w, bin_h))
                            
                            # Check if we have alpha channel
                            if bin_resized.shape[2] == 4:
                                # Extract alpha channel
                                alpha = bin_resized[:, :, 3] / 255.0
                                
                                # Extract BGR channels
                                bin_bgr = bin_resized[:, :, :3]
                                
                                # Get ROI from output image
                                roi_y_end = min(bin_y + bin_h, output_img.shape[0])
                                roi_x_end = min(bin_x + bin_w, output_img.shape[1])
                                roi_h = roi_y_end - bin_y
                                roi_w = roi_x_end - bin_x
                                
                                # Adjust bin image size if ROI is smaller
                                bin_bgr = bin_bgr[:roi_h, :roi_w]
                                alpha = alpha[:roi_h, :roi_w]
                                
                                # Get ROI from output image
                                roi = output_img[bin_y:roi_y_end, bin_x:roi_x_end]
                                
                                # Ensure shapes match
                                if roi.shape[:2] == bin_bgr.shape[:2]:
                                    # Vectorized alpha compositing
                                    alpha = np.expand_dims(alpha, axis=2)  # 2D -> 3D
                                    roi[:] = (bin_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                        except Exception as e:
                            logging.error(f"Optimized bin overlay error: {e}")
        
        # Optionally display labels using optimized text rendering
        if DISPLAY_LABELS and 0 <= class_id < 8:
            # Pre-defined class names
            class_names = ["Nothing"] + [f"Class {i}" for i in range(1, 8)]
            class_text = class_names[class_id]
            label = f"Detected: {class_text}"
            
            # Optimized text rendering - pre-calculate size once
            font_scale = 0.7
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Create text background for better visibility
            text_x, text_y = 10, 30
            bg_rect = np.array([
                [text_x - 5, text_y - text_size[1] - 5],
                [text_x + text_size[0] + 5, text_y + 5]
            ])
            
            # Draw rectangle and text in one shot
            cv2.rectangle(output_img, 
                        (bg_rect[0][0], bg_rect[0][1]), 
                        (bg_rect[1][0], bg_rect[1][1]), 
                        (0, 0, 0), 
                        cv2.FILLED)
                        
            cv2.putText(output_img, 
                      label, 
                      (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale, 
                      (255, 255, 255), 
                      font_thickness,
                      cv2.LINE_AA)
        
        # Draw overlay text if provided
        if overlay_text:
            try:
                # Optimize text rendering with precomputed values
                font_scale = 1.2
                font_thickness = 3
                text_size, _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                text_x, text_y = 30, 50
                box_pt1 = (text_x - 10, text_y - text_size[1] - 10)
                box_pt2 = (text_x + text_size[0] + 10, text_y + 10)
                
                # Draw background and text in one operation
                cv2.rectangle(output_img, box_pt1, box_pt2, (0, 0, 0), cv2.FILLED)
                cv2.putText(output_img, overlay_text, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 
                          font_thickness, cv2.LINE_AA)
            except Exception as e:
                logging.error(f"Optimized overlay text error: {e}")
        
        return output_img
        
    except Exception as e:
        logging.error(f"Error in draw_ui_optimized: {e}")
        # Return a simple error screen in case of failure
        if memory_manager:
            error_img = memory_manager.get_buffer((screen_height, screen_width, 3), np.uint8, "error_img")
        else:
            error_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
        cv2.putText(error_img, "UI Rendering Error", (screen_width//4, screen_height//2), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        return error_img

# Function to overlay PNG images with alpha channel using vectorized operations
def fast_overlay_png(background, overlay, pos):
    """
    Overlay a PNG image with alpha channel onto a background using vectorized operations.
    Much faster than the standard cvzone.overlayPNG function.
    
    Args:
        background: Background image as numpy array
        overlay: Overlay image with alpha channel as numpy array (BGRA)
        pos: Position (x, y) to place the overlay
        
    Returns:
        Background image with overlay composited
    """
    x, y = pos
    
    # Check if overlay has alpha channel
    if overlay.shape[2] != 4:
        return background
        
    # Extract overlay dimensions
    h, w = overlay.shape[:2]
    
    # Calculate ROI boundaries
    y_end = min(y + h, background.shape[0])
    x_end = min(x + w, background.shape[1])
    
    # Check if ROI is valid
    if y >= background.shape[0] or x >= background.shape[1] or y_end <= y or x_end <= x:
        return background
        
    # Calculate visible portion of overlay
    h_visible = y_end - y
    w_visible = x_end - x
    
    # Get ROI from background
    roi = background[y:y_end, x:x_end]
    
    # Extract visible portion of overlay
    overlay_visible = overlay[:h_visible, :w_visible]
    
    # Extract alpha channel and normalize to 0-1
    alpha = overlay_visible[:, :, 3:4] / 255.0
    
    # Perform alpha blending - vectorized
    blended = overlay_visible[:, :, :3] * alpha + roi * (1.0 - alpha)
    
    # Update background ROI with blended result
    background[y:y_end, x:x_end] = blended.astype(np.uint8)
    
    return background
