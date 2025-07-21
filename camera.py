#camera.py
import cv2
import sys
import logging
import time
from pygrabber.dshow_graph import FilterGraph

def check_camera_feed(index, cap_flag=cv2.CAP_DSHOW):
    """
    Attempts to open the camera and grab a frame using the specified backend.
    
    Args:
        index: The camera index to check
        cap_flag: The backend flag to use (cv2.CAP_DSHOW, cv2.CAP_FFMPEG, cv2.CAP_ANY, etc.)
        
    Returns:
        True if successfully opened and grabbed a frame, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(index, cap_flag)
        if not cap.isOpened():
            logging.debug(f"Camera {index} could not be opened with backend {cap_flag}")
            return False
            
        # Try to grab a frame with timeout
        start_time = time.time()
        frame_grabbed = False
        timeout = 2.0  # 2 seconds timeout
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_grabbed = True
                break
            time.sleep(0.1)
        
        cap.release()
        
        if frame_grabbed:
            logging.debug(f"Successfully grabbed frame from camera {index} with backend {cap_flag}")
            return True
        else:
            logging.debug(f"Failed to grab frame from camera {index} with backend {cap_flag} within timeout")
            return False
    except Exception as e:
        logging.debug(f"Error checking camera {index} with backend {cap_flag}: {str(e)}")
        return False


def find_camera_index():
    """
    Finds and returns the first accessible camera index using DirectShow.
    
    The function first tries to use the FilterGraph to enumerate DirectShow devices,
    then falls back to scanning indices if that fails. It prioritizes cameras that
    can successfully capture frames.

    Returns:
        An integer index or -1 if no camera is found.
    """
    found_devices = []
    
    try:
        # Use pygrabber to get the list of available devices
        graph = FilterGraph()
        devices = graph.get_input_devices()
        logging.info(f"Found {len(devices)} camera devices using DirectShow graph")

        for index, name in enumerate(devices):
            if check_camera_feed(index):
                logging.info(f"Found working camera: {name} (index {index})")
                found_devices.append((index, name, True))  # Working camera
            else:
                logging.debug(f"Found non-working camera device: {name} (index {index})")
                found_devices.append((index, name, False))  # Non-working camera

        # If no working camera is found with pygrabber but devices were detected
        if not any(device[2] for device in found_devices) and found_devices:
            logging.warning("Detected cameras, but none are working with DirectShow graph. Trying with different backend...")
            # Try with a different backend (FFMPEG) for devices found by DirectShow
            for index, name, _ in found_devices:
                if check_camera_feed(index, cv2.CAP_FFMPEG):
                    logging.info(f"Camera works with FFMPEG backend: {name} (index {index})")
                    return index
    except Exception as e:
        logging.warning(f"Error using DirectShow graph to find cameras: {str(e)}. Scanning indices...")

    # If we found devices with DirectShow, prioritize working ones
    working_devices = [device for device in found_devices if device[2]]
    if working_devices:
        index, name, _ = working_devices[0]
        logging.info(f"Using first working camera: {name} (index {index})")
        return index
    
    # Fall back to scanning indices if no working camera was found with DirectShow
    logging.info("DirectShow didn't find working cameras. Scanning indices directly...")
    for index in range(10):
        # First try DirectShow backend
        if check_camera_feed(index, cv2.CAP_DSHOW):
            logging.info(f"Found working camera at index {index} using DirectShow")
            return index
        # Then try FFMPEG backend
        elif check_camera_feed(index, cv2.CAP_FFMPEG):
            logging.info(f"Found working camera at index {index} using FFMPEG")
            return index
        # Skip quickly if the camera doesn't exist at all
        elif not check_camera_feed(index, cv2.CAP_ANY):
            continue

    logging.error("No working camera found after scanning all methods")
    return -1


def open_camera(camera_index, use_async=False, buffer_size=3, use_hw_accel=True, 
                backend=cv2.CAP_DSHOW, frame_skip=0, high_performance=False):
    """
    Open a camera by index optimized for Windows.
    
    This function will try the specified backend first. If that fails, it will
    attempt other backends as fallbacks.
    
    Args:
        camera_index: Index of the camera to open
        use_async: Whether to use AsyncCamera for threaded capture
        buffer_size: Size of the frame buffer for AsyncCamera
        use_hw_accel: Whether to enable hardware acceleration
        backend: Capture backend to use (cv2.CAP_DSHOW or cv2.CAP_FFMPEG)
        frame_skip: Number of frames to skip for performance (only for AsyncCamera)
        high_performance: Enable additional performance tuning options
        
    Returns:
        Camera object (either cv2.VideoCapture or AsyncCamera) or None if all attempts fail
    """
    logging.info(f"Attempting to open camera at index {camera_index}")
    
    # List of backends to try in order if the primary backend fails
    backends_to_try = [backend]  # Start with the requested backend
    
    # Add alternative backends if not already included
    if backend != cv2.CAP_DSHOW and check_camera_feed(camera_index, cv2.CAP_DSHOW):
        backends_to_try.append(cv2.CAP_DSHOW)
    if backend != cv2.CAP_FFMPEG and check_camera_feed(camera_index, cv2.CAP_FFMPEG):
        backends_to_try.append(cv2.CAP_FFMPEG)
    if backend != cv2.CAP_ANY and backend != 0 and check_camera_feed(camera_index, cv2.CAP_ANY):
        backends_to_try.append(cv2.CAP_ANY)
    
    logging.info(f"Will try the following backends: {backends_to_try}")
    
    # Disable use of AsyncCamera due to issues
    use_async = False
    
    # Try each backend in the list
    for current_backend in backends_to_try:
        try:
            if use_async:
                # AsyncCamera is optional and only used if specifically requested
                logging.info(f"Opening asynchronous camera at index {camera_index} with backend {current_backend}")
                from async_camera import AsyncCamera
                camera = AsyncCamera(camera_index, buffer_size=buffer_size, use_hw_accel=use_hw_accel,
                                    backend=current_backend, frame_skip=frame_skip, high_performance=high_performance)
                if not camera.is_opened():
                    logging.warning(f"Failed to open AsyncCamera at index {camera_index} with backend {current_backend}")
                    continue  # Try next backend
            else:
                # Use standard synchronous camera with the current backend
                if current_backend == cv2.CAP_FFMPEG:
                    backend_name = "FFMPEG"
                elif current_backend == cv2.CAP_DSHOW:
                    backend_name = "DirectShow"
                else:
                    backend_name = "Any/Default"
                    
                logging.info(f"Opening camera at index {camera_index} with {backend_name} backend")
                camera = cv2.VideoCapture(camera_index, current_backend)
                    
                if not camera.isOpened():
                    logging.warning(f"Failed to open camera at index {camera_index} with {backend_name} backend")
                    continue  # Try next backend
                
                # Set buffer size based on performance settings
                if high_performance:
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, max(3, buffer_size // 2))
                
                # Try to set higher resolution for better image quality if available
                # If high_performance is enabled, use a lower resolution
                if high_performance:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                else:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Try to enable hardware acceleration if requested
                if use_hw_accel:
                    hw_accel_result = camera.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    if hw_accel_result:
                        logging.info("Hardware acceleration enabled for camera")
                    else:
                        logging.warning("Failed to enable hardware acceleration for camera - not supported by device")
                        
                # Additional performance optimizations for standard camera
                if high_performance:
                    try:
                        # Try to disable auto-exposure for faster frame rate
                        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                        # Try to reduce frame precision if supported
                        camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                        logging.info("Applied additional performance optimizations")
                    except Exception as e:
                        logging.warning(f"Some performance optimizations not supported: {e}")
            
            # Verify camera works by reading a test frame
            ret, frame = camera.read()
            if not ret or frame is None or frame.size == 0:
                logging.warning(f"Camera opened but failed to read a valid frame with {backend_name} backend")
                camera.release()
                continue  # Try next backend
            
            # Get camera properties for logging
            if isinstance(camera, cv2.VideoCapture):
                width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = camera.get(cv2.CAP_PROP_FPS)
                hw_accel = camera.get(cv2.CAP_PROP_HW_ACCELERATION)
                logging.info(f"Camera opened successfully: {width:.0f}x{height:.0f} @ {fps:.1f}fps")
                logging.info(f"Hardware acceleration status: {hw_accel}")
                logging.info(f"Backend: {backend_name}")
            else:
                logging.info(f"Asynchronous camera opened successfully at index {camera_index}")
                
            return camera
            
        except Exception as e:
            logging.error(f"Error opening camera with backend {current_backend}: {str(e)}")
            continue  # Try next backend
    
    # If we reach here, all backends failed
    logging.error(f"Failed to open camera at index {camera_index} with any backend")
    return None


def list_camera_devices():
    """
    Lists available cameras using Windows DirectShow.

    Returns:
        A list of tuples (index, device name)
    """
    devices = []
    try:
        # Use pygrabber to get device names (Windows-specific)
        graph = FilterGraph()
        device_names = graph.get_input_devices()
        for i, name in enumerate(device_names):
            if check_camera_feed(i):
                devices.append((i, name))
        
        if devices:
            logging.info(f"Found {len(devices)} camera devices")
            return devices
    except Exception as e:
        logging.error(f"Error listing camera devices with pygrabber: {str(e)}")
    
    # Fallback to simple index scan if pygrabber fails
    logging.warning("Falling back to index scanning for camera devices")
    for i in range(10):
        if check_camera_feed(i):
            devices.append((i, f"Camera {i}"))
    
    return devices


# Simple Camera class that wraps cv2.VideoCapture with Windows optimizations
class Camera:
    """Camera implementation optimized for Windows DirectShow."""
    
    def __init__(self, camera_index=0):
        """
        Initialize the Windows camera.
        
        Args:
            camera_index: The camera device index
        """
        self.camera_index = camera_index
        self.cap = None
        
    def start(self):
        """
        Open the camera and start capturing using DirectShow.
        
        Returns:
            True if successfully started, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Try to enable hardware acceleration if available
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            # Try to set higher resolution for better image quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Try to get the first frame to confirm camera works
            ret, _ = self.cap.read()
            if not ret:
                logging.error(f"Failed to capture initial frame from camera {self.camera_index}")
                self.cap.release()
                self.cap = None
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error starting camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def read(self):
        """
        Read the next frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        return self.cap.read()
        
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def isOpened(self):
        """
        Check if the camera is opened.
        
        Returns:
            True if the camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
        
    def get(self, prop_id):
        """
        Get a camera property.
        
        Args:
            prop_id: The property ID (cv2.CAP_PROP_*)
            
        Returns:
            The property value
        """
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(prop_id)
        return 0
        
    def set(self, prop_id, value):
        """
        Set a camera property.
        
        Args:
            prop_id: The property ID (cv2.CAP_PROP_*)
            value: The value to set
            
        Returns:
            True if successful, False otherwise
        """
        if self.cap is not None and self.cap.isOpened():
            return self.cap.set(prop_id, value)
        return False
