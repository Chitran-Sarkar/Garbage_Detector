#initializer.py
import time
import sys
import os
import logging
import cv2
import ctypes
import threading
import numpy as np
from utils import get_screen_scale, resource_path, redirect_stdout_if_needed, setup_logging
from camera import find_camera_index, open_camera, list_camera_devices
from resources import load_images_from_folder
from classifier_module import load_classifier, load_ensemble_classifier, SingleClassifier, EnsembleClassifierWrapper
from servo_controller import ServoController
from config import ORIG_BG_WIDTH, ORIG_BG_HEIGHT, CLASS_MAPPING
from serial_manager import SerialManager
from ultrasonic_sensor import UltrasonicSensor

# Constants for dropdown
DROPDOWN_WIDTH = 340
DROPDOWN_HEIGHT = 40
DROPDOWN_MARGIN = 20

def initialize_app(use_async_camera=False, buffer_size=3, use_ensemble=False, 
                  models_dir=None, aggregation_method="weighted_vote", max_workers=None,
                  use_hw_accel=True, backend=cv2.CAP_DSHOW, frame_skip=0, high_performance=False,
                  no_camera=False, no_servo=False, start_fullscreen=False):
    """
    Initialize application components.
    
    Args:
        use_async_camera: Whether to use AsyncCamera instead of standard camera
        buffer_size: Frame buffer size for AsyncCamera
        use_ensemble: Whether to use ensemble classifier
        models_dir: Directory containing models for ensemble classifier
        aggregation_method: Method for aggregating ensemble predictions
        max_workers: Maximum number of worker threads for ensemble
        use_hw_accel: Whether to use hardware acceleration
        backend: Camera capture backend to use (cv2.CAP_DSHOW or cv2.CAP_FFMPEG)
        frame_skip: Number of frames to skip for performance (0=disabled)
        high_performance: Enable high performance mode with lower resolution
        no_camera: Whether to disable camera
        no_servo: Whether to disable servo
        start_fullscreen: Whether to start in fullscreen mode
        
    Returns:
        Dictionary with application context
    """
    # Set environment variables for better memory allocation
    os.environ['OPENCV_BUFFER_AREA_ALWAYS_SAFE'] = '0'  # Disable safe mode for multi-buffer allocations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Better thread handling for GPU
    os.environ['TF_GPU_THREAD_COUNT'] = '2'  # Limit GPU threads to reduce contention
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF log verbosity
    os.environ['KMP_BLOCKTIME'] = '0'  # Set thread blocktime to 0 for better CPU utilization
    os.environ['KMP_SETTINGS'] = '0'  # Disable displaying OpenMP settings
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''  # Disable OpenCL to avoid conflicts with CUDA/MKL
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    logging.info("Initializing application...")
    logging.info("Running TensorFlow on CPU.")

    # Log Windows 11 optimization information
    logging.info("Application optimized for Windows 11")
    if use_hw_accel:
        logging.info("Hardware acceleration enabled")
    
    # Log performance settings
    if high_performance:
        logging.info("High performance mode enabled")
        if frame_skip > 0:
            logging.info(f"Frame skipping enabled: {frame_skip}")
    
    backend_name = "FFMPEG" if backend == cv2.CAP_FFMPEG else "DirectShow"
    logging.info(f"Using {backend_name} camera backend")
    
    # Screen scaling
    screen_width, screen_height, scale_x, scale_y = get_screen_scale(ORIG_BG_WIDTH, ORIG_BG_HEIGHT)

    # Camera initialization - Try FIXED INDEX 0 FOR PERFORMANCE first, then fallback to find_camera_index
    init_cam_index = 0  # Default camera index
    device_name = "Default Camera"

    # Use async camera if requested
    if use_async_camera:
        logging.info("Setting up asynchronous camera mode...")

    # First try with the default camera index
    cap = open_camera(camera_index=init_cam_index, use_async=use_async_camera, 
                     buffer_size=buffer_size, use_hw_accel=use_hw_accel,
                     backend=backend, frame_skip=frame_skip, high_performance=high_performance)

    # If default camera isn't available, try to find an available camera
    if not cap:
        logging.warning(f"Unable to open camera at default index {init_cam_index}. Trying to find an available camera...")
        found_index = find_camera_index()
        
        if found_index >= 0:
            logging.info(f"Found alternative camera at index {found_index}")
            init_cam_index = found_index
            device_name = f"Camera {found_index}"
            
            # Try to open the camera with the found index
            cap = open_camera(camera_index=init_cam_index, use_async=use_async_camera, 
                             buffer_size=buffer_size, use_hw_accel=use_hw_accel,
                             backend=backend, frame_skip=frame_skip, high_performance=high_performance)
        
        # If still no camera is available, exit the application
        if not cap:
            logging.error("No accessible camera found. Exiting application.")
            sys.exit(1)

    logging.info(f"Camera Found at index {init_cam_index}.")

    # Classifier initialization
    if use_ensemble:
        logging.info("Initializing ensemble classifier...")
        
        # If models_dir is not specified, use the default Models directory
        if models_dir is None:
            models_dir = resource_path('Resources/Models')
            
        # Check if models directory exists, if not create it
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir)
                logging.info(f"Created models directory at {models_dir}")
            except Exception as e:
                logging.error(f"Failed to create models directory: {str(e)}")
                
        # If the models directory is empty, copy the default model into a subdirectory
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            logging.info("Models directory is empty. Creating default model copy...")
            try:
                # Create a default subdirectory
                default_model_dir = os.path.join(models_dir, "default_model")
                if not os.path.exists(default_model_dir):
                    os.makedirs(default_model_dir)
                
                # Copy the default model and labels to the subdirectory
                import shutil
                default_model_path = resource_path('Resources/Model/keras_model.h5')
                default_labels_path = resource_path('Resources/Model/labels.txt')
                shutil.copy2(default_model_path, os.path.join(default_model_dir, "keras_model.h5"))
                shutil.copy2(default_labels_path, os.path.join(default_model_dir, "labels.txt"))
                
                logging.info("Default model copied to ensemble directory")
            except Exception as e:
                logging.error(f"Failed to copy default model: {str(e)}")
                logging.info("Falling back to single model mode")
                use_ensemble = False
        
    if use_ensemble:
        # Create an ensemble of models
        ensemble = load_ensemble_classifier(
            models_dir=models_dir,
            # Remove the custom_models parameter that was overriding the ensemble config
            custom_models=None,
            max_workers=max_workers
        )
        
        if ensemble:
            # Set the aggregation method to weighted_vote (fixed)
            ensemble.set_aggregation_method("weighted_vote")
            
            # Wrap the ensemble classifier to match the expected interface
            classifier = EnsembleClassifierWrapper(ensemble)
            logging.info(f"Ensemble classifier initialized with {len(ensemble.models)} models")
            logging.info(f"Using weighted vote aggregation method")
        else:
            logging.error("Failed to initialize ensemble classifier. Falling back to single model.")
            classifier = SingleClassifier(resource_path('Resources/Model/keras_model.h5'),
                                         resource_path('Resources/Model/labels.txt'))
    else:
        # Use single classifier
        single_classifier = load_classifier(resource_path('Resources/Model/keras_model.h5'),
                                        resource_path('Resources/Model/labels.txt'))
        if not single_classifier:
            logging.error("Classifier model failed to load.")
            sys.exit(1)
            
        # Wrap in our interface
        classifier = SingleClassifier(resource_path('Resources/Model/keras_model.h5'),
                                     resource_path('Resources/Model/labels.txt'))

    # Resource loading
    img_arrow = cv2.imread(resource_path('Resources/arrow.png'), cv2.IMREAD_UNCHANGED)
    if img_arrow is None:
        logging.error("Arrow image not found.")
        sys.exit(1)
    img_bg = cv2.imread(resource_path('Resources/background.png'))
    if img_bg is None:
        logging.error("Background image not loaded.")
        sys.exit(1)
    img_waste_list = load_images_from_folder(resource_path('Resources/Waste'))
    img_bins_list = load_images_from_folder(resource_path('Resources/Bins'))
    if not img_waste_list or not img_bins_list:
        logging.error("Resource images failed to load.")
        sys.exit(1)

    # Serial port setup (update COM port as needed)
    serial_port = 'COM5'  # Change as needed for your system
    serial_manager = SerialManager(serial_port)
    if not serial_manager.open():
        logging.error(f"Failed to open serial port {serial_port}. Servo and sensor will not work.")
    
    # Use multi-bin logic
    NUM_BINS = 4  # Or get from config
    COOLDOWN_SECONDS = 10
    servo_controller = ServoController(serial_manager, num_bins=NUM_BINS, cooldown_seconds=COOLDOWN_SECONDS)
    servo_enabled = True  # Assume enabled if serial_manager is open
    ultrasonic_sensor = UltrasonicSensor(serial_manager, threshold_distance=30)

    logging.info("UI components initialized")
    logging.info("Application ready")

    # Return context dictionary with all initialized components
    return {
        "cap": cap,
        "classifier": classifier,
        "img_arrow": img_arrow,
        "img_bg": img_bg,
        "img_waste_list": img_waste_list,
        "img_bins_list": img_bins_list,
        "servo_controller": servo_controller,
        "servo_enabled": servo_enabled,
        "window_name": "Waste Detector",
        "screen_width": screen_width,
        "screen_height": screen_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "status_message": "",
        "status_timestamp": 0,
        "current_camera_index": init_cam_index,
        "current_device_name": device_name,
        "classifier_threshold": 0.85,  # Fixed threshold value
        "DEBOUNCE_TIME": 0.25,
        "use_ensemble": use_ensemble,
        "ultrasonic_sensor": ultrasonic_sensor
    }
