#app_loop.py
import cv2
import sys
import ctypes
import time
import threading
import logging
import queue
import numpy as np
from ctypes import wintypes
from ui_renderer import draw_ui, draw_ui_optimized, fast_overlay_png
from config import CLASS_MAPPING
from utils import resource_path, BackgroundTaskScheduler, HardwareMonitor, MemoryManager, PriorityTaskScheduler, PerformanceMonitor
import gc
import traceback
import weakref
from stream_server import update_frame

# Windows API constants
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
KEY_F11 = 0x0100003A  # F11 key code for OpenCV waitKeyEx

# UI update queue for async rendering
ui_update_queue = queue.Queue(maxsize=5)

# Global managers and schedulers
background_scheduler = None
priority_scheduler = None
memory_manager = None
performance_monitor = None
hardware_monitor = None

# UI rendering state with lock protection
ui_rendering_state = {
    "last_output_img": None,
    "last_class_id": 0,
    "last_update_time": 0,
    "rendering_in_progress": False,
    "frame_count": 0,
    "last_fps_time": 0,
    "fps": 0
}

# Thread-safe UI state access
ui_state_lock = threading.RLock()

# Performance settings
USE_OPTIMIZED_RENDERING = True
USE_MEMORY_LOCKING = True
USE_PRIORITY_SCHEDULING = True
USE_PERFORMANCE_MONITORING = True

# Memory leak detection
memory_tracking = {
    "active_buffers": weakref.WeakSet(),
    "last_check_time": time.time(),
    "check_interval": 30.0,  # Check for leaks every 30 seconds
    "last_buffer_count": 0
}

# Platform detection
IS_WINDOWS = sys.platform.startswith('win')

def set_window_dark_mode(hwnd, enable=True):
    """Enable dark mode for Windows 10+ windows"""
    if not IS_WINDOWS:
        return False
        
    try:
        attribute = wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE)
        value = wintypes.BOOL(1 if enable else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd,
                                                attribute,
                                                ctypes.byref(value),
                                                ctypes.sizeof(value))
        return True
    except (AttributeError, OSError):
        return False

def set_window_icon(window_name, icon_path):
    """Set window icon using Windows API"""
    if not IS_WINDOWS:
        return False
        
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            hicon = ctypes.windll.user32.LoadImageW(0, icon_path, 1, 0, 0, 0x00000010)
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)
            set_window_dark_mode(hwnd, enable=True)
            return True
        return False
    except Exception as e:
        logging.error(f"Error setting window icon: {e}")
        return False

def set_process_priority_high():
    """Set process priority to high for better responsiveness (Windows)"""
    if not IS_WINDOWS:
        return False
        
    try:
        try:
            import win32process
            import win32api
            import win32con
        except ImportError:
            logging.warning("win32api not found, process priority not adjusted")
            return False
        
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
        win32api.CloseHandle(handle)
        
        logging.info("Set application process priority to HIGH_PRIORITY_CLASS")
        return True
    except Exception as e:
        logging.error(f"Failed to set process priority: {str(e)}")
        return False

def check_for_memory_leaks():
    """Check for potential memory leaks in buffer management"""
    global memory_tracking
    
    current_time = time.time()
    if current_time - memory_tracking["last_check_time"] < memory_tracking["check_interval"]:
        return  # Not time to check yet
        
    buffer_count = len(memory_tracking["active_buffers"])
    
    if buffer_count > memory_tracking["last_buffer_count"] + 10:
        logging.warning(f"Potential memory leak detected: Buffer count increased from "
                      f"{memory_tracking['last_buffer_count']} to {buffer_count}")
        
        gc.collect()
        
        buffer_count = len(memory_tracking["active_buffers"])
        
        if buffer_count > memory_tracking["last_buffer_count"] + 5:
            logging.error(f"Memory leak confirmed: Buffer count is still {buffer_count} after garbage collection")
            
    memory_tracking["last_buffer_count"] = buffer_count
    memory_tracking["last_check_time"] = current_time

def draw_performance_metrics(img, fps, ensemble=None, use_ensemble=False, confidence=0.0):
    """Draw performance metrics on the image with translucent background"""
    h, w = img.shape[:2]
    
    metrics = []
    metrics.append(f"FPS: {fps:.1f}")
    
    if use_ensemble and ensemble:
        try:
            ensemble_stats = ensemble.get_ensemble_stats()
            model_count = ensemble_stats.get("model_count", 0)
            total_models = ensemble_stats.get("total_models", 0)
            
            system = ensemble_stats.get("system_metrics", {})
            cpu = system.get("cpu_percent", 0)
            
            gpu_usage = system.get("gpu_percent", 0)
            
            metrics.append(f"CPU: {cpu:.1f}%")
            metrics.append(f"GPU: {gpu_usage:.1f}%")
            metrics.append(f"Confidence: {confidence:.2f}")
            metrics.append(f"Models: {model_count}/{total_models}")
            
            avg_time = ensemble_stats.get('avg_inference_time', 0) * 1000  # Convert to ms
            metrics.append(f"Ensemble time: {avg_time:.1f}ms")
            
            if "preprocessing_time" in ensemble_stats:
                preprocessing = ensemble_stats.get("preprocessing_time", 0) * 1000  # Convert to ms
                metrics.append(f"Preproc: {preprocessing:.1f}ms")

        except Exception as e:
            logging.error(f"Error showing ensemble stats: {e}")
    
    if performance_monitor is not None:
        try:
            status = performance_monitor.get_status()
            
            if 'memory_percent' in status:
                metrics.append(f"Memory: {status['memory_percent']:.1f}%")
                
            if 'disk_io' in status and status['disk_io'] > 0:
                metrics.append(f"Disk I/O: {status['disk_io']:.1f} MB/s")
            
        except Exception as e:
            logging.debug(f"Error getting performance metrics: {e}")
    
    line_height = 25
    panel_height = line_height * len(metrics) + 20  # +20 for padding
    panel_width = 220
    
    panel_x = w - panel_width - 10
    panel_y = 10
    
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    alpha = 0.7
    mask = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    mask_region = img[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width]
    if mask.shape == mask_region.shape:  # Ensure shapes match
        img[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width] = cv2.addWeighted(
            overlay[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width], 
            alpha, 
            mask_region, 
            1-alpha, 
            0
        )
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (255, 255, 255)  # White text
    thickness = 1
    
    for i, metric in enumerate(metrics):
        y_pos = panel_y + 20 + i * line_height
        cv2.putText(img, metric, (panel_x + 10, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return img

def render_ui_async(frame, ui_data, state, on_complete=None):
    """
    Asynchronously render UI elements without blocking the main thread
    
    Args:
        frame: Camera frame
        ui_data: UI data needed for rendering
        state: Current application state
        on_complete: Optional callback when rendering is complete
    """
    global ui_rendering_state, background_scheduler, priority_scheduler, memory_manager, performance_monitor
    
    if frame is None:
        logging.warning("Received None frame in render_ui_async")
        return ui_rendering_state["last_output_img"] if ui_rendering_state["last_output_img"] is not None else None
    
    scheduler = priority_scheduler if priority_scheduler is not None else background_scheduler
    
    if scheduler is None:
        try:
            if USE_OPTIMIZED_RENDERING:
                output_img = draw_ui_optimized(
                    frame, ui_data["img_bg"], ui_data["class_id"], 
                    ui_data["img_waste_list"], ui_data["img_arrow"], 
                    ui_data["img_bins_list"], CLASS_MAPPING,
                    ui_data["scale_x"], ui_data["scale_y"], 
                    ui_data["screen_width"], ui_data["screen_height"],
                    memory_manager=memory_manager
                )
            else:
                output_img = draw_ui(
                    frame, ui_data["img_bg"], ui_data["class_id"], 
                    ui_data["img_waste_list"], ui_data["img_arrow"], 
                    ui_data["img_bins_list"], CLASS_MAPPING,
                    ui_data["scale_x"], ui_data["scale_y"], 
                    ui_data["screen_width"], ui_data["screen_height"]
                )
                
            ui_rendering_state["last_output_img"] = output_img
            ui_rendering_state["last_class_id"] = ui_data["class_id"]
            ui_rendering_state["last_update_time"] = time.time()
            
            ui_rendering_state["frame_count"] += 1
            elapsed = time.time() - ui_rendering_state["last_fps_time"]
            if elapsed >= 1.0:  # Update FPS every second
                ui_rendering_state["fps"] = ui_rendering_state["frame_count"] / elapsed
                ui_rendering_state["frame_count"] = 0
                ui_rendering_state["last_fps_time"] = time.time()
                
                if performance_monitor is not None:
                    performance_monitor.record_fps(ui_rendering_state["fps"])
            
            if on_complete:
                on_complete(output_img)
                
            return output_img
        except Exception as e:
            logging.error(f"Error in synchronous UI rendering: {e}")
            return ui_rendering_state["last_output_img"] if ui_rendering_state["last_output_img"] is not None else frame
    
    with ui_state_lock:
        if ui_rendering_state["rendering_in_progress"]:
            return ui_rendering_state["last_output_img"]
            
        ui_rendering_state["rendering_in_progress"] = True
    
    try:
        if memory_manager is not None:
            frame_copy = memory_manager.get_buffer(frame.shape, frame.dtype, "frame_copy")
            np.copyto(frame_copy, frame)
        else:
            frame_copy = frame.copy()
        
        render_data = {
            "frame": frame_copy,
            "ui_data": ui_data.copy(),
            "state": state.copy() if state else {}
        }
    except Exception as e:
        logging.error(f"Error copying frame data for async rendering: {e}")
        ui_rendering_state["rendering_in_progress"] = False
        return ui_rendering_state["last_output_img"] if ui_rendering_state["last_output_img"] is not None else frame
    
    def ui_render_task(data):
        """Function to be run in background thread to render UI"""
        try:
            start_time = time.time()
            
            if USE_OPTIMIZED_RENDERING:
                output_img = draw_ui_optimized(
                    data["frame"], data["ui_data"]["img_bg"], 
                    data["ui_data"]["class_id"], data["ui_data"]["img_waste_list"],
                    data["ui_data"]["img_arrow"], data["ui_data"]["img_bins_list"], 
                    CLASS_MAPPING, data["ui_data"]["scale_x"], 
                    data["ui_data"]["scale_y"], data["ui_data"]["screen_width"],
                    data["ui_data"]["screen_height"],
                    memory_manager=memory_manager
                )
            else:
                output_img = draw_ui(
                    data["frame"], data["ui_data"]["img_bg"], 
                    data["ui_data"]["class_id"], data["ui_data"]["img_waste_list"],
                    data["ui_data"]["img_arrow"], data["ui_data"]["img_bins_list"], 
                    CLASS_MAPPING, data["ui_data"]["scale_x"], 
                    data["ui_data"]["scale_y"], data["ui_data"]["screen_width"],
                    data["ui_data"]["screen_height"]
                )
                
            render_time = time.time() - start_time
            
            if render_time > 0.05:  # More than 50ms is considered slow
                logging.debug(f"Slow UI rendering: {render_time*1000:.1f}ms")
                
            return output_img
        except Exception as e:
            logging.error(f"Error in UI rendering task: {e}")
            return data["frame"] if "frame" in data else None
            
    def render_complete(output_img):
        """Callback when rendering is complete"""
        global ui_rendering_state, memory_manager
        
        if output_img is None:
            ui_rendering_state["rendering_in_progress"] = False
            return
        
        ui_rendering_state["last_output_img"] = output_img
        ui_rendering_state["last_class_id"] = ui_data["class_id"]
        ui_rendering_state["last_update_time"] = time.time()
        ui_rendering_state["rendering_in_progress"] = False
        
        ui_rendering_state["frame_count"] += 1
        elapsed = time.time() - ui_rendering_state["last_fps_time"]
        if elapsed >= 1.0:  # Update FPS every second
            ui_rendering_state["fps"] = ui_rendering_state["frame_count"] / elapsed
            ui_rendering_state["frame_count"] = 0
            ui_rendering_state["last_fps_time"] = time.time()
            
            if performance_monitor is not None:
                try:
                    performance_monitor.record_fps(ui_rendering_state["fps"])
                except Exception as e:
                    logging.debug(f"Error recording FPS: {e}")
        
        if on_complete:
            try:
                on_complete(output_img)
            except Exception as e:
                logging.error(f"Error in render completion callback: {e}")
            
        try:
            if not ui_update_queue.full():
                ui_update_queue.put_nowait(output_img)
        except Exception as e:
            logging.debug(f"Error adding rendered frame to update queue: {e}")
            
        try:
            if memory_manager is not None and "frame" in render_data:
                memory_manager.release_buffer(render_data["frame"], "frame_copy")
        except Exception as e:
            logging.debug(f"Error releasing frame buffer: {e}")
            
    try:
        if priority_scheduler is not None:
            priority_scheduler.schedule_task(ui_render_task, PriorityTaskScheduler.NORMAL, render_complete, render_data)
        else:
            background_scheduler.schedule_task(ui_render_task, render_complete, render_data)
    except Exception as e:
        logging.error(f"Failed to schedule UI rendering task: {e}")
        ui_rendering_state["rendering_in_progress"] = False
        if memory_manager is not None and "frame" in render_data:
            try:
                memory_manager.release_buffer(render_data["frame"], "frame_copy")
            except Exception:
                pass
    
    return ui_rendering_state["last_output_img"]

def handle_hardware_throttling(hardware_status, context, state):
    """Adjust settings based on hardware throttling detection"""
    global performance_monitor
    
    if not hardware_status:
        return
        
    try:
        if performance_monitor is not None:
            recommendations = performance_monitor.get_recommendations()
        else:
            recommendations = []
        if recommendations is None:
            recommendations = []
            
        is_throttling = performance_monitor.should_reduce_workload() if performance_monitor else False
        is_throttling = is_throttling or hardware_status.get("throttling_detected", False)
        
        if is_throttling:
            logging.info("Performance throttling detected, reducing workload")
            
            if "frame_skip" in state and state["frame_skip"] < 3:
                state["frame_skip"] += 1
                logging.info(f"Increased frame skip to {state['frame_skip']}")
                
            if "classifier_threshold" in state and state["classifier_threshold"] > 0.5:
                state["classifier_threshold"] = max(0.5, state["classifier_threshold"] - 0.1)
                logging.info(f"Reduced classifier threshold to {state['classifier_threshold']}")
                
            for rec in recommendations:
                if rec['component'] == 'model' and rec['action'] == 'optimize_model':
                    try:
                        if hasattr(context["classifier"], "enable_optimizations"):
                            context["classifier"].enable_optimizations()
                            logging.info("Enabled model optimizations")
                    except Exception as e:
                        logging.debug(f"Failed to enable model optimizations: {e}")
                        
                elif rec['component'] == 'memory' and rec['action'] == 'reduce_memory':
                    try:
                        gc.collect()
                        logging.info("Forced garbage collection to reduce memory pressure")
                    except Exception as e:
                        logging.debug(f"Failed to perform garbage collection: {e}")
    except Exception as e:
        logging.error(f"Error handling hardware throttling: {e}")

def process_frame(frame, context, ensemble=None, use_ensemble=False):
    ultrasonic_sensor = context.get('ultrasonic_sensor')
    if ultrasonic_sensor and not ultrasonic_sensor.is_object_detected():
        # Assign 'Nothing' class (class_id = 0) when no object detected
        return None, 0, 0.0
    # Optionally log distance
    if ultrasonic_sensor:
        distance = ultrasonic_sensor.get_distance()
        if distance is not None:
            logging.debug(f"Processing frame with object at {distance}cm")
    # ... rest of frame processing ...

def run_app_loop(context):
    """
    Main application loop with asynchronous UI rendering and hardware monitoring.
    """
    global background_scheduler, hardware_monitor, ui_update_queue, ui_rendering_state
    global priority_scheduler, memory_manager, performance_monitor
    
    try:
        set_process_priority_high()
        
        try:
            background_scheduler = BackgroundTaskScheduler(max_workers=2, name="UITasks")
            logging.info("Background task scheduler initialized")
        except Exception as e:
            logging.error(f"Failed to initialize background scheduler: {e}")
            background_scheduler = None
        
        if USE_PRIORITY_SCHEDULING:
            try:
                priority_scheduler = PriorityTaskScheduler(
                    high_workers=1,  # High priority (camera and classification)
                    normal_workers=2,  # Normal priority (UI rendering)
                    low_workers=1  # Low priority (cleanup, logging)
                )
                logging.info("Priority-based task scheduling enabled")
            except Exception as e:
                logging.error(f"Failed to initialize priority scheduler: {e}")
                priority_scheduler = None
        else:
            priority_scheduler = None
        
        if USE_MEMORY_LOCKING:
            try:
                memory_manager = MemoryManager()
                if memory_manager.can_lock_memory:
                    logging.info("Memory page locking enabled for critical buffers")
                else:
                    logging.info("Memory manager initialized (page locking not available)")
            except Exception as e:
                logging.error(f"Failed to initialize memory manager: {e}")
                memory_manager = None
        else:
            memory_manager = None
        
        performance_monitor = None
        if USE_PERFORMANCE_MONITORING:
            try:
                performance_monitor = PerformanceMonitor()
                performance_monitor.start()
                
                def on_performance_status(status):
                    check_for_memory_leaks()
                    
                    if status.get('cpu_percent', 0) > 90 or status.get('memory_percent', 0) > 90:
                        handle_hardware_throttling(status, context, None)
                
                performance_monitor.register_callback(on_performance_status)
                logging.info("Performance monitoring enabled")
            except Exception as e:
                logging.error(f"Failed to initialize performance monitor: {e}")
        
        hardware_monitor = None
        try:
            hardware_monitor = HardwareMonitor(check_interval=5.0)
            hardware_monitor.start()
            logging.info("Hardware monitoring started")
        except Exception as e:
            logging.error(f"Failed to initialize hardware monitor: {e}")
        
        ui_rendering_state = {
            "last_output_img": None,
            "last_class_id": 0,
            "last_update_time": 0,
            "rendering_in_progress": False,
            "frame_count": 0,
            "last_fps_time": time.time(),
            "fps": 0
        }
        
        try:
            ui_update_queue = queue.Queue(maxsize=3)
        except Exception as e:
            logging.error(f"Failed to create UI update queue: {e}")
            ui_update_queue = None

        f11_pressed = False
        
        def start_key_detection_thread():
            """Start a separate thread to detect F11 key presses using msvcrt on Windows"""
            if not sys.platform.startswith('win'):
                return None
            
            def detect_f11_key():
                nonlocal f11_pressed
                try:
                    import msvcrt
                    logging.info("F11 key detection thread started")
                    
                    while True:
                        try:
                            ch = msvcrt.getch()
                            if ch in (b'\x00', b'\xe0'):
                                code = msvcrt.getch()
                                if code == b'\x57':
                                    logging.info("F11 key detected via msvcrt")
                                    f11_pressed = True
                                    time.sleep(0.5)
                                    f11_pressed = False
                            elif ch == b'\x1b':
                                break
                        except Exception as e:
                            logging.error(f"Error reading key: {e}")
                            break
                        time.sleep(0.01)
                except Exception as e:
                    logging.error(f"Error in key detection thread: {e}")
        
            try:
                key_thread = threading.Thread(target=detect_f11_key, daemon=True)
                key_thread.start()
                return key_thread
            except Exception as e:
                logging.error(f"Failed to start key detection thread: {e}")
                return None
        
        key_thread = None
        if sys.platform.startswith('win'):
            try:
                key_thread = start_key_detection_thread()
                if key_thread:
                    logging.info("F11 key detection thread started successfully")
            except Exception as e:
                logging.error(f"Failed to start key detection: {e}")
        
        cap = context["cap"]
        window_name = context["window_name"]

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        is_fullscreen = True
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        logging.info("Starting in fullscreen mode")

        icon_file = resource_path('icon.ico')
        set_window_icon(window_name, icon_file)

        if sys.platform.startswith("win"):
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                set_window_dark_mode(hwnd, enable=True)

        use_ensemble = context.get("use_ensemble", False)
        show_model_stats = False
        show_performance = True
        
        context["status_message"] = "ESC: Window Mode | ENTER: Toggle Fullscreen | TAB: Exit"
        context["status_timestamp"] = time.time()
        
        frames_shown = 0

        state = {
            "current_camera_index": context["current_camera_index"],
            "current_device_name": context["current_device_name"],
            "classifier_threshold": context["classifier_threshold"],
            "DEBOUNCE_TIME": context["DEBOUNCE_TIME"],
            "cap": cap,
            "last_click_time": 0.0,
            "frame_count": 0,
            "last_prediction_time": time.time(),
            "consecutive_predictions": {},
            "fps": 0,
            "last_fps_time": time.time(),
            "last_bin_index": None,
            "last_detection_time": 0
        }

        cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: None)

        ensemble = None
        if use_ensemble:
            try:
                from ensemble_classifier import EnsembleClassifier
                ensemble = context["classifier"].ensemble
            except (ImportError, AttributeError):
                logging.warning("Failed to access ensemble classifier. Model stats will be disabled.")

        frame_count = 0
        start_time = time.time()
        fps = 0
        last_fps_update = time.time()
        
        last_key_press = 0
        KEY_DEBOUNCE_TIME = 0.5

        camera_stats = {}
        if hasattr(cap, "get_performance_stats"):
            try:
                camera_stats = cap.get_performance_stats()
                logging.info(f"Camera stats: {camera_stats}")
            except Exception as e:
                logging.warning(f"Error getting camera stats: {e}")

        servo_controller = context.get("servo_controller")
        
        # Initialize TTS for servo controller
        if servo_controller:
            # Check if TTS is available in context
            if context.get("tts_player"):
                try:
                    # Check if TTS is properly initialized
                    tts_player = context["tts_player"]
                    if hasattr(tts_player, "initialized") and tts_player.initialized:
                        logging.info("TTS player available and initialized. Audio feedback enabled.")
                    else:
                        logging.warning("TTS player available but not initialized. Audio feedback disabled.")
                except Exception as e:
                    logging.warning(f"Error initializing TTS for servo controller: {e}")
            else:
                logging.info("TTS player not available. Servo audio feedback will be disabled.")
            
            # Configure auto-close for servos if supported
            try:
                if hasattr(servo_controller, "set_auto_close_time"):
                    servo_controller.set_auto_close_time(10.0)  # 10 seconds auto-close time
                    logging.info("Servo auto-close feature enabled (10 seconds)")
                
                if hasattr(servo_controller, "set_smooth_movement"):
                    servo_controller.set_smooth_movement(
                        enabled=True, 
                        delay=0.015  # Controls sweep speed - lower = faster
                    )
                    logging.info("Servo sweep movement enabled")
            except Exception as e:
                logging.warning(f"Error configuring servo features: {e}")

        # Initialize all bins to 0 degrees at startup
        if servo_controller:
            for bin_idx in range(servo_controller.num_bins):
                servo_controller.close_lid(bin_idx)
            state["last_bin_index"] = None
            state["bin_cooldowns"] = {}  # Track cooldown timers for bins
            state["bin_open_times"] = {}  # Track when bins were opened

        # Set up object detection callback
        def on_object_detected(distance):
            logging.info(f"Object detected at {distance}cm - Ready for classification")
        
        if "servo_controller" in context and context["servo_controller"]:
            logging.info("Servo controller enabled but no object detection callback set.")

        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("Window closed by user. Exiting process.")
                sys.exit(0)

            ret, frame = state["cap"].read()
            if not ret:
                logging.warning("Failed to grab frame from camera. Attempting to reopen...")
                state["cap"].release()
                time.sleep(1)
                from camera import open_camera, find_camera_index
                try:
                    from async_camera import AsyncCamera
                    use_async = isinstance(state["cap"], AsyncCamera)
                    buffer_size = getattr(state["cap"], "buffer_size", 3) if use_async else 3
                except ImportError:
                    use_async = False
                    buffer_size = 3
                    
                # First try to reopen with the current camera index
                state["cap"] = open_camera(camera_index=state["current_camera_index"], 
                                          use_async=use_async, 
                                          buffer_size=buffer_size)
                
                # If reopening failed, try to find any available camera
                if not state["cap"]:
                    logging.warning(f"Unable to reopen camera at index {state['current_camera_index']}. Searching for available cameras...")
                    found_index = find_camera_index()
                    
                    if found_index >= 0 and found_index != state["current_camera_index"]:
                        logging.info(f"Found alternative camera at index {found_index}")
                        state["current_camera_index"] = found_index
                        state["current_device_name"] = f"Camera {found_index}"
                        
                        # Try to open the camera with the found index
                        state["cap"] = open_camera(camera_index=state["current_camera_index"], 
                                                  use_async=use_async, 
                                                  buffer_size=buffer_size)
                        
                        if state["cap"]:
                            context["status_message"] = f"Reconnected to camera {found_index}"
                            context["status_timestamp"] = time.time()
                    
                # If still no camera is available, keep retrying with the original index
                if not state["cap"]:
                    logging.error(f"Unable to find any working camera. Retrying with original index...")
                    time.sleep(1)
                
                continue

            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
                last_fps_update = current_time
                
                if hasattr(cap, "get_performance_stats"):
                    try:
                        camera_stats = cap.get_performance_stats()
                    except Exception:
                        pass
            
            frames_shown += 1

            # --- ULTRASONIC SENSOR DISTANCE CHECK ---
            ultrasonic_sensor = context.get('ultrasonic_sensor')
            object_in_range = True
            if ultrasonic_sensor:
                distance = ultrasonic_sensor.get_distance()
                if distance is None or distance > ultrasonic_sensor.threshold_distance:
                    # Out of range, set class to 'Nothing'
                    class_id = 0
                    confidence = 1.0
                    prediction = [1.0] + [0.0] * 7  # Assuming 8 classes
                    object_in_range = False
                else:
                    object_in_range = True

            if object_in_range:
                try:
                    start_time = time.time()
                    prediction, class_id = context["classifier"].get_prediction(frame,
                                                                threshold=state["classifier_threshold"],
                                                                draw=False)
                    inference_time = time.time() - start_time
                    confidence = prediction[class_id] if len(prediction) > class_id else 0
                    class_id = class_id if confidence > state["classifier_threshold"] and class_id != 0 else 0
                    if inference_time > 0.1:  # More than 100ms is slow
                        logging.debug(f"Slow inference: {inference_time*1000:.1f}ms")
                except Exception as e:
                    logging.exception("Error during classification.")
                    class_id = 0
                    inference_time = 0

            if hardware_monitor and hardware_monitor.should_reduce_workload():
                handle_hardware_throttling(hardware_monitor.get_status(), context, state)

            ui_data = {
                "img_bg": context["img_bg"],
                "class_id": class_id,
                "img_waste_list": context["img_waste_list"],
                "img_arrow": context["img_arrow"],
                "img_bins_list": context["img_bins_list"],
                "scale_x": context["scale_x"],
                "scale_y": context["scale_y"],
                "screen_width": context["screen_width"],
                "screen_height": context["screen_height"]
            }
            
            output_img = render_ui_async(frame, ui_data, state)
            
            if output_img is None:
                output_img = ui_rendering_state["last_output_img"] if ui_rendering_state["last_output_img"] is not None else frame
                
            try:
                if not ui_update_queue.empty():
                    new_output_img = ui_update_queue.get_nowait()
                    output_img = new_output_img
            except Exception:
                pass

            if context["status_message"] and (time.time() - context["status_timestamp"] < 3 or frames_shown < 180):
                cv2.putText(output_img, context["status_message"], (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if show_performance:
                output_img = draw_performance_metrics(output_img, fps, ensemble, use_ensemble, confidence)
                if camera_stats and "fps" in camera_stats:
                    cam_fps = camera_stats.get("fps", 0)
                    cam_name = camera_stats.get("camera_name", "Camera")
                    frames_captured = camera_stats.get("frames_captured", 0)

            if show_model_stats and ensemble:
                try:
                    stats = ensemble.get_model_stats()
                    y_pos = 210
                    for idx, model_stat in enumerate(stats):
                        if not model_stat.get("loaded", False):
                            continue
                        name = model_stat.get("name", f"Model {idx}")
                        avg_time = model_stat.get("avg_inference_time", 0) * 1000
                        weight = ensemble.model_weights[idx] if idx < len(ensemble.model_weights) else 0
                        info_text = f"{name} - {avg_time:.1f}ms (weight: {weight:.2f})"
                        cv2.putText(output_img, info_text, (20, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                        y_pos += 25
                except Exception as e:
                    logging.error(f"Error showing model stats: {e}")

            if show_performance and context.get("use_ensemble", False):
                try:
                    from ensemble_classifier import EnsembleClassifier
                    ensemble = context['classifier'].ensemble
                    if ensemble and hasattr(ensemble, 'get_ensemble_stats'):
                        stats = ensemble.get_ensemble_stats()
                        model_stats = ensemble.get_model_stats() if hasattr(ensemble, 'get_model_stats') else []
                        logging.info(f"ENSEMBLE DEBUG: Stats: {stats}")
                        if model_stats:
                            logging.info(f"ENSEMBLE DEBUG: Using {len(model_stats)} models")
                            for i, model in enumerate(model_stats):
                                logging.info(f"ENSEMBLE DEBUG: Model {i+1}: {model['name']}, format: {model['format']}")
                except Exception as e:
                    logging.warning(f"Failed to access ensemble classifier. Model stats will be disabled.")

            # --- SERVO BIN CONTROL LOGIC ---
            if context["servo_enabled"] and class_id != 0:
                bin_index = CLASS_MAPPING.get(class_id, 0)
                if bin_index is not None:
                    last_bin = state.get("last_bin_index")
                    bin_cooldowns = state.get("bin_cooldowns", {})
                    bin_open_times = state.get("bin_open_times", {})
                    now = time.time()

                    # Handle cooldowns for bins
                    bins_to_remove = []
                    for b, cooldown_end in bin_cooldowns.items():
                        if now >= cooldown_end:
                            context["servo_controller"].close_lid(b)
                            bins_to_remove.append(b)
                    for b in bins_to_remove:
                        del bin_cooldowns[b]

                    # If the same bin as last frame, keep it open
                    if bin_index == last_bin:
                        # Keep the bin open (do nothing)
                        pass
                    else:
                        # If a different bin is detected
                        if last_bin is not None:
                            # Start cooldown for previous bin
                            bin_cooldowns[last_bin] = now + 10  # 10 second cooldown
                            # (Bin will be closed after cooldown above)
                        # Open the new bin if not already open
                        context["servo_controller"].open_lid(bin_index)
                        bin_open_times[bin_index] = now
                        state["last_bin_index"] = bin_index
                        
                        # Add TTS feedback for new classification
                        if context.get("tts_player") and context["tts_player"].initialized:
                            from config import CLASS_NAMES
                            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                            context["tts_player"].speak(f"Detected {class_name}")
                            
                    # Save state
                    state["bin_cooldowns"] = bin_cooldowns
                    state["bin_open_times"] = bin_open_times

            # --- STREAM SERVER FIX: send the UI frame to the stream server ---
            update_frame(output_img)
            # --------------------------------------------------------------

            cv2.imshow(window_name, output_img)
            
            key = cv2.waitKeyEx(5)

            if key != -1:
                logging.info(f"Key pressed: {key} (hex: {hex(key)})")
                
            current_time = time.time()
            
            if f11_pressed and not is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_fullscreen = True
                context["status_message"] = "Fullscreen mode enabled"
                context["status_timestamp"] = time.time()
                
            if key != -1 and current_time - last_key_press > KEY_DEBOUNCE_TIME:
                last_key_press = current_time
                
                if key == 27:
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        is_fullscreen = False
                        context["status_message"] = "Window mode enabled"
                        context["status_timestamp"] = time.time()

                elif (key == ord('F') or key == ord('f') or 
                      key == KEY_F11 or key == 0x7A or key == 0x7B or
                      key == 122 or key == 123 or key == 0x10000001C or 
                      key == 0x10000003A or key == 65480 or key == 65470+10 or
                      key == 270) and not is_fullscreen:
                    logging.info(f"F11 key detected with code: {key} (hex: {hex(key)})")
                    
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
                    context["status_message"] = "Fullscreen mode enabled"
                    context["status_timestamp"] = time.time()
                    
                elif not is_fullscreen and (key >= 65470 and key <= 65482 and key - 65470 == 10):
                    logging.info(f"F11 key detected via fallback method: {key}")
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
                    context["status_message"] = "Fullscreen mode enabled"
                    context["status_timestamp"] = time.time()

                elif not is_fullscreen and (key >= 65470 and key <= 65482 and key - 65470 == 10):
                    logging.info(f"F11 key detected via fallback method: {key}")
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
                    context["status_message"] = "Fullscreen mode enabled"
                    context["status_timestamp"] = time.time()

                elif key == 13:
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        is_fullscreen = False
                        context["status_message"] = "Window mode enabled (ENTER)"
                        context["status_timestamp"] = time.time()
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        is_fullscreen = True
                        context["status_message"] = "Fullscreen mode enabled (ENTER)"
                        context["status_timestamp"] = time.time()
                    logging.info(f"ENTER fullscreen toggle: {is_fullscreen}")
                    
                elif key == ord('F') or key == ord('f'):
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        is_fullscreen = False
                        context["status_message"] = "Window mode enabled (F key)"
                        context["status_timestamp"] = time.time()
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        is_fullscreen = True
                        context["status_message"] = "Fullscreen mode enabled (F key)"
                        context["status_timestamp"] = time.time()
                    logging.info(f"F key fullscreen toggle: {is_fullscreen}")

                elif key == ord('Q') or key == ord('q'):
                    logging.info("Q pressed — exiting loop.")
                    break

                elif key == 9:
                    logging.info("Tab pressed — exiting process.")
                    sys.exit(0)
                    
                elif key == ord('M') or key == ord('m'):
                    if use_ensemble and ensemble:
                        show_model_stats = not show_model_stats
                        context["status_message"] = f"Model stats: {'ON' if show_model_stats else 'OFF'}"
                        context["status_timestamp"] = time.time()
                    
                elif key == ord('P') or key == ord('p'):
                    show_performance = not show_performance
                    context["status_message"] = f"Performance metrics: {'ON' if show_performance else 'OFF'}"
                    context["status_timestamp"] = time.time()

    except Exception as e:
        logging.error(f"Error in main application loop: {e}")
        traceback.print_exc()
    finally:
        cleanup_resources(context)


def cleanup_resources(context):
    """Clean up all resources properly before exit, including closing all servo bins."""
    logging.info("Cleaning up resources...")
    
    # Close all servo bins if controller is present
    servo_controller = context.get("servo_controller")
    servo_enabled = context.get("servo_enabled", False)
    if servo_controller and servo_enabled:
        logging.info("Closing all servo bins before shutdown...")
        try:
            for bin_idx in range(servo_controller.num_bins):
                logging.info(f"Closing bin {bin_idx} via servo controller...")
                servo_controller.close_lid(bin_idx)
                time.sleep(0.2)  # Small delay to ensure command is sent
            servo_controller.cleanup()
            logging.info("All servo bins closed and servo controller cleaned up.")
        except Exception as e:
            logging.error(f"Error closing servo bins on shutdown: {e}")

    cv2.destroyAllWindows()
    
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            cv2.cuda.resetDevice()
            logging.info("CUDA device reset")
        except Exception as e:
            logging.error(f"Error resetting CUDA device: {e}")
    
    if background_scheduler:
        try:
            background_scheduler.shutdown(wait=True)
            logging.info("Background scheduler shutdown")
        except Exception as e:
            logging.error(f"Error shutting down background scheduler: {e}")
    
    if priority_scheduler:
        try:
            priority_scheduler.shutdown(wait=True)
            logging.info("Priority scheduler shutdown")
        except Exception as e:
            logging.error(f"Error shutting down priority scheduler: {e}")
    
    if memory_manager:
        try:
            memory_manager.cleanup()
            logging.info("Memory manager resources released")
        except Exception as e:
            logging.error(f"Error cleaning up memory manager: {e}")
    
    if performance_monitor:
        try:
            performance_monitor.stop()
            logging.info("Performance monitor stopped")
        except Exception as e:
            logging.error(f"Error stopping performance monitor: {e}")
    
    gc.collect()
    logging.info("Resource cleanup complete")
