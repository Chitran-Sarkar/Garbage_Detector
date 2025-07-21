#!/usr/bin/env python
# async_camera.py - Windows-optimized AsyncCamera implementation using DirectShow

import cv2
import threading
import time
import logging
import queue
import numpy as np
import gc
import sys
from pygrabber.dshow_graph import FilterGraph
import win32api
import os
import platform
import win32process

try:
    from shared_memory import SharedFrameBuffer
    HAS_SHARED_MEMORY = True
except ImportError:
    HAS_SHARED_MEMORY = False

class AsyncCamera:
    """Asynchronous camera implementation optimized for Windows with background processing"""
    
    def __init__(self, camera_index=0, buffer_size=3, width=1280, height=720, use_hw_accel=True, 
                backend=cv2.CAP_DSHOW, frame_skip=0, high_performance=False):
        """Initialize the async camera with requested features"""
        self.camera_index = camera_index
        self.buffer_size = max(2, buffer_size)  # Minimum buffer size of 2
        self.width = width
        self.height = height
        self.use_hw_accel = use_hw_accel
        self.backend = backend
        self.frame_skip = frame_skip
        self.high_performance = high_performance
        
        # Try to get camera name
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if 0 <= camera_index < len(devices):
                self.camera_name = devices[camera_index]
            else:
                self.camera_name = f"Camera {camera_index}"
        except Exception as e:
            logging.debug(f"Couldn't get camera name: {e}")
            self.camera_name = f"Camera {camera_index}"
            
        logging.info(f"Initializing async camera: {self.camera_name}")
        
        # Internal state
        self.running = False
        self.cap = None
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.last_frame = None
        self.last_frame_lock = threading.RLock()
        self.thread = None
        self.lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._skip_event = threading.Event()
        
        # Performance metrics
        self.frames_captured = 0
        self.frames_skipped = 0
        self.capture_start_time = 0
        self.current_fps = 0
        self.last_update_time = 0
        self.hw_acceleration_status = 0
        self.dynamic_skip = 0
        self.last_load_check = 0
        
        # Initialize frame buffers for memory efficiency
        self._init_preallocated_buffers()
        
        # Set up shared memory if available
        self.shared_frame_buffer = None
        self.use_shared_memory = False
        if HAS_SHARED_MEMORY:
            try:
                self.shared_frame_buffer = SharedFrameBuffer(
                    buffer_size=self.buffer_size,
                    width=self.width,
                    height=self.height,
                    channels=3,
                    dtype=np.uint8
                )
                self.use_shared_memory = True
                logging.info(f"Initialized shared memory frame buffer with {self.buffer_size} slots")
            except Exception as e:
                logging.warning(f"Failed to initialize shared memory: {e}")
                self.use_shared_memory = False
        
        # Start the camera
        if not self._start_camera():
            raise RuntimeError(f"Failed to start camera {camera_index}")
        
        # Start maintenance thread for buffer management
        self._start_maintenance_thread()
        
    def _init_preallocated_buffers(self):
        """Initialize preallocated buffers for better frame handling performance"""
        try:
            self.preallocated_buffers = []
            self.pinned_memory_available = False
            
            # Try using CUDA pinned memory if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    import pycuda.driver as cuda
                    
                    if not cuda.Context.get_current():
                        cuda.init()
                        cuda_ctx = cuda.Device(0).make_context()
                        cuda_ctx.pop()
                    
                    # Create pinned memory for faster transfers
                    for _ in range(self.buffer_size * 2):
                        pinned_mem = cuda.pagelocked_empty((self.height, self.width, 3), dtype=np.uint8)
                        self.preallocated_buffers.append(pinned_mem)
                    
                    self.pinned_memory_available = True
                    logging.info(f"Preallocated {len(self.preallocated_buffers)} CUDA pinned memory buffers")
                except (ImportError, Exception) as e:
                    logging.debug(f"Could not allocate CUDA pinned memory: {e}")
                    self.pinned_memory_available = False
            
            # Use regular numpy arrays if pinned memory not available
            if not self.pinned_memory_available:
                buffer_multiplier = 2
                if self.high_performance:
                    buffer_multiplier = 3
                
                # Adjust buffer size based on system memory
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    available_gb = mem.available / (1024 * 1024 * 1024)
                    
                    if available_gb > 8:
                        buffer_multiplier = 4
                    if available_gb > 16:
                        buffer_multiplier = 6
                except ImportError:
                    pass
                
                # Create standard buffers
                buffer_count = self.buffer_size * buffer_multiplier
                for _ in range(buffer_count):
                    buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.preallocated_buffers.append(buffer)
                
                logging.info(f"Preallocated {buffer_count} standard memory buffers")
            
            self.buffer_index = 0
            self.buffer_lock = threading.RLock()
        except Exception as e:
            logging.error(f"Failed to preallocate buffers: {e}")
            self.preallocated_buffers = None
            self.pinned_memory_available = False
        
    def _start_maintenance_thread(self):
        """Start a thread to periodically clean up resources"""
        def maintenance_routine():
            try:
                import psutil
                psutil_available = True
            except ImportError:
                psutil_available = False
                logging.warning("psutil not available, limited maintenance capabilities")
            
            while self.running and not self._shutdown_event.is_set():
                try:
                    # Check current memory usage
                    if psutil_available:
                        mem_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                        
                        # Clean up if memory usage is high
                        if mem_usage > 1000:  # 1000 MB threshold
                            self._cleanup_unused_frames()
                            
                        # Explicit GC after cleanup if memory is tight
                        if mem_usage > 1500:
                            gc.collect()
                            
                        # Adjust frame skip based on system load
                        self._adjust_dynamic_frame_skip()
                    else:
                        # Periodic cleanup even without psutil
                        self._cleanup_unused_frames()
                        
                        # Call GC periodically
                        if time.time() - self.last_load_check > 60:  # Once per minute
                            gc.collect()
                            self.last_load_check = time.time()
                
                except Exception as e:
                    logging.debug(f"Maintenance thread error: {e}")
                finally:
                    # Check every few seconds
                    time.sleep(5)
        
        self._maintenance_thread = threading.Thread(
            target=maintenance_routine,
            daemon=True,
            name="CameraMaintenanceThread"
        )
        self._maintenance_thread.start()
        logging.debug("Started camera maintenance thread")
        
    def _cleanup_unused_frames(self):
        """Clean up unused frames from the buffer"""
        # Only attempt cleanup if buffer is more than half full
        if self.frame_buffer.qsize() < self.frame_buffer.maxsize // 2:
            return
        
        # Calculate how many frames to remove
        to_remove = max(1, self.frame_buffer.qsize() - self.frame_buffer.maxsize // 2)
        
        # Remove oldest frames
        for _ in range(to_remove):
            if not self.frame_buffer.empty():
                try:
                    # Non-blocking get to avoid deadlocks
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.task_done()
                except queue.Empty:
                    break
        
    def _adjust_dynamic_frame_skip(self):
        """Dynamically adjust frame skip based on system load"""
        try:
            import psutil
            
            if time.time() - self.last_load_check < 5:  # Only check every 5 seconds
                return
                
            self.last_load_check = time.time()
            
            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Check memory pressure
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Combined system pressure metric (0-100)
            system_pressure = max(cpu_percent, memory_percent)
            
            # Adjust dynamic frame skip based on system pressure
            old_skip = self.dynamic_skip
            
            if system_pressure > 90:  # Heavy load
                self.dynamic_skip = min(4, self.frame_skip + 3)
            elif system_pressure > 80:  # High load
                self.dynamic_skip = min(3, self.frame_skip + 2)
            elif system_pressure > 70:  # Moderate load
                self.dynamic_skip = min(2, self.frame_skip + 1)
            elif system_pressure > 50:  # Normal load
                self.dynamic_skip = max(0, self.frame_skip)
            else:  # Light load
                self.dynamic_skip = max(0, self.frame_skip - 1)
                
            if old_skip != self.dynamic_skip:
                logging.debug(f"Adjusted dynamic frame skip: {old_skip} -> {self.dynamic_skip} (System pressure: {system_pressure:.1f}%)")
        except Exception as e:
            logging.debug(f"Error adjusting frame skip: {e}")
            
    def _start_camera(self):
        """Initialize and start the camera capture thread"""
        with self.lock:
            try:
                # Configure camera
                self.cap = cv2.VideoCapture(self.camera_index, self.backend)
                
                if not self.cap.isOpened():
                    logging.error(f"Failed to open camera at index {self.camera_index}")
                    return False
                
                # Set requested resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Try to enable hardware acceleration
                if self.use_hw_accel and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    self.hw_acceleration_status = self.cap.set(
                        cv2.CAP_PROP_HW_ACCELERATION, 
                        cv2.VIDEO_ACCELERATION_ANY
                    )
                    if self.hw_acceleration_status:
                        logging.info("Hardware acceleration enabled for camera")
                        
                        # Try to initialize additional hardware acceleration features
                        success = self._init_hardware_acceleration()
                        if success:
                            logging.info("Additional hardware acceleration features initialized")
                    else:
                        logging.warning("Failed to enable hardware acceleration for camera")
                
                # Set buffer size (related to internal camera driver buffer)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                
                # Get actual camera properties
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                # Adjust our buffer dimensions if the camera returned different dimensions
                if actual_width != self.width or actual_height != self.height:
                    logging.warning(f"Camera returned different dimensions: requested {self.width}x{self.height}, got {actual_width}x{actual_height}")
                    self.width = int(actual_width)
                    self.height = int(actual_height)
                    
                    # Reinitialize buffers with new dimensions
                    self._init_preallocated_buffers()
                
                logging.info(f"Camera started: {actual_width}x{actual_height} @ {actual_fps} FPS, "
                             f"Hardware acceleration: {bool(self.hw_acceleration_status)}")
                
                # Start capture thread
                self.running = True
                self.capture_start_time = time.time()
                self.thread = threading.Thread(target=self._capture_thread, daemon=True,
                                              name="CameraCaptureThread")
                self.thread.start()
                
                # Set thread priority if on Windows
                if platform.system() == 'Windows':
                    try:
                        thread_handle = win32api.OpenThread(
                            0x0400, False, win32api.GetThreadId(self.thread.native_id)
                        )
                        win32process.SetThreadPriority(
                            thread_handle, win32process.THREAD_PRIORITY_ABOVE_NORMAL
                        )
                        win32api.CloseHandle(thread_handle)
                        logging.info("Set camera thread priority to ABOVE_NORMAL")
                    except Exception as e:
                        logging.debug(f"Failed to set thread priority: {e}")
                
                # Wait briefly to ensure a frame is captured
                for _ in range(10):  # Wait up to 1 second
                    if not self.frame_buffer.empty() or self.last_frame is not None:
                        break
                    time.sleep(0.1)
                
                return True
            
            except Exception as e:
                logging.error(f"Error starting camera: {e}")
                self._cleanup_resources()
                return False
            
    def _init_hardware_acceleration(self):
        """Initialize additional hardware acceleration features"""
        try:
            # Check if CUDA is available for OpenCV
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Initialize CUDA stream
                cuda_stream = cv2.cuda_Stream()
                
                # Warm up CUDA functions
                if self.width > 0 and self.height > 0:
                    try:
                        # Create a test frame and upload to GPU
                        test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(test_frame)
                        
                        # Try GPU-accelerated operations
                        result = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                        result.download()
                        logging.info("CUDA processing pipeline verified")
                        return True
                    except Exception as e:
                        logging.warning(f"CUDA test operations failed: {e}")
            
            # Check for OpenCL (alternative acceleration)
            if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    logging.info("OpenCL acceleration enabled")
                    return True
            
            return False
        except Exception as e:
            logging.debug(f"Error initializing hardware acceleration: {e}")
            return False
            
    def _capture_thread(self):
        """Background thread that captures frames from the camera"""
        skip_counter = 0
        frame_count = 0
        last_fps_update = time.time()
        consecutive_errors = 0
        
        logging.info("Camera capture thread started")
        
        while self.running and not self._shutdown_event.is_set():
            try:
                if not self.cap or not self.cap.isOpened():
                    logging.error("Camera connection lost, attempting reset...")
                    if not self._reset_camera_connection():
                        # Failed to reset, sleep to avoid tight loop
                        time.sleep(1.0)
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            logging.error("Too many consecutive camera errors, stopping capture thread")
                            break
                        continue
                    else:
                        consecutive_errors = 0
                
                # Check if we should skip this frame (for performance)
                skip_this_frame = False
                if self.dynamic_skip > 0:
                    skip_counter = (skip_counter + 1) % (self.dynamic_skip + 1)
                    skip_this_frame = skip_counter != 0
                
                if skip_this_frame:
                    # Just read and discard the frame
                    _, _ = self.cap.read()
                    self.frames_skipped += 1
                    continue
                
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    logging.warning("Received invalid frame from camera")
                    consecutive_errors += 1
                    
                    if consecutive_errors > 5:
                        logging.error("Too many invalid frames, attempting to reset camera...")
                        self._reset_camera_connection()
                        time.sleep(0.5)  # Brief pause before continuing
                    else:
                        time.sleep(0.1)  # Small delay to avoid tight loop
                    continue
                else:
                    consecutive_errors = 0  # Reset error counter on success
                
                # --- Handle captured frame ---
                frames_since_start += 1
                
                # Skip frames if requested
                if self.frame_skip > 0 and frames_since_start % (self.frame_skip + 1) != 0:
                    self.frames_skipped += 1
                    continue
                
                # Process the frame
                if frame is not None:
                    # Ensure frame has the right dimensions
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Record timestamp
                    current_time = time.time()
                    
                    # Use shared memory if available
                    if self.use_shared_memory and self.shared_frame_buffer:
                        try:
                            # Write frame to shared memory
                            self.shared_frame_buffer.write_frame(
                                frame, 
                                timestamp=current_time,
                                frame_number=self.frames_captured
                            )
                            
                            # Store the frame in our regular buffer as well for backward compatibility
                            with self.last_frame_lock:
                                self.last_frame = frame.copy()
                                
                            # Try to put in frame queue but don't block if full
                            try:
                                self.frame_buffer.put_nowait((frame.copy(), current_time))
                            except queue.Full:
                                # Queue full, drop oldest frame
                                try:
                                    self.frame_buffer.get_nowait()
                                    self.frame_buffer.put_nowait((frame.copy(), current_time))
                                except (queue.Empty, queue.Full):
                                    pass
                        except Exception as e:
                            logging.error(f"Shared memory write error: {e}")
                            # Fall back to the standard buffer mechanism
                            self._process_frame_with_buffer(frame, current_time)
                    else:
                        # Use standard buffer mechanism
                        self._process_frame_with_buffer(frame, current_time)
                    
                    # Update stats
                    self.frames_captured += 1
                    
                    # Calculate FPS periodically
                    frames_in_window += 1
                    if current_time - fps_start_time >= 1.0:  # Update FPS every second
                        self.current_fps = frames_in_window / (current_time - fps_start_time)
                        frames_in_window = 0
                        fps_start_time = current_time
                
            except Exception as e:
                logging.error(f"Error in capture thread: {e}")
                time.sleep(0.1)
                consecutive_errors += 1
                if consecutive_errors > 10:
                    logging.error("Too many consecutive errors in capture thread, attempting reset")
                    self._reset_camera_connection()
                    consecutive_errors = 0
                
        # Clean up
        self._cleanup_resources()
        logging.info("Camera capture thread stopped")
    
    def _process_frame(self, frame):
        """Apply basic processing to frame"""
        try:
            # Apply hardware-accelerated processing if enabled
            if self.use_hw_accel:
                try:
                    frame = self._process_frame_hw_accel(frame)
                except Exception as e:
                    logging.debug(f"Hardware-accelerated processing failed: {e}")
                    # Fall back to regular processing
            
            # Get next preallocated buffer
            if self.preallocated_buffers:
                with self.buffer_lock:
                    buffer_idx = self.buffer_index
                    self.buffer_index = (self.buffer_index + 1) % len(self.preallocated_buffers)
                
                # Copy frame data to preallocated buffer
                try:
                    np.copyto(self.preallocated_buffers[buffer_idx], frame)
                    result_frame = self.preallocated_buffers[buffer_idx]
                except ValueError:  # If shapes don't match
                    result_frame = frame.copy()
            else:
                result_frame = frame.copy()
            
            return result_frame
        except Exception as e:
            logging.debug(f"Basic processing error: {e}")
            return frame  # Return original frame on error
    
    def _process_frame_hw_accel(self, frame):
        """Apply hardware-accelerated processing to frame using CUDA if available"""
        try:
            # Use CUDA if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Create GPU Matrix
                gpu_frame = cv2.cuda_GpuMat()
                
                # Upload the frame to GPU
                gpu_frame.upload(frame)
                
                # Apply processing (here using a simple color correction as demonstration)
                # In a real app, you could apply more complex transformations
                
                # Example: Simple color enhancement with hardware acceleration
                if gpu_frame.channels() == 3:
                    # Apply slight contrast enhancement using GPU
                    alpha = 1.1  # Contrast factor
                    beta = 5     # Brightness boost
                    
                    # Convert to float32 for arithmetic operations
                    gpu_float = gpu_frame.convertTo(cv2.CV_32F)
                    
                    # Apply contrast/brightness adjustment
                    gpu_result = cv2.cuda.addWeighted(gpu_float, alpha, gpu_float, 0, beta)
                    
                    # Convert back to uint8
                    gpu_output = gpu_result.convertTo(cv2.CV_8U)
                    
                    # Download from GPU
                    result = gpu_output.download()
                    return result
                else:
                    # Just return the unprocessed frame
                    return frame
            elif hasattr(cv2, 'UMat') and hasattr(cv2, 'ocl') and cv2.ocl.useOpenCL():
                # Use OpenCL acceleration if available
                umat_frame = cv2.UMat(frame)
                
                # Apply processing using OpenCL
                alpha = 1.1  # Contrast factor
                beta = 5     # Brightness boost
                result = cv2.addWeighted(umat_frame, alpha, umat_frame, 0, beta)
                
                # Get result back from device
                return result.get()
            else:
                # No hardware acceleration available, return original frame
                return frame
        except Exception as e:
            logging.debug(f"Error in hardware-accelerated processing: {e}")
            return frame  # Return original frame on error

    def _reset_camera_connection(self):
        """Reset the connection to the camera if it gets disrupted"""
        try:
            logging.info("Resetting camera connection...")
            with self.lock:
                # Release old connection
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception as e:
                        logging.debug(f"Error releasing camera: {e}")
                    self.cap = None
                
                # Wait a moment for camera to reset
                time.sleep(1.0)
                
                # Try to reopen the camera with the same parameters
                self.cap = cv2.VideoCapture(self.camera_index, self.backend)
                
                if not self.cap.isOpened():
                    logging.error("Failed to reopen camera after reset")
                    return False
                
                # Restore configuration
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                
                # Restore hardware acceleration if it was enabled
                if self.use_hw_accel and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    
                # Log successful reset
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logging.info(f"Camera reset successful. New resolution: {actual_width}x{actual_height}")
                
                return True
        except Exception as e:
            logging.error(f"Error resetting camera: {e}")
            return False
    
    def _cleanup_resources(self):
        """Clean up resources when shutting down"""
        try:
            # Release capture object
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
            # Clear frame buffer
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.task_done()
                except queue.Empty:
                    break
            
            # Clear reference to last frame to free memory
            with self.last_frame_lock:
                self.last_frame = None
            
            # Clear preallocated buffers
            if hasattr(self, 'preallocated_buffers') and self.preallocated_buffers:
                with self.buffer_lock:
                    self.preallocated_buffers = None
            
            # Force garbage collection
            gc.collect()
            
            logging.info("Camera resources cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up camera resources: {e}")
            
    def read(self):
        """
        Read the next frame from the camera with buffering.
        
        Returns:
            Tuple of (ret, frame) where ret is True if frame is valid
        """
        # Try shared memory first if available
        if self.use_shared_memory and self.shared_frame_buffer:
            frame, metadata = self.get_shared_frame()
            if frame is not None:
                return True, frame
        
        # Fall back to queue-based approach
        try:
            frame, timestamp = self.frame_buffer.get(timeout=0.5)
            return True, frame
        except queue.Empty:
            # If queue is empty, try to get the last frame directly
            with self.last_frame_lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            
            # No frames available
            return False, None
        
    def release(self):
        """Release camera resources, similar to cv2.VideoCapture.release()"""
        with self.lock:
            if not self.running:
                return
            
            # Signal thread to shut down
            self.running = False
            self._shutdown_event.set()
            
            # Wait for thread to terminate (with timeout)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=3.0)
            
            # Clean up resources
            self._cleanup_resources()
            logging.info("Camera released")
            
    def is_opened(self):
        """Check if camera is opened, similar to cv2.VideoCapture.isOpened()"""
        return self.running and (self.cap is not None and self.cap.isOpened())
        
    def get(self, prop_id):
        """Get camera property, similar to cv2.VideoCapture.get()"""
        if not self.running or self.cap is None:
            return 0
        
        if prop_id == cv2.CAP_PROP_FPS:
            # Return our calculated FPS instead of camera-reported FPS
            if self.current_fps > 0:
                return self.current_fps
        
        # Forward other properties to the underlying VideoCapture
        return self.cap.get(prop_id)
        
    def set(self, prop_id, value):
        """Set camera property, similar to cv2.VideoCapture.set()"""
        if not self.running or self.cap is None:
            return False
        
        with self.lock:
            # Handle special cases
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = int(value)
                # Reinitialize buffers if needed
                if self.preallocated_buffers:
                    self._init_preallocated_buffers()
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = int(value)
                # Reinitialize buffers if needed
                if self.preallocated_buffers:
                    self._init_preallocated_buffers()
                
            # Forward to the underlying VideoCapture
            return self.cap.set(prop_id, value)
        
    def get_performance_stats(self):
        """Get camera performance statistics"""
        stats = {
            "fps": self.current_fps,
            "frames_captured": self.frames_captured,
            "frames_skipped": self.frames_skipped,
            "buffer_usage": self.frame_buffer.qsize() / self.frame_buffer.maxsize if self.frame_buffer else 0,
            "dynamic_skip": self.dynamic_skip,
            "hw_acceleration": bool(self.hw_acceleration_status),
            "runtime": time.time() - self.capture_start_time if self.capture_start_time else 0
        }
        return stats

    def get_shared_frame(self):
        """
        Get the latest frame from shared memory if available
        
        Returns:
            Tuple of (frame, metadata) or (None, None) if not available
        """
        if not self.use_shared_memory or self.shared_frame_buffer is None:
            return None, None
        
        try:
            return self.shared_frame_buffer.peek_latest_frame()
        except Exception as e:
            logging.error(f"Error getting frame from shared memory: {e}")
            return None, None

    def _process_frame_with_buffer(self, frame, timestamp):
        """Process a frame using the standard buffer mechanism"""
        # Use preallocated buffers if available for better memory efficiency
        if self.preallocated_buffers is not None:
            with self.buffer_lock:
                if self.buffer_index >= len(self.preallocated_buffers):
                    self.buffer_index = 0
                
                # Get next buffer
                buffer = self.preallocated_buffers[self.buffer_index]
                
                # Copy frame data into preallocated buffer
                np.copyto(buffer, frame)
                self.buffer_index += 1
                
                # Keep a reference to the most recent frame
                with self.last_frame_lock:
                    self.last_frame = buffer
                
                # Try to put in frame queue but don't block if full
                try:
                    self.frame_buffer.put_nowait((buffer, timestamp))
                except queue.Full:
                    # Queue full, drop oldest frame
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((buffer, timestamp))
                    except (queue.Empty, queue.Full):
                        pass
        else:
            # Use standard NumPy copy
            with self.last_frame_lock:
                self.last_frame = frame.copy()
            
            # Try to put in frame queue but don't block if full
            try:
                self.frame_buffer.put_nowait((frame.copy(), timestamp))
            except queue.Full:
                # Queue full, drop oldest frame
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait((frame.copy(), timestamp))
                except (queue.Empty, queue.Full):
                    pass 