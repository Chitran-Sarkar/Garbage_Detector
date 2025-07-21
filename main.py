#main.py
#!/usr/bin/env python
import argparse
import logging
import multiprocessing
import os
import platform
import sys
import threading
import time

import cv2

from app_loop import run_app_loop
from initializer import initialize_app
from stream_server import start_stream_server, update_frame, stop_stream_server
from tts_player import tts_player
from utils import BackgroundTaskScheduler, MemoryManager

# Set up logging
try:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'waste_app.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
except Exception as e:
    # Simple fallback if logging setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if hasattr(sys, 'frozen') and getattr(sys, 'frozen'):
        try:
            app_dir = os.path.dirname(sys.executable)
            fallback_log = os.path.join(app_dir, 'waste_app_error.log')
            file_handler = logging.FileHandler(fallback_log)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        except Exception:
            pass

# Global background task scheduler
background_scheduler = None

def check_windows_11():
    """Check if running on Windows 11"""
    if not sys.platform.startswith('win'):
        return False
    
    try:
        version = platform.version()
        build = int(version.split('.')[2])
        return build >= 22000
    except Exception:
        return False

def determine_optimal_buffer_size():
    """Calculate optimal buffer size based on system resources"""
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        current_load = psutil.cpu_percent(interval=0.1) / 100.0
        
        has_cuda = False
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                has_cuda = True
        except Exception:
            pass
            
        available_memory_gb = mem.available / (1024 * 1024 * 1024)
        
        disk_io_high = False
        try:
            disk_io = psutil.disk_io_counters(perdisk=False)
            if disk_io and hasattr(disk_io, 'read_bytes') and hasattr(disk_io, 'write_bytes'):
                time.sleep(0.05)
                disk_io2 = psutil.disk_io_counters(perdisk=False)
                read_rate = (disk_io2.read_bytes - disk_io.read_bytes) / 0.05 / 1024 / 1024
                write_rate = (disk_io2.write_bytes - disk_io.write_bytes) / 0.05 / 1024 / 1024
                
                if read_rate > 20 or write_rate > 20:
                    disk_io_high = True
        except Exception:
            pass
            
        buffer_size = 4
        
        if available_memory_gb > 16:
            buffer_size += 4
        elif available_memory_gb > 8:
            buffer_size += 2
        elif available_memory_gb < 2:
            buffer_size -= 1
            
        if physical_cores >= 8:
            buffer_size += 2
        elif physical_cores >= 4:
            buffer_size += 1
        elif physical_cores < 2:
            buffer_size -= 1
            
        if has_cuda:
            buffer_size += 2
            
        if current_load > 0.8:
            buffer_size -= 2
        elif current_load > 0.6:
            buffer_size -= 1
        elif current_load < 0.3:
            buffer_size += 1
            
        if disk_io_high:
            buffer_size -= 1
            
        buffer_size = max(3, min(12, buffer_size))
        
        logging.info(f"Calculated optimal buffer size: {buffer_size} (Memory: {available_memory_gb:.1f}GB, " +
                    f"CPU: {physical_cores}/{cpu_count} cores, Load: {current_load:.2f}, " +
                    f"CUDA: {has_cuda}, High Disk I/O: {disk_io_high})")
        
        return buffer_size
    except Exception as e:
        logging.warning(f"Failed to determine optimal buffer size: {e}")
        return 4

def detect_hardware_capabilities():
    """Detect hardware capabilities for performance optimization"""
    capabilities = {
        'cuda': False,
        'opencl': False,
        'avx': False,
        'avx2': False,
        'memory_gb': 4,
        'cpu_cores': 4,
        'gpu_vendor': None,
        'gpu_memory': 0
    }
    
    try:
        import psutil
        import cv2
        
        capabilities['cpu_cores'] = psutil.cpu_count(logical=False)
        capabilities['logical_cores'] = psutil.cpu_count(logical=True)
        
        mem = psutil.virtual_memory()
        capabilities['memory_gb'] = mem.total / (1024 * 1024 * 1024)
        
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            capabilities['cuda'] = True
            cv2.cuda.printCudaDeviceInfo(0)
            
            try:
                import pycuda.driver as cuda
                cuda.init()
                device = cuda.Device(0)
                capabilities['gpu_memory'] = device.total_memory() / (1024 * 1024 * 1024)
                capabilities['gpu_vendor'] = device.name()
            except ImportError:
                logging.info("PyCUDA not available, can't get detailed GPU info")
            except Exception as e:
                logging.debug(f"Could not get GPU info: {e}")
                
        if hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(True)
            capabilities['opencl'] = cv2.ocl.useOpenCL()
            if capabilities['opencl']:
                logging.info("OpenCL is available")
                
        try:
            import numpy as np
            np.__config__.show()
            
            if sys.platform.startswith('win'):
                try:
                    import cpuinfo
                    info = cpuinfo.get_cpu_info()
                    capabilities['avx'] = 'avx' in info.get('flags', [])
                    capabilities['avx2'] = 'avx2' in info.get('flags', [])
                except ImportError:
                    logging.debug("cpuinfo not available, can't detect AVX support")
        except Exception as e:
            logging.debug(f"Error detecting CPU features: {e}")
            
        logging.info(f"Hardware capabilities: CUDA={capabilities['cuda']}, "
                    f"OpenCL={capabilities['opencl']}, "
                    f"CPU Cores={capabilities['cpu_cores']}, "
                    f"RAM={capabilities['memory_gb']:.1f}GB")
                    
        if capabilities['cuda']:
            logging.info(f"GPU: {capabilities['gpu_vendor']} with {capabilities['gpu_memory']:.1f}GB memory")
            
    except Exception as e:
        logging.error(f"Error detecting hardware capabilities: {e}")
    
    return capabilities

def enable_hw_optimizations():
    """Enable hardware-specific optimizations"""
    try:
        import numpy as np
        import cv2
        import os
        import platform
        import multiprocessing
        
        # Detect physical core count for optimal thread settings
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or 4
            logical_cores = psutil.cpu_count(logical=True) or 8
        except ImportError:
            # Fallback if psutil not available
            physical_cores = multiprocessing.cpu_count() // 2 or 4
            logical_cores = multiprocessing.cpu_count() or 8
        
        # Set OpenCV thread optimizations - use number of physical cores
        cv2.setNumThreads(physical_cores)
        logging.info(f"Set OpenCV to use {physical_cores} threads")
        
        # Enable OpenCL if available
        if hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                logging.info("OpenCL acceleration enabled for OpenCV")
                
                # Set OpenCL device queue depth for better parallelism
                try:
                    cv2.ocl.Device.getDefault().setDeviceProperty(
                        cv2.ocl.QUEUE_DEPTH, 8)
                    logging.info("Set OpenCL queue depth to 8")
                except Exception as e:
                    logging.debug(f"Failed to set OpenCL queue depth: {e}")
            
        # Set NumPy thread optimizations
        try:
            # Try Intel MKL optimization first
            import mkl
            mkl.set_num_threads(physical_cores)
            # Enable fast vectorized operations
            mkl.set_fast_math(1)
            logging.info(f"MKL optimizations enabled for NumPy with {physical_cores} threads")
        except ImportError:
            try:
                # Show numpy configuration to check available optimizations
                np.show_config()
                
                # Set OpenBLAS threads if using OpenBLAS
                os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cores)
                logging.info(f"OpenBLAS thread count set to {physical_cores}")
                
                # Additional environment variables for linear algebra libraries
                os.environ['OMP_NUM_THREADS'] = str(physical_cores)
                os.environ['MKL_NUM_THREADS'] = str(physical_cores)
                os.environ['VECLIB_MAXIMUM_THREADS'] = str(physical_cores)
                os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)
                
                # Enable AVX/AVX2 if available through environment
                if platform.processor().startswith('Intel'):
                    logging.info("Intel processor detected - setting vectorization flags")
                    os.environ['NPY_DISABLE_CPU_FEATURES'] = ''  # Enable all features
                    
                    # Try to enable flush-to-zero and denormals-are-zero modes
                    # for better FP performance
                    try:
                        import ctypes
                        # Constants for MXCSR register
                        MXCSR_FTZ = 1 << 15  # Flush-to-zero
                        MXCSR_DAZ = 1 << 6   # Denormals-are-zero
                        MXCSR_MASK = MXCSR_FTZ | MXCSR_DAZ
                        
                        # Get current MXCSR value
                        mxcsr = ctypes.c_uint32()
                        ctypes.cdll.msvcrt._controlfp(ctypes.byref(mxcsr), 0)
                        
                        # Set FTZ and DAZ flags
                        new_mxcsr = mxcsr.value | MXCSR_MASK
                        ctypes.cdll.msvcrt._controlfp(new_mxcsr, MXCSR_MASK)
                        logging.info("Enabled FTZ and DAZ for better SIMD performance")
                    except Exception as e:
                        logging.debug(f"Failed to set FTZ/DAZ: {e}")
            except Exception as e:
                logging.debug(f"Failed to configure NumPy optimizations: {e}")
                
        # Windows-specific optimizations
        if platform.system() == 'Windows':
            # Set process priority to Above Normal
            try:
                import win32process
                import win32api
                import win32con
                
                pid = win32api.GetCurrentProcessId()
                handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
                win32process.SetPriorityClass(handle, win32process.ABOVE_NORMAL_PRIORITY_CLASS)
                win32api.CloseHandle(handle)
                logging.info("Set process priority to ABOVE_NORMAL_PRIORITY_CLASS")
            except Exception as e:
                logging.debug(f"Failed to set process priority: {e}")
                
            # CPU thread and memory optimization for Windows
            try:
                import psutil
                
                # Get the current process
                process = psutil.Process()
                
                try:
                    # Set high I/O priority (suppress errors)
                    if hasattr(psutil, 'IOPRIO_HIGH'):
                        try:
                            process.ionice(psutil.IOPRIO_HIGH)
                        except (PermissionError, psutil.AccessDenied):
                            pass  # Silently ignore permission errors
                        
                    # Adjust memory priority (suppress errors)
                    if hasattr(process, 'nice'):
                        try:
                            process.nice(psutil.HIGH_PRIORITY_CLASS)
                        except (PermissionError, psutil.AccessDenied):
                            pass  # Silently ignore permission errors
                            
                    logging.info("Process priority settings applied (where permissions allowed)")
                except Exception as e:
                    logging.debug(f"Some process priorities could not be set: {e}")
            except Exception as e:
                logging.debug(f"Failed to access process information: {e}")
                
    except Exception as e:
        logging.error(f"Failed to enable hardware optimizations: {e}")

def check_dependencies():
    """Check if all required Windows-specific dependencies are installed"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import win32api
    except ImportError:
        missing_deps.append("pywin32")
    
    try:
        import pygrabber
    except ImportError:
        missing_deps.append("pygrabber")
    
    try:
        import tensorflow
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def parse_args():
    """Parse command-line arguments for the waste classification app."""
    # Fix for pyinstaller/auto-py-to-exe hidden console mode
    # Since sys.stderr and sys.stdout are None in windowed mode
    import io
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    
    parser = argparse.ArgumentParser(description="Waste Classification System")
    parser.add_argument('--opencv-only', action='store_true', help='Use only OpenCV camera (no async)')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera functionality')
    parser.add_argument('--no-servo', action='store_true', help='Disable servo motor control')
    parser.add_argument('--test-servos', action='store_true', help='Run servo test mode with sweep and easing motion options')
    parser.add_argument('--test-tts', action='store_true', help='Test text-to-speech system')
    
    return parser.parse_args()

def enable_simd_avx_optimizations():
    """
    Enable SIMD/AVX optimizations for NumPy and OpenCV operations
    """
    logging.info("Configuring SIMD/AVX optimizations...")
    
    try:
        import numpy as np
        
        # Display NumPy config to check SIMD optimizations
        logging.debug("NumPy configuration:")
        np.__config__.show()
        
        # For NumPy linked against MKL, enable all optimizations
        try:
            import mkl
            mkl.set_threading_layer('intel')
            mkl.set_num_threads(multiprocessing.cpu_count())
            logging.info("Enabled Intel MKL optimizations for NumPy")
        except ImportError:
            logging.debug("Intel MKL not available")
        
        # Enable OpenMP parallelism for NumPy operations
        cpu_count = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        logging.debug(f"Set threading environment variables to use {cpu_count} threads")
        
        # For OpenCV, check and enable optimizations
        build_info = cv2.getBuildInformation()
        logging.debug("Checking OpenCV build information for optimization capabilities")
        
        if 'CPU_BASELINE' in build_info:
            # Split the string safely
            parts = build_info.split('CPU_BASELINE')
            if len(parts) > 1:
                baseline_info = parts[1].split('\n')[0]
                logging.info(f"OpenCV CPU optimizations available: {baseline_info}")
        
        # Enable OpenCV thread optimizations
        cv2.setNumThreads(cpu_count)
        logging.info(f"Set OpenCV to use {cpu_count} threads")
        
        # Check for CPU features
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            avx_support = 'avx' in flags
            avx2_support = 'avx2' in flags
            fma_support = 'fma' in flags
            
            # Log CPU features at appropriate level
            logging.info(f"CPU SIMD features - AVX: {avx_support}, AVX2: {avx2_support}, FMA: {fma_support}")
            
            # Set environment variables based on CPU features
            if avx2_support:
                os.environ['OPENBLAS_CORETYPE'] = 'Haswell'
                logging.debug("Set OPENBLAS_CORETYPE=Haswell for AVX2 support")
            elif avx_support:
                os.environ['OPENBLAS_CORETYPE'] = 'Sandybridge'
                logging.debug("Set OPENBLAS_CORETYPE=Sandybridge for AVX support")
                
        except ImportError:
            logging.warning("cpuinfo package not available, cannot detect CPU SIMD features")
            
        logging.info("SIMD/AVX optimizations configuration completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error enabling SIMD/AVX optimizations: {str(e)}")
        logging.debug(f"SIMD optimization error details", exc_info=True)
        return False

def initialize_background_scheduler():
    """Initialize the background task scheduler"""
    global background_scheduler
    background_scheduler = BackgroundTaskScheduler(
        max_workers=2,
        name="AppBGTasks"
    )
    return background_scheduler

def main():
    """Main entry point for the application."""
    try:
        # Log startup information
        logging.info("Starting Waste Classification System")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"OpenCV version: {cv2.__version__}")
        logging.info(f"Platform: {platform.platform()}")
        
        # Check if we're running as a frozen application
        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            logging.info(f"Running as frozen application from: {sys.executable}")
        
        if is_frozen and (sys.stderr is None or sys.stdout is None):
            # We're running as a frozen application with no console
            # Use default values instead of parsing arguments
            args = argparse.Namespace(
                opencv_only=False,
                no_camera=False, 
                model_path=None,
                ensemble=True,  # Default to using ensemble
                no_servo=True,  # Default to no servo
                width=1280,
                height=720,
                fullscreen=False,
                camera_index=0,
                debug=False
            )
            logging.info("Running in frozen mode with default arguments")
        else:
            # Normal operation - parse arguments
            logging.info("Parsing command line arguments")
            args = parse_args()
            logging.info(f"Arguments: {args}")
            
        # Check if all required dependencies are installed
        logging.info("Checking dependencies")
        check_dependencies()
        
        # Enable SIMD optimizations for numpy if available
        logging.info("Setting up performance optimizations")
        enable_simd_avx_optimizations()
        
        # Initialize background task scheduler
        logging.info("Initializing background task scheduler")
        global background_scheduler
        background_scheduler = initialize_background_scheduler()
        
        # Detect hardware capabilities
        logging.info("Detecting hardware capabilities")
        hw_capabilities = detect_hardware_capabilities()
        logging.info(f"Detected hardware: {hw_capabilities}")
        
        # Enable hardware-specific optimizations
        logging.info("Enabling hardware optimizations")
        enable_hw_optimizations()
        
        # Import and initialize Windows-specific memory optimizations
        try:
            # Log availability of memory page locking
            memory_manager = MemoryManager()
            if memory_manager.can_lock_memory:
                logging.info("Memory page locking capability available")
            
            # Check for SIMD support
            from utils import HAS_NUMPY_SIMD
            if HAS_NUMPY_SIMD:
                logging.info("NumPy SIMD optimizations available")
        except ImportError:
            logging.warning("Performance optimization modules not available")
        
        # Dynamically determine performance settings based on hardware capabilities
        high_performance = False
        frame_skip = 0
        
        # Set high performance mode if CPU cores are limited or system memory is low
        if hw_capabilities.get('cpu_cores', 4) < 4 or hw_capabilities.get('memory_gb', 8) < 4:
            high_performance = True
            logging.info("Automatically enabling high performance mode due to hardware constraints")
        
        # Determine frame skip based on system capabilities
        if high_performance:
            # More aggressive frame skipping for limited hardware
            frame_skip = 2 if hw_capabilities.get('cpu_cores', 4) < 2 else 1
            logging.info(f"Setting frame skip to {frame_skip} for better performance")
        
        # Initialize application with auto-detected hardware
        logging.info("Initializing application components")
        # Always attempt to use ensemble classification with fallback to single model
        context = initialize_app(
            use_async_camera=not args.opencv_only,
            frame_skip=frame_skip,
            use_ensemble=True,  # Always try to use ensemble
            high_performance=high_performance,
            use_hw_accel=True,
            start_fullscreen=True,  # Always start in fullscreen mode
            no_camera=args.no_camera,
            no_servo=args.no_servo
        )
        
        # Handle test modes
        if hasattr(args, 'test_servos') and args.test_servos:
            logging.info("Running servo test mode")
            if test_servos():
                logging.info("Servo test completed successfully")
            else:
                logging.error("Servo test failed")
            return
            
        if hasattr(args, 'test_tts') and args.test_tts:
            logging.info("Running TTS test mode")
            if test_tts():
                logging.info("TTS test completed successfully")
            else:
                logging.error("TTS test failed")
            return
        
        # Add TTS player to context
        context["tts_player"] = tts_player
        
        # Start streaming server in a separate thread
        stream_thread = threading.Thread(
            target=lambda: start_stream_server(host='0.0.0.0', port=5000),
            daemon=True
        )
        stream_thread.start()
        
        # Run the main application loop
        logging.info("Starting main application loop")
        run_app_loop(context)
        
        # Clean up TTS resources
        if tts_player:
            tts_player.cleanup()
        
        # Inside your main processing loop, after processing each frame:
        if processed_frame is not None:  # Your processed frame with UI
            update_frame(processed_frame)
        
    except KeyboardInterrupt:
        stop_stream_server()
        # ... existing cleanup code ...
    except Exception as e:
        stop_stream_server()
        # ... existing error handling code ...

# Add a function to show error message in a GUI window
def show_error_message(message):
    """Display an error message in a GUI window."""
    try:
        # Try using tkinter if available
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Waste Classification System Error", message)
        root.destroy()
    except ImportError:
        # Fallback to console
        try:
            print(f"ERROR: {message}", file=sys.stderr)
        except:
            pass

def test_servos():
    """
    Run a servo test routine to verify proper operation of all connected servos.
    This function will cycle through all servos, testing their sweep movements.
    """
    print("=== SERVO TEST MODE ===")
    print("Testing all configured servos with sweep motion. Press Ctrl+C to cancel.")
    
    try:
        # Import required modules
        from servo_controller import ServoController
        from config import SERVO_PINS, BIN_NAMES
        import time
        import logging
        
        # Configure logging for the test
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Set up servo controller
        logging.info("Initializing servo controller...")
        servo = ServoController()
        
        if not servo.is_connected():
            logging.error("Failed to connect to Arduino. Check connections and try again.")
            return False
        
        # Check servo status and reset any non-operational servos
        print("\nChecking servo status...")
        servo_status = servo.get_servo_status()
        
        any_non_operational = False
        for bin_index, status in servo_status.items():
            bin_name = BIN_NAMES.get(bin_index, f"Bin {bin_index}")
            if not status.get("operational", True):
                print(f"WARNING: {bin_name} (bin {bin_index}) is marked as non-operational")
                print(f"  - Last error: {status.get('last_error', 'Unknown')}")
                print(f"  - Last operation: {status.get('last_operation', 'Unknown')}")
                any_non_operational = True
            else:
                print(f"OK: {bin_name} (bin {bin_index}) is operational")
        
        if any_non_operational:
            print("\nWould you like to reset non-operational servos? (y/n)")
            choice = input("Enter choice: ").lower().strip()
            if choice == 'y' or choice == 'yes':
                servo.reset_servo_status()
                print("All servos have been reset to operational status.")
                
        # Enable sweep motion with default settings
        servo.set_smooth_movement(enabled=True, delay=0.015)
        logging.info("Sweep motion enabled for servo testing")
            
        # First, close all bins
        for bin_index in SERVO_PINS.keys():
            bin_name = BIN_NAMES.get(bin_index, f"Bin {bin_index}")
            logging.info(f"Closing {bin_name}...")
            servo.close_lid(bin_index)
            time.sleep(0.5)
            
        # Allow user to select speed for the test
        print("\nSelect sweep speed:")
        print("1. Fast (0.01s delay)")
        print("2. Medium (0.015s delay)")
        print("3. Slow (0.02s delay)")
        
        try:
            choice = input("Enter choice (1-3): ")
            choice = int(choice)
            if choice == 1:
                sweep_delay = 0.01
                logging.info("Fast sweep selected")
            elif choice == 2:
                sweep_delay = 0.015
                logging.info("Medium sweep selected")
            elif choice == 3:
                sweep_delay = 0.02
                logging.info("Slow sweep selected")
            else:
                sweep_delay = 0.015  # Default to medium
                logging.info("Invalid choice. Using medium sweep")
        except:
            sweep_delay = 0.015  # Default to medium
            logging.info("Using default medium sweep")
            
        # Update sweep settings
        servo.set_smooth_movement(enabled=True, delay=sweep_delay)
        
        # Ask which bins to test
        print("\nSelect bins to test:")
        print("0. Test all bins")
        for bin_index in sorted(SERVO_PINS.keys()):
            bin_name = BIN_NAMES.get(bin_index, f"Bin {bin_index}")
            print(f"{bin_index+1}. Test only {bin_name}")
        
        try:
            choice = input("Enter choice: ")
            choice = int(choice)
            if choice == 0:
                # Test all bins
                bins_to_test = sorted(SERVO_PINS.keys())
            elif choice-1 in SERVO_PINS:
                # Test only the selected bin
                bins_to_test = [choice-1]
            else:
                print("Invalid choice. Testing all bins.")
                bins_to_test = sorted(SERVO_PINS.keys())
        except:
            print("Invalid choice. Testing all bins.")
            bins_to_test = sorted(SERVO_PINS.keys())
            
        # Test each selected bin in sequence
        for bin_index in bins_to_test:
            bin_name = BIN_NAMES.get(bin_index, f"Bin {bin_index}")
            
            logging.info(f"Testing {bin_name}...")
            
            # Open the bin
            logging.info(f"Opening {bin_name}...")
            result = servo.open_lid(bin_index)
            if not result:
                print(f"ERROR: Failed to open {bin_name}!")
                servo_status = servo.get_servo_status(bin_index)
                if servo_status:
                    print(f"Error details: {servo_status.get('last_error', 'Unknown')}")
                    
                # Ask if user wants to continue testing other bins
                if len(bins_to_test) > 1:
                    print(f"Continue testing other bins? (y/n)")
                    choice = input("Enter choice: ").lower().strip()
                    if choice != 'y' and choice != 'yes':
                        break
                continue
                
            time.sleep(2)
            
            # Close the bin
            logging.info(f"Closing {bin_name}...")
            result = servo.close_lid(bin_index)
            if not result:
                print(f"ERROR: Failed to close {bin_name}!")
                servo_status = servo.get_servo_status(bin_index)
                if servo_status:
                    print(f"Error details: {servo_status.get('last_error', 'Unknown')}")
                    
                # Ask if user wants to continue testing other bins
                if len(bins_to_test) > 1:
                    print(f"Continue testing other bins? (y/n)")
                    choice = input("Enter choice: ").lower().strip()
                    if choice != 'y' and choice != 'yes':
                        break
                continue
                
            time.sleep(1)
            
            # Test again with different timing
            logging.info(f"Opening {bin_name} again...")
            result = servo.open_lid(bin_index)
            if not result:
                print(f"ERROR: Failed to open {bin_name} on second attempt!")
                continue
                
            time.sleep(1)
            
            logging.info(f"Closing {bin_name} again...")
            result = servo.close_lid(bin_index)
            if not result:
                print(f"ERROR: Failed to close {bin_name} on second attempt!")
                continue
                
            time.sleep(1)
            
            logging.info(f"Test complete for {bin_name}")
            
        # Display final servo status
        print("\nFinal servo status:")
        servo_status = servo.get_servo_status()
        for bin_index, status in servo_status.items():
            bin_name = BIN_NAMES.get(bin_index, f"Bin {bin_index}")
            if status.get("operational", True):
                print(f"OK: {bin_name} (bin {bin_index}) is operational")
            else:
                print(f"FAILED: {bin_name} (bin {bin_index}) is non-operational")
                print(f"  - Last error: {status.get('last_error', 'Unknown')}")
            
        logging.info("Servo test completed!")
        return True
        
    except KeyboardInterrupt:
        print("Test interrupted by user.")
        return False
    except Exception as e:
        logging.error(f"Error during servo test: {e}")
        return False

def test_tts():
    """Test the TTS system"""
    try:
        from tts_player import tts_player
        print("Testing TTS system...")
        
        if not tts_player.initialized:
            print("ERROR: TTS system not initialized!")
            return False
            
        print(f"TTS initialized: {tts_player.initialized}")
        print(f"TTS speaker: {tts_player.speaker}")
        
        # Test basic speech
        print("Testing basic speech...")
        tts_player.speak("Testing audio feedback system")
        time.sleep(2)
        
        # Test bin announcements
        print("Testing bin announcements...")
        for bin_index in range(4):
            tts_player.speak_bin_open(bin_index)
            time.sleep(1)
            tts_player.speak_bin_close(bin_index)
            time.sleep(1)
            
        print("TTS test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during TTS test: {e}")
        return False

if __name__ == "__main__":
    main()