#!/usr/bin/env python
# hardware_accel.py - Hardware acceleration support for different platforms

import os
import sys
import logging
import platform
import ctypes
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import multiprocessing
import threading
import importlib
import subprocess
import json
from pathlib import Path
import time
import importlib.util
import traceback

# Flag to check if we're on Windows
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'

# Global acceleration flags
HAS_CUDA = False
HAS_OPENCL = False
HAS_DIRECTML = False
HAS_ONEDNN = False
HAS_DPCPP = False  # Intel Data Parallel C++
HAS_MKL = False    # Intel Math Kernel Library

# Safely check for dependencies
def is_module_available(module_name):
    """Check if a module is available without importing it"""
    return importlib.util.find_spec(module_name) is not None

# Windows-specific features
HAS_WIN32 = is_module_available('win32api') if IS_WINDOWS else False

# Initialize hardware acceleration support
def initialize_acceleration(force_cpu: bool = False, force_opencl: bool = False) -> Dict[str, Any]:
    """
    Initialize hardware acceleration support and return capabilities.
    
    Args:
        force_cpu: Force CPU-only mode even if GPU is available
        force_opencl: If True, prefer OpenCL over CUDA
        
    Returns:
        Dictionary of acceleration capabilities
    """
    global HAS_CUDA, HAS_OPENCL, HAS_DIRECTML, HAS_ONEDNN, HAS_DPCPP, HAS_MKL
    
    capabilities = {
        'platform': platform.system(),
        'cpu_cores': multiprocessing.cpu_count(),
        'force_cpu': force_cpu,
        'cuda': {
            'available': False,
            'devices': [],
            'version': None
        },
        'opencl': {
            'available': False,
            'devices': [],
            'version': None
        },
        'directml': {
            'available': False,
            'version': None
        },
        'intel': {
            'onednn': False,
            'dpcpp': False,
            'mkl': False,
            'ipp': False,
            'vtune': False
        }
    }
    
    # Set relevant environment variables
    if not force_cpu:
        # Common optimization variables
        os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
        
        # TensorFlow optimizations
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF log verbosity
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        # OpenCV optimizations
        os.environ['OPENCV_OPENCL_RUNTIME'] = ''  # Let OpenCV decide OpenCL runtime
    else:
        # Disable GPU acceleration if force_cpu is True
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        os.environ['TF_FORCE_UNIFIED_MEMORY'] = 'false'
        
    if not force_cpu:
        # Check for CUDA support
        try:
            import torch
            if torch.cuda.is_available():
                HAS_CUDA = True
                capabilities['cuda']['available'] = True
                capabilities['cuda']['version'] = torch.version.cuda
                capabilities['cuda']['devices'] = []
                
                # Get device information
                for i in range(torch.cuda.device_count()):
                    device = {
                        'name': torch.cuda.get_device_name(i),
                        'memory': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                        'compute_capability': f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                    }
                    capabilities['cuda']['devices'].append(device)
                    
                logging.info(f"CUDA acceleration available: {torch.cuda.device_count()} device(s)")
        except (ImportError, Exception) as e:
            # Try with cv2
            try:
                import cv2
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    HAS_CUDA = True
                    capabilities['cuda']['available'] = True
                    capabilities['cuda']['version'] = cv2.getBuildInformation().split('CUDA:')[1].split('\n')[0].strip() if 'CUDA:' in cv2.getBuildInformation() else 'Unknown'
                    logging.info(f"CUDA acceleration available through OpenCV")
            except (ImportError, Exception):
                logging.debug("CUDA not available")
        
        # Check for OpenCL support
        try:
            import pyopencl as cl
            HAS_OPENCL = True
            capabilities['opencl']['available'] = True
            
            try:
                platforms = cl.get_platforms()
                capabilities['opencl']['version'] = platforms[0].version if platforms else None
                
                for platform in platforms:
                    for device in platform.get_devices():
                        device_info = {
                            'name': device.name,
                            'type': cl.device_type.to_string(device.type),
                            'platform': platform.name,
                            'memory': device.global_mem_size / (1024**3),  # GB
                            'compute_units': device.max_compute_units
                        }
                        capabilities['opencl']['devices'].append(device_info)
                
                logging.info(f"OpenCL acceleration available: {len(capabilities['opencl']['devices'])} device(s)")
            except cl.LogicError as e:
                logging.warning(f"OpenCL initialization error: {e}")
                HAS_OPENCL = False
        except ImportError:
            try:
                # Check if OpenCV has OpenCL support
                import cv2
                if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                    HAS_OPENCL = True
                    capabilities['opencl']['available'] = True
                    cv2.ocl.setUseOpenCL(True)
                    logging.info("OpenCL acceleration available through OpenCV")
            except (ImportError, Exception):
                logging.debug("OpenCL not available")
        
        # Check for DirectML support (Windows only)
        if IS_WINDOWS:
            try:
                # Try to import DirectML
                import tensorflow as tf
                from tensorflow.python.eager import context
                
                # Check if DirectML is in the device list
                devices = tf.config.list_physical_devices()
                for device in devices:
                    if 'DirectML' in device.name:
                        HAS_DIRECTML = True
                        capabilities['directml']['available'] = True
                        logging.info("DirectML acceleration available through TensorFlow")
                        break
                
                # If not found through TensorFlow, check through ONNX Runtime
                if not HAS_DIRECTML:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if 'DmlExecutionProvider' in providers:
                        HAS_DIRECTML = True
                        capabilities['directml']['available'] = True
                        logging.info("DirectML acceleration available through ONNX Runtime")
            except (ImportError, Exception):
                logging.debug("DirectML not available")
                
        # Check for Intel oneAPI/oneDNN support
        try:
            # Check for Intel MKL (Math Kernel Library)
            try:
                import mkl
                HAS_MKL = True
                capabilities['intel']['mkl'] = True
                
                # Configure MKL for best performance
                mkl.set_num_threads(multiprocessing.cpu_count())
                logging.info("Intel MKL acceleration available")
            except ImportError:
                logging.debug("Intel MKL not available")
            
            # Check for oneDNN (Deep Neural Network Library)
            try:
                # Check if TensorFlow was built with oneDNN support
                import tensorflow as tf
                build_info = tf.sysconfig.get_build_info()
                if 'onednn' in build_info.get('cpu_compiler_flags', '').lower() or 'mkldnn' in build_info.get('cpu_compiler_flags', '').lower():
                    HAS_ONEDNN = True
                    capabilities['intel']['onednn'] = True
                    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
                    logging.info("Intel oneDNN acceleration available through TensorFlow")
            except (ImportError, AttributeError, Exception):
                try:
                    # Alternative check through torch
                    import torch
                    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
                        HAS_ONEDNN = True
                        capabilities['intel']['onednn'] = True
                        logging.info("Intel oneDNN acceleration available through PyTorch")
                except ImportError:
                    logging.debug("Intel oneDNN not available")
            
            # Check for Intel IPP (Integrated Performance Primitives)
            try:
                import cv2
                build_info = cv2.getBuildInformation()
                if 'IPP:' in build_info and 'NO' not in build_info.split('IPP:')[1].split('\n')[0]:
                    capabilities['intel']['ipp'] = True
                    logging.info("Intel IPP acceleration available through OpenCV")
            except (ImportError, Exception):
                logging.debug("Intel IPP not available")
            
        except Exception as e:
            logging.debug(f"Error detecting Intel acceleration: {e}")
    
    # Log hardware acceleration status
    logging.info(f"Hardware acceleration status: CUDA={HAS_CUDA}, OpenCL={HAS_OPENCL}, DirectML={HAS_DIRECTML}, oneDNN={HAS_ONEDNN}, MKL={HAS_MKL}")
    
    return capabilities

def optimize_tensorflow_for_cpu():
    """Configure TensorFlow for optimal CPU performance with oneDNN"""
    try:
        import tensorflow as tf
        
        # Enable oneDNN optimizations
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        
        # Enable optimizations for modern CPUs
        os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'
        
        # Set intra/inter op threads based on CPU count
        physical_cores = multiprocessing.cpu_count()
        tf.config.threading.set_intra_op_parallelism_threads(physical_cores)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, physical_cores // 2))
        
        # If oneDNN is available, use oneDNN optimized ops
        if HAS_ONEDNN:
            logging.info("Configuring TensorFlow to use oneDNN optimizations")
        
        logging.info(f"TensorFlow CPU optimization complete: {physical_cores} cores, "
                    f"intra_threads={physical_cores}, inter_threads={max(1, physical_cores // 2)}")
        
        return True
    except (ImportError, Exception) as e:
        logging.error(f"Failed to optimize TensorFlow for CPU: {e}")
        return False

def optimize_tensorflow_for_directml():
    """Configure TensorFlow to use DirectML on Windows"""
    if not IS_WINDOWS or not HAS_DIRECTML:
        return False
        
    try:
        import tensorflow as tf
        
        # Force TensorFlow to use DirectML
        # This requires tensorflow-directml package
        try:
            # This will only work if tensorflow-directml package is installed
            tf.config.set_visible_devices([], 'GPU')  # Hide CUDA devices if any
            
            # Set environment variable for DirectML
            os.environ['TF_DIRECTML_ENABLE'] = '1'
            
            logging.info("Configured TensorFlow to use DirectML acceleration")
            return True
        except Exception as e:
            logging.error(f"Failed to configure TensorFlow for DirectML: {e}")
            return False
    except ImportError:
        logging.error("TensorFlow not available for DirectML configuration")
        return False

def optimize_onnx_for_directml():
    """Configure ONNX Runtime to use DirectML on Windows"""
    if not IS_WINDOWS or not HAS_DIRECTML:
        return False
        
    try:
        import onnxruntime as ort
        
        # Check available providers
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            # Set default provider order to prefer DirectML
            ort.set_default_logger_severity(3)  # Set to warning level
            
            # Don't need to do anything else - the ModelWrapper class in ensemble_classifier.py
            # will detect and use DirectML provider when creating sessions
            
            logging.info("ONNX Runtime configured to use DirectML acceleration")
            return True
        else:
            logging.warning("DirectML provider not available in ONNX Runtime")
            return False
    except ImportError:
        logging.error("ONNX Runtime not available for DirectML configuration")
        return False

def optimize_opencv_for_opencl():
    """Configure OpenCV to use OpenCL acceleration when available"""
    if not HAS_OPENCL:
        return False
        
    try:
        import cv2
        
        if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
            # Enable OpenCL
            cv2.ocl.setUseOpenCL(True)
            
            # Log OpenCL device info
            device_name = cv2.ocl.Device_getDefault().name()
            logging.info(f"OpenCV configured to use OpenCL acceleration on {device_name}")
            return True
        else:
            logging.warning("OpenCL support not available in OpenCV")
            return False
    except (ImportError, Exception) as e:
        logging.error(f"Failed to configure OpenCV for OpenCL: {e}")
        return False

def benchmark_acceleration(iterations=10):
    """Run a quick benchmark to measure hardware acceleration performance"""
    results = {
        'cpu': 0,
        'cuda': 0,
        'opencl': 0,
        'directml': 0,
        'onednn': 0
    }
    
    # Define test sizes
    size = (512, 512)
    large_size = (1024, 1024)
    
    # Generate test data
    np_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    large_array = np.random.randint(0, 255, (*large_size, 3), dtype=np.uint8)
    
    # Test CPU performance (NumPy baseline)
    try:
        start = time.time()
        for _ in range(iterations):
            # Perform typical image operations
            gray = np.dot(np_array, [0.299, 0.587, 0.114])
            blurred = np.zeros_like(np_array)
            # Simulate Gaussian blur with a simple 5x5 box filter
            for i in range(2, np_array.shape[0]-2):
                for j in range(2, np_array.shape[1]-2):
                    blurred[i, j] = np_array[i-2:i+3, j-2:j+3].mean(axis=(0, 1))
            
            # Perform resizing
            h, w = np_array.shape[:2]
            new_h, new_w = h//2, w//2
            resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            for i in range(new_h):
                for j in range(new_w):
                    resized[i, j] = np_array[i*2, j*2]
        
        cpu_time = (time.time() - start) / iterations
        results['cpu'] = cpu_time
        logging.info(f"CPU baseline: {cpu_time:.4f} seconds per iteration")
    except Exception as e:
        logging.error(f"CPU benchmark failed: {e}")
    
    # Test OpenCV with CUDA
    if HAS_CUDA:
        try:
            import cv2
            start = time.time()
            for _ in range(iterations):
                # Upload to GPU
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(np_array)
                
                # Perform operations on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGR2GRAY)
                gpu_blur = cv2.cuda.blur(gpu_src, (5, 5))
                gpu_resize = cv2.cuda.resize(gpu_src, (new_w, new_h))
                
                # Download results
                gray = gpu_gray.download()
                blurred = gpu_blur.download()
                resized = gpu_resize.download()
                
            cuda_time = (time.time() - start) / iterations
            results['cuda'] = cuda_time
            speedup = cpu_time / cuda_time if cuda_time > 0 else 0
            logging.info(f"CUDA: {cuda_time:.4f} seconds per iteration ({speedup:.2f}x speedup)")
        except Exception as e:
            logging.error(f"CUDA benchmark failed: {e}")
    
    # Test OpenCV with OpenCL
    if HAS_OPENCL:
        try:
            import cv2
            cv2.ocl.setUseOpenCL(True)
            
            start = time.time()
            for _ in range(iterations):
                # Create UMat objects
                src_umat = cv2.UMat(np_array)
                
                # Perform operations
                gray_umat = cv2.cvtColor(src_umat, cv2.COLOR_BGR2GRAY)
                blur_umat = cv2.blur(src_umat, (5, 5))
                resize_umat = cv2.resize(src_umat, (new_w, new_h))
                
                # Get results
                gray = gray_umat.get()
                blurred = blur_umat.get()
                resized = resize_umat.get()
                
            opencl_time = (time.time() - start) / iterations
            results['opencl'] = opencl_time
            speedup = cpu_time / opencl_time if opencl_time > 0 else 0
            logging.info(f"OpenCL: {opencl_time:.4f} seconds per iteration ({speedup:.2f}x speedup)")
        except Exception as e:
            logging.error(f"OpenCL benchmark failed: {e}")
    
    # Test with oneDNN (TensorFlow)
    if HAS_ONEDNN:
        try:
            import tensorflow as tf
            # Enable oneDNN optimizations
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
            
            tf_array = tf.convert_to_tensor(large_array)
            
            start = time.time()
            for _ in range(iterations):
                # Convert to grayscale
                rgb_weights = tf.constant([0.299, 0.587, 0.114])
                gray = tf.tensordot(tf_array, rgb_weights, axes=1)
                
                # Blur
                blur_filter = tf.ones((5, 5, 3, 3), dtype=tf.float32) / 25.0
                blur_input = tf.expand_dims(tf.cast(tf_array, tf.float32), 0)
                blurred = tf.nn.conv2d(blur_input, blur_filter, strides=[1, 1, 1, 1], padding='SAME')
                
                # Resize
                resized = tf.image.resize(blur_input, (new_h, new_w))
                
                # Ensure operations complete (via eager execution)
                gray_np = gray.numpy()
                blurred_np = tf.squeeze(blurred).numpy().astype(np.uint8)
                resized_np = tf.squeeze(resized).numpy().astype(np.uint8)
                
            onednn_time = (time.time() - start) / iterations
            results['onednn'] = onednn_time
            speedup = cpu_time / onednn_time if onednn_time > 0 else 0
            logging.info(f"oneDNN: {onednn_time:.4f} seconds per iteration ({speedup:.2f}x speedup)")
        except Exception as e:
            logging.error(f"oneDNN benchmark failed: {e}")
    
    # Test with DirectML (ONNX Runtime)
    if HAS_DIRECTML:
        try:
            import onnxruntime as ort
            import numpy as np
            
            # Create a simple ONNX model for testing
            try:
                import onnx
                from onnx import helper
                from onnx import TensorProto
                
                # Create a simple model that does image processing
                X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 512, 512])
                Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 256, 256])
                
                # Simple resize + blur node
                resize_node = helper.make_node(
                    'Resize',
                    inputs=['X'],
                    outputs=['resized'],
                    mode='linear',
                    sizes=[1, 3, 256, 256]
                )
                
                # Output node
                identity_node = helper.make_node(
                    'Identity',
                    inputs=['resized'],
                    outputs=['Y']
                )
                
                graph = helper.make_graph(
                    [resize_node, identity_node],
                    'test_model',
                    [X],
                    [Y]
                )
                
                model = helper.make_model(graph)
                onnx.checker.check_model(model)
                
                # Save model to temp file
                import tempfile
                model_path = os.path.join(tempfile.gettempdir(), 'test_model.onnx')
                onnx.save(model, model_path)
                
                # Create session with DirectML
                dml_session = ort.InferenceSession(
                    model_path, 
                    providers=['DmlExecutionProvider']
                )
                
                # Prepare input
                input_array = np.transpose(large_array.astype(np.float32), (2, 0, 1))
                input_array = np.expand_dims(input_array, 0)  # Add batch dimension
                
                start = time.time()
                for _ in range(iterations):
                    output = dml_session.run(None, {'X': input_array})
                
                directml_time = (time.time() - start) / iterations
                results['directml'] = directml_time
                speedup = cpu_time / directml_time if directml_time > 0 else 0
                logging.info(f"DirectML: {directml_time:.4f} seconds per iteration ({speedup:.2f}x speedup)")
                
                # Cleanup
                os.remove(model_path)
            except Exception as e:
                logging.error(f"DirectML ONNX model creation failed: {e}")
        except Exception as e:
            logging.error(f"DirectML benchmark failed: {e}")
    
    # Return benchmark results with speedup factors
    if results['cpu'] > 0:
        for key in results:
            if key != 'cpu' and results[key] > 0:
                results[f"{key}_speedup"] = results['cpu'] / results[key]
    
    return results

# Additional utility functions

def get_optimal_device():
    """Returns the optimal computing device based on availability"""
    if HAS_CUDA:
        return 'cuda'
    elif HAS_DIRECTML and IS_WINDOWS:
        return 'directml'
    elif HAS_OPENCL:
        return 'opencl'
    elif HAS_ONEDNN:
        return 'cpu-onednn'
    else:
        return 'cpu'

def configure_environment_for_device(device: str):
    """Configure environment variables for the specified device"""
    if device == 'cuda':
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '2'
        return True
    elif device == 'directml' and IS_WINDOWS:
        os.environ['TF_DIRECTML_ENABLE'] = '1'
        return True
    elif device == 'opencl':
        os.environ['OPENCV_OPENCL_DEVICE'] = ''  # Use default device
        return True
    elif device == 'cpu-onednn':
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
        return True
    else:
        # Default CPU configuration
        os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
        return True

def create_wrapper_functions():
    """Create wrapper functions for platform-specific implementations"""
    # Windows-specific features
    if IS_WINDOWS:
        HAS_DIRECTML = False  # Will be set during initialization
        HAS_DIRECTML = init_directml()

    # GPU acceleration
    HAS_CUDA = False
    HAS_OPENCL = False
    CUDA_VERSION = None

    # Create wrapper functions for platform-specific implementations
    def init_directml():
        """Initialize DirectML support (Windows only)"""
        global HAS_DIRECTML
        
        if not IS_WINDOWS:
            logging.info("DirectML is only supported on Windows")
            return False
        
        try:
            # Check if tensorflow-directml is installed
            if not is_module_available('tensorflow_directml'):
                logging.info("tensorflow-directml not installed, DirectML acceleration not available")
                return False
            
            import tensorflow as tf
            import tensorflow_directml as tfdml
            
            # Initialize DirectML device
            dml_device = tfdml.get_device()
            tfdml.set_default_device(dml_device)
            
            # Test if DirectML works
            logging.info(f"DirectML device initialized: {dml_device}")
            HAS_DIRECTML = True
            return True
        except ImportError as e:
            logging.warning(f"Could not initialize DirectML: {e}")
            return False
        except Exception as e:
            logging.warning(f"DirectML initialization error: {e}")
            traceback.print_exc()
            return False

    def init_opencv_cuda():
        """Initialize CUDA support for OpenCV"""
        global HAS_CUDA, CUDA_VERSION
        
        try:
            import cv2
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Try to get CUDA device info
                cv2.cuda.printCudaDeviceInfo(0)
                
                # Try to get CUDA version
                try:
                    cuda_version = cv2.cuda.getDevice()
                    CUDA_VERSION = f"CUDA Device {cuda_version}"
                except:
                    CUDA_VERSION = "Unknown CUDA version"
                    
                HAS_CUDA = True
                logging.info(f"OpenCV CUDA support initialized: {CUDA_VERSION}")
                return True
            else:
                logging.info("OpenCV CUDA support not available")
                return False
        except Exception as e:
            logging.warning(f"Error initializing OpenCV CUDA: {e}")
            return False

    def init_opencl():
        """Initialize OpenCL for cross-platform GPU acceleration"""
        global HAS_OPENCL
        
        try:
            import cv2
            if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                # Try to enable OpenCL
                cv2.ocl.setUseOpenCL(True)
                
                if cv2.ocl.useOpenCL():
                    # Get available OpenCL devices
                    cv2.ocl.Device_getDefault()
                    
                    HAS_OPENCL = True
                    logging.info(f"OpenCL acceleration enabled")
                    return True
                else:
                    logging.info("OpenCL is available but not enabled")
                    return False
            else:
                logging.info("OpenCL not available")
                return False
        except Exception as e:
            logging.warning(f"Error initializing OpenCL: {e}")
            return False

    def get_platform_info():
        """Get detailed platform information"""
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'is_windows': IS_WINDOWS,
            'is_linux': IS_LINUX,
            'is_macos': IS_MACOS
        }
        
        # Add Windows-specific info
        if IS_WINDOWS:
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                    platform_info['windows_build'] = winreg.QueryValueEx(key, "CurrentBuild")[0]
                    platform_info['windows_ubr'] = winreg.QueryValueEx(key, "UBR")[0]
                    platform_info['windows_product_name'] = winreg.QueryValueEx(key, "ProductName")[0]
            except:
                pass
        
        return platform_info

    def init_platform_optimizations():
        """Initialize platform-specific optimizations"""
        
        capabilities = {
            'platform': get_platform_info(),
            'hardware_acceleration': {
                'cuda': HAS_CUDA,
                'opencl': HAS_OPENCL,
                'directml': HAS_DIRECTML
            },
            'optimizations': {}
        }
        
        # Initialize platform-specific optimizations
        if IS_WINDOWS:
            capabilities['optimizations']['windows'] = {
                'win32_api': HAS_WIN32,
                'dark_mode': check_dark_mode_support(),
                'memory_priority': init_memory_priority() if HAS_WIN32 else False
            }
        
        # Common optimizations for all platforms
        capabilities['optimizations']['common'] = {
            'opencl': init_opencl(),
            'cuda': init_opencv_cuda()
        }
        
        # Windows-specific DirectML
        if IS_WINDOWS:
            capabilities['optimizations']['directml'] = init_directml()
        
        logging.info(f"Platform optimizations initialized: {capabilities}")
        return capabilities

    def check_dark_mode_support():
        """Check if Windows 10 dark mode is supported (Windows 10 1809+)"""
        if not IS_WINDOWS:
            return False
        
        try:
            version = platform.version()
            build = int(version.split('.')[2])
            return build >= 17763  # Windows 10 1809 or higher
        except:
            return False

    def init_memory_priority():
        """Initialize memory priority for Windows"""
        if not IS_WINDOWS or not HAS_WIN32:
            return False
        
        try:
            import win32process
            import win32api
            
            # Set process memory priority to above normal
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32process.PROCESS_SET_INFORMATION, False, pid)
            win32process.SetPriorityClass(handle, win32process.ABOVE_NORMAL_PRIORITY_CLASS)
            win32api.CloseHandle(handle)
            
            logging.info("Windows memory priority set to ABOVE_NORMAL")
            return True
        except Exception as e:
            logging.warning(f"Could not set memory priority: {e}")
            return False

    return init_platform_optimizations() 