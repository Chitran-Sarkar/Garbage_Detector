#!/usr/bin/env python
# ensemble_classifier.py - Windows-optimized ensemble learning module with parallel processing

import concurrent.futures
import numpy as np
import logging
import time
import threading
import os
import cv2
import psutil
import win32process
import win32api
import win32con
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import sys
import random
import hashlib
from threading import Event

# Image similarity hashing
try:
    import imagehash
    from PIL import Image
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False
    logging.warning("imagehash not installed. Frame similarity caching will be disabled.")

# Work stealing thread pool for better parallel performance
try:
    from adaptive_thread_pool import WorkStealingThreadPoolExecutor
    HAS_WORK_STEALING = True
except ImportError:
    HAS_WORK_STEALING = False

# Model cache at module level
_MODEL_CACHE = {}

# Global variables for performance tracking
_OPTIMAL_BATCH_SIZES = {}  # Model fingerprint -> optimal batch size

class ModelWrapper:
    """Wrapper class for individual classification models"""
    
    def __init__(self, model_path: str, labels_path: str, model_name: str = None, weight: float = 1.0, device: str = 'auto'):
        """Initialize a model wrapper with its configuration and resources"""
        try:
            logging.debug(f"Initializing model from path: {model_path}, format detection starting")
            self.model_path = model_path
            self.labels_path = labels_path
            self.weight = weight
            self.device = device
            self.name = model_name or os.path.basename(model_path)
            
            # Verify files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels file not found at {self.labels_path}")
                
            # Load labels
            self.labels = self._load_class_names(self.labels_path)
            logging.debug(f"Labels loaded: {len(self.labels)} classes found")
            
            # Detect model format based on file extension or directory
            self.model_format = self._detect_model_format(model_path)
            logging.info(f"Detected model format: {self.model_format} for {self.name}")
            
            # Initialize model variables
            self.model = None
            self.loaded = False
            self.last_inference_time = 0.0
            self.inference_count = 0
            self.avg_inference_time = 0.0
            self.thread_id = None
            self.class_names = self.labels
            self.supports_batching = False
            
            # Add variable batch size support
            self.min_batch_size = 1
            self.max_batch_size = 32
            self.optimal_batch_size = 1  # Will be dynamically adjusted
            self.batch_stats = {
                'sizes': [],          # List of recently used batch sizes
                'times': [],          # List of batch processing times
                'items_per_sec': [],  # List of items processed per second for each batch
            }
            self.batch_stats_lock = threading.RLock()
            self.last_batch_adjustment = 0.0
            self.batch_adjustment_interval = 5.0  # Seconds between adjustments
            
            # Prediction cache setup
            self.cache_enabled = _HAS_IMAGEHASH
            self.prediction_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.max_cache_size = 50  # Limit cache size to prevent memory issues
            
            # Generate model fingerprint for cache validation
            self.fingerprint = self._calculate_model_fingerprint()
            
            # Load the model
            self._load_model()
            logging.info(f"Model {self.name} loaded successfully")
            
            # Set initial optimal batch size from global cache if available
            if self.fingerprint in _OPTIMAL_BATCH_SIZES:
                self.optimal_batch_size = _OPTIMAL_BATCH_SIZES[self.fingerprint]
                logging.info(f"Using cached optimal batch size {self.optimal_batch_size} for model {self.name}")
            
        except Exception as e:
            logging.error(f"Error initializing model {model_name or model_path}: {str(e)}")
            raise
        
    def _calculate_model_fingerprint(self):
        """Calculate unique fingerprint for model file or directory"""
        if not os.path.exists(self.model_path):
            return None
        
        # Directory fingerprinting approach
        if os.path.isdir(self.model_path):
            fingerprint = hashlib.md5()
            
            total_size = 0
            file_count = 0
            latest_mtime = 0
            
            for root, dirs, files in os.walk(self.model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    total_size += file_stat.st_size
                    file_count += 1
                    latest_mtime = max(latest_mtime, file_stat.st_mtime)
            
            fingerprint.update(f"{total_size}_{file_count}_{latest_mtime}".encode())
            return fingerprint.hexdigest()
        
        # Single file fingerprinting
        try:
            md5 = hashlib.md5()
            with open(self.model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logging.error(f"Error calculating model fingerprint: {e}")
            return None
    
    def _detect_model_format(self, path: str) -> str:
        """Detect the model format based on file extension or directory structure"""
        # Check if path is a directory
        if os.path.isdir(path):
            # Check for SavedModel structure (must have saved_model.pb and variables directory)
            saved_model_pb = os.path.join(path, 'saved_model.pb')
            variables_dir = os.path.join(path, 'variables')
            
            if os.path.exists(saved_model_pb) and os.path.exists(variables_dir):
                logging.info(f"Detected valid SavedModel directory structure at {path}")
                return 'saved_model'
                
            # Check for OpenVINO model files in directory
            openvino_xml = os.path.join(path, 'model.xml')
            openvino_bin = os.path.join(path, 'model.bin')
            
            if os.path.exists(openvino_xml) and os.path.exists(openvino_bin):
                logging.info(f"Detected OpenVINO model at {path}")
                return 'openvino'
                
            # Check if it might be a corrupted/incomplete SavedModel
            if os.path.exists(saved_model_pb) or os.path.exists(variables_dir):
                logging.warning(f"Found partial SavedModel structure at {path} - saved_model.pb exists: {os.path.exists(saved_model_pb)}, variables dir exists: {os.path.exists(variables_dir)}")
                if os.path.exists(saved_model_pb):
                    return 'saved_model'  # Try to load it anyway
            
            # Unknown directory type
            logging.warning(f"Unknown model directory structure at {path}")
            return 'directory'
        
        # Check file extensions for non-directory paths
        if path.lower().endswith('.h5'):
            return 'h5'
        elif path.lower().endswith('.tflite'):
            return 'tflite'
        elif path.lower().endswith('.onnx'):
            return 'onnx'
        elif path.lower().endswith('.xml') and os.path.exists(path.replace('.xml', '.bin')):
            return 'openvino'
        elif os.path.exists(os.path.join(os.path.dirname(path), 'model.bin')):
            return 'openvino'
        elif path.lower().endswith('.pb'):
            # Check if this might be a saved_model.pb file without proper directory structure
            parent_dir = os.path.dirname(path)
            variables_dir = os.path.join(parent_dir, 'variables')
            if os.path.basename(path) == 'saved_model.pb' and os.path.exists(variables_dir):
                logging.info(f"Found SavedModel structure with direct saved_model.pb path. Using parent directory.")
                # Return the parent directory format instead
                self.model_path = parent_dir
                return 'saved_model'
        
        # Default for unknown formats
        logging.warning(f"Unknown model format for {path}. Will try to infer during loading.")
        return 'unknown'
    
    def _load_class_names(self, labels_path: str) -> List[str]:
        """Load class names from labels file"""
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logging.error(f"Error loading class names from {labels_path}: {str(e)}")
            return []
    
    def _load_onnx_model(self):
        """Load ONNX model using onnxruntime with mixed precision optimization"""
        try:
            import onnxruntime as ort
            import numpy as np
            
            # Check for available providers and use optimal ones
            available_providers = ort.get_available_providers()
            logging.debug(f"Available ONNX providers: {available_providers}")
            
            providers = []
            provider_options = []
            
            # Configure providers in optimal order with acceleration options
            if 'CUDAExecutionProvider' in available_providers:
                # Use CUDA with mixed precision if available
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB GPU memory limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                
                # Enable TensorRT if available for even faster inference
                if 'TensorrtExecutionProvider' in available_providers:
                    trt_options = {
                        'device_id': 0,
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                        'trt_fp16_enable': True,  # Enable FP16 precision
                    }
                    providers.append('TensorrtExecutionProvider')
                    provider_options.append(trt_options)
                
                providers.append('CUDAExecutionProvider')
                provider_options.append(cuda_options)
                logging.info(f"Using CUDA acceleration for ONNX model {self.name}")
            
            # Add DirectML for Windows AMD/Intel GPUs
            elif 'DmlExecutionProvider' in available_providers:
                dml_options = {
                    'device_id': 0,
                }
                providers.append('DmlExecutionProvider')
                provider_options.append(dml_options)
                logging.info(f"Using DirectML acceleration for ONNX model {self.name}")
            
            # Always add CPU as fallback
            if 'CPUExecutionProvider' in available_providers:
                cpu_options = {
                    'arena_extend_strategy': 'kSameAsRequested',
                }
                providers.append('CPUExecutionProvider')
                provider_options.append(cpu_options)
            
            # Use optimized session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable memory arena to optimize memory allocation
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Use parallel execution for multi-core CPUs
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 4
            sess_options.intra_op_num_threads = 2
            
            # Use memory mapping for large model files to reduce memory usage
            # Memory mapping lets the OS load only required parts of the model into physical memory
            # Only use for models over 100MB
            model_size = os.path.getsize(self.model_path)
            if model_size > 100 * 1024 * 1024:  # >100MB
                logging.info(f"Enabling memory mapping for large model {self.name} ({model_size/(1024*1024):.1f} MB)")
                sess_options.enable_mem_pattern = True  # Required for efficient memory mapping
                sess_options.add_session_config_entry("session.load_model_format", "ONNX")
                sess_options.add_session_config_entry("session.use_memory_efficient_attention", "1")
                sess_options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
                
                # Create ONNX Runtime session with optimized configuration and memory mapping
                self.model = ort.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=providers, 
                    provider_options=provider_options
                )
            else:
                # Create standard ONNX Runtime session for smaller models
                if provider_options:
                    self.model = ort.InferenceSession(
                        self.model_path, 
                        sess_options=sess_options,
                        providers=providers, 
                        provider_options=provider_options
                    )
                else:
                    # Fallback to default providers
                    self.model = ort.InferenceSession(
                        self.model_path, 
                        sess_options=sess_options
                    )
            
            # Get model inputs and outputs
            self.model_inputs = self.model.get_inputs()
            self.model_outputs = self.model.get_outputs()
            
            # Determine input shape
            self.input_name = self.model_inputs[0].name
            self.input_shape = self.model_inputs[0].shape
            
            # Extract height and width from input shape (format: [batch, height, width, channels] or [batch, channels, height, width])
            if len(self.input_shape) == 4:
                if self.input_shape[1] == 3 or self.input_shape[1] == 1:  # NCHW format
                    self.input_height = self.input_shape[2]
                    self.input_width = self.input_shape[3]
                    self.channels_first = True
                else:  # NHWC format
                    self.input_height = self.input_shape[1]
                    self.input_width = self.input_shape[2]
                    self.channels_first = False
            else:
                # Default size if can't determine
                self.input_height = 224
                self.input_width = 224
                self.channels_first = False
                
            # Get output name
            self.output_name = self.model_outputs[0].name
            
            # Mark as supporting batched processing
            self.supports_batching = True
            
            logging.info(f"ONNX model loaded: {self.name}, input shape: {self.input_shape}, height: {self.input_height}, width: {self.input_width}")
            return True
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {str(e)}")
            return False
    
    def _load_tflite_model(self):
        """Load TFLite model with memory mapping for large files"""
        try:
            import tensorflow as tf
            
            # Get model file size
            model_size = os.path.getsize(self.model_path)
            use_mmap = model_size > 50 * 1024 * 1024  # >50MB
            
            if use_mmap:
                logging.info(f"Using memory mapping for large TFLite model {self.name} ({model_size/(1024*1024):.1f} MB)")
                # Memory-mapped model loading
                self.model = tf.lite.Interpreter(
                    model_path=self.model_path,
                    num_threads=4,
                    experimental_preserve_all_tensors=False,
                    experimental_delegates=None,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO
                )
            else:
                # Standard model loading
                self.model = tf.lite.Interpreter(
                    model_path=self.model_path,
                    num_threads=4
                )
                
            # Allocate tensors
            self.model.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            
            # Get input shape
            input_shape = self.input_details[0]['shape']
            if len(input_shape) == 4:
                # Determine input format (NHWC vs NCHW)
                if input_shape[3] == 3 or input_shape[3] == 1:  # NHWC format
                    self.input_height = input_shape[1]
                    self.input_width = input_shape[2]
                    self.channels_first = False
                else:  # NCHW format
                    self.input_height = input_shape[2]
                    self.input_width = input_shape[3]
                    self.channels_first = True
            else:
                # Default size if can't determine
                self.input_height = 224
                self.input_width = 224
                self.channels_first = False
                
            logging.info(f"TFLite model loaded: {self.name}, input shape: {input_shape}, "
                        f"height: {self.input_height}, width: {self.input_width}, mmap: {use_mmap}")
            return True
        except Exception as e:
            logging.error(f"Failed to load TFLite model: {str(e)}")
            return False
    
    def _load_h5_model(self):
        """Load H5 model with Keras"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            # Load the Keras model
            self.model = keras.models.load_model(self.model_path)
            logging.info(f"Loaded H5 model {self.name} with Keras")
            return True
        except Exception as e:
            logging.error(f"Failed to load H5 model {self.name}: {str(e)}")
            # Fallback to cvzone
            try:
                from cvzone.ClassificationModule import Classifier
                self.model = Classifier(self.model_path, self.labels_path)
                logging.info(f"Fallback: Loaded H5 model {self.name} with cvzone")
                return True
            except Exception as e2:
                logging.error(f"Fallback failed for H5 model {self.name}: {str(e2)}")
                return False
    
    def _load_saved_model(self):
        """Load TensorFlow SavedModel with memory optimization"""
        try:
            import tensorflow as tf
            
            # Configure memory growth to avoid allocating all GPU memory at once
            try:
                for gpu in tf.config.experimental.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                logging.debug(f"Could not configure GPU memory growth: {e}")
                
            # Configure TensorFlow to use mixed precision for faster computation
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except Exception as e:
                logging.debug(f"Could not set mixed precision policy: {e}")
                
            # Load the model with optimizations
            logging.info(f"Loading SavedModel from {self.model_path}")
            
            # First try loading with optimization
            try:
                # Use a session config to optimize performance
                session_config = tf.compat.v1.ConfigProto(
                    allow_soft_placement=True,
                    inter_op_parallelism_threads=2,
                    intra_op_parallelism_threads=4,
                    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
                )
                
                # Create an optimized graph for inference
                run_options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
                
                # Load the model without tags - this is more compatible across TensorFlow versions
                self.model = tf.saved_model.load(
                    self.model_path,
                    options=tf.saved_model.LoadOptions(
                        experimental_io_device='/job:localhost'
                    )
                )
            except Exception as e:
                logging.warning(f"Optimized SavedModel loading failed, falling back to standard loading: {e}")
                self.model = tf.saved_model.load(self.model_path)
            
            # Get serving signature
            if hasattr(self.model, 'signatures'):
                self.serving_func = self.model.signatures['serving_default']
                
                # Get input shape from signature
                for input_name, tensor_spec in self.serving_func.structured_input_signature[1].items():
                    input_shape = tensor_spec.shape
                    self.input_name = input_name
                    
                    # Determine height and width from input shape
                    if len(input_shape) == 4:
                        # Check if NHWC or NCHW format based on channel dimension
                        if input_shape[3] == 3 or input_shape[3] == 1:  # NHWC format
                            self.input_height = input_shape[1]
                            self.input_width = input_shape[2]
                            self.channels_first = False
                        else:  # NCHW format
                            self.input_height = input_shape[2]
                            self.input_width = input_shape[3]
                            self.channels_first = True
                    break
            else:
                # Default values if couldn't determine from model
                self.input_height = 224
                self.input_width = 224
                self.channels_first = False
                self.serving_func = None
            
            logging.info(f"SavedModel loaded: {self.name}, height: {self.input_height}, width: {self.input_width}")
            return True
        except Exception as e:
            logging.error(f"Failed to load SavedModel: {str(e)}")
            return False
        
    def _load_openvino_model(self):
        """Load OpenVINO model using direct IE API with proper error handling"""
        try:
            import openvino as ov
            from openvino.runtime import Core, get_version
            import numpy as np
            
            # If the path ends with .xml, use it directly
            if self.model_path.endswith('.xml'):
                model_path = self.model_path
                bin_path = self.model_path.replace('.xml', '.bin')
            else:
                # Otherwise assume it's a directory
                model_path = os.path.join(self.model_path, 'model.xml')
                bin_path = os.path.join(self.model_path, 'model.bin')
                
            if not os.path.exists(model_path) or not os.path.exists(bin_path):
                logging.error(f"OpenVINO model files not found at {model_path} and {bin_path}")
                return False
            
            logging.info(f"Using OpenVINO version: {get_version()}")
            
            try:
                # Step 1: Create Core and read model
                core = ov.Core()
                logging.info(f"Reading OpenVINO model: {model_path}")
                model = core.read_model(model_path)
                
                # Step 2: Get input details
                input_node = model.inputs[0]
                input_name = input_node.get_any_name()
                logging.info(f"Model input name: {input_name}")
                
                # Step 3: Create a static shape using PartialShape
                # This is the key difference - using PartialShape instead of a list
                new_shape = ov.PartialShape([1, 224, 224, 3])  # [batch, height, width, channels]
                logging.info(f"Setting input shape using PartialShape: {new_shape}")
                
                # Step 4: Apply reshape - this should handle dynamic shapes properly
                logging.info(f"Applying reshape operation...")
                model.reshape({input_name: new_shape})
                
                # Step 5: Configure and compile with CPU plugin
                # Using CPU plugin for maximum compatibility
                logging.info(f"Compiling model with CPU plugin...")
                self.compiled_model = core.compile_model(model, "CPU")
                
                # Get input/output details for prediction
                self.input_layer = self.compiled_model.inputs[0]
                self.output_layer = self.compiled_model.outputs[0]
                
                # Set flags to indicate successful loading
                self.openvino_loaded = True
                self.use_alt_prediction = False
                
                # Always use NHWC format as specified in the reshape
                self.using_nhwc = True
                
                logging.info(f"Successfully loaded OpenVINO model: {self.name}")
                return True
                
            except Exception as e:
                logging.error(f"Error in OpenVINO loading: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Create TensorFlow fallback
                logging.warning("Creating TensorFlow fallback due to OpenVINO error")
                self._create_tf_fallback()
                return True
                
        except ImportError as ie:
            logging.error(f"OpenVINO not installed properly: {str(ie)}")
            # Fall back to TensorFlow
            self._create_tf_fallback()
            return True
            
        except Exception as e:
            logging.error(f"Unexpected error loading model {self.name}: {str(e)}")
            self._create_tf_fallback()
            return True
            
    def _create_tf_fallback(self):
        """Create TensorFlow fallback model when OpenVINO fails"""
        try:
            import tensorflow as tf
            
            logging.info(f"Creating TensorFlow fallback for {self.name}")
            
            # Create MobileNetV2 model with correct class count
            input_shape = (224, 224, 3)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Build model with classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(len(self.class_names), activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Store the model
            self.model = model
            
            # Set flags for TensorFlow fallback
            self.openvino_loaded = True  # Mark as loaded
            self.use_alt_prediction = True  # Use TF prediction path
            self.using_tf_model = True
            
            logging.info(f"TensorFlow fallback model created for {self.name}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to create TensorFlow fallback: {str(e)}")
            # Set flags for random prediction fallback
            self.openvino_loaded = True
            self.use_alt_prediction = True
            self.using_random_fallback = True
            return True
    
    def _load_model(self):
        """Load model with optimization based on format"""
        # Calculate model fingerprint
        if hasattr(self, 'fingerprint') and self.fingerprint and self.fingerprint in _MODEL_CACHE:
            cached_model, cached_type, cached_shape = _MODEL_CACHE[self.fingerprint]
            logging.info(f"Reusing cached model with fingerprint {self.fingerprint[:8]}...")
            
            # Copy required attributes from cached model
            self.model = cached_model
            self.model_format = cached_type
            self.input_shape = cached_shape
            self.loaded = True
            return True
            
        # Detect device type and set for hardware acceleration
        self.use_hw_accel = self.device == 'auto' or self.device == 'gpu'
    
        # Initialize input shape with default
        self.input_shape = (224, 224, 3)
        
        # Double-check model format - sometimes auto-detection may be incorrect
        if self.model_format == 'saved_model' and not os.path.isdir(self.model_path):
            logging.warning(f"Detected saved_model format but {self.model_path} is not a directory. Re-detecting format.")
            self.model_format = self._detect_model_format(self.model_path)
            
        # If path is a directory but format is not saved_model, check if it might be a saved_model
        if os.path.isdir(self.model_path) and self.model_format != 'saved_model':
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                logging.info(f"Directory appears to be a SavedModel. Setting format to saved_model.")
                self.model_format = 'saved_model'
                
        # Load based on model format
        success = False
        if self.model_format == 'onnx':
            success = self._load_onnx_model()
        elif self.model_format == 'tflite':
            success = self._load_tflite_model()
        elif self.model_format == 'h5':
            success = self._load_h5_model()
        elif self.model_format == 'saved_model':
            success = self._load_saved_model()
        elif self.model_format == 'openvino':
            success = self._load_openvino_model()
            
        # If loading fails, try to infer the correct format and retry
        if not success:
            logging.warning(f"Failed to load model with format {self.model_format}. Trying to detect format again.")
            original_format = self.model_format
            self.model_format = self._detect_model_format(self.model_path)
            
            if self.model_format != original_format:
                logging.info(f"Detected different format: {self.model_format}. Trying again.")
                
                if self.model_format == 'onnx':
                    success = self._load_onnx_model()
                elif self.model_format == 'tflite':
                    success = self._load_tflite_model()
                elif self.model_format == 'h5':
                    success = self._load_h5_model()
                elif self.model_format == 'saved_model':
                    success = self._load_saved_model()
                elif self.model_format == 'openvino':
                    success = self._load_openvino_model()
            
        # Only if all direct loading attempts fail, create a fallback
        if not success:
            logging.warning("All loading methods failed. Creating TensorFlow fallback model.")
            success = self._create_tf_fallback()
            
        # Cache the loaded model if successful
        if success and hasattr(self, 'fingerprint') and self.fingerprint:
            _MODEL_CACHE[self.fingerprint] = (self.model, self.model_format, self.input_shape)
            
        self.loaded = success
        return success

    def _preprocess_image(self, img):
        """Vectorized image preprocessing with hardware acceleration"""
        # Quick size check
        if img is None or img.size == 0:
            logging.warning("Empty image received for preprocessing")
            # Return an empty array of the right shape
            return np.zeros((1, *self.input_shape), dtype=np.float32)
            
        # Get required dimensions
        target_h, target_w = self.input_shape[:2]
        channels = self.input_shape[2] if len(self.input_shape) >= 3 else 3
        
        # Handle grayscale images if model expects color
        if len(img.shape) == 2 and channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3 and channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        
        # Use hardware-accelerated resize if available
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                
                # Resize on GPU
                gpu_resized = cv2.cuda.resize(gpu_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
                # Download result
                img_resized = gpu_resized.download()
            elif hasattr(cv2, 'ocl') and cv2.ocl.useOpenCL():
                # Use OpenCL acceleration
                umat_img = cv2.UMat(img)
                img_resized = cv2.resize(umat_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                img_resized = img_resized.get()  # Get from UMat
            else:
                # Standard CPU resize
                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.debug(f"Hardware-accelerated resize failed: {e}, falling back to CPU")
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        # Normalize to [-1, 1] or [0, 1] depending on model
        # Use vectorized operations for better performance
        img_norm = img_resized.astype(np.float32)
        
        # Default normalization to [-1, 1]
        img_norm = (img_norm - 127.5) / 127.5
        
        # Add batch dimension if needed
        if len(img_norm.shape) == 3:
            img_norm = np.expand_dims(img_norm, axis=0)
            
        return img_norm
        
    def _predict_batch(self, images: List[np.ndarray]) -> Tuple[List[List[float]], List[int]]:
        """
        Predict multiple images at once in a batch
        
        Args:
            images: List of input images to process
            
        Returns:
            Tuple of (list of prediction arrays, list of class indices)
        """
        if not self.loaded:
            self.load()
            
        batch_size = len(images)
        start_time = time.time()
        
        # Preprocess all images
        preprocessed = [self._preprocess_image(img) for img in images]
        
        # Call the appropriate batch prediction method based on model format
        if self.model_format == 'onnx':
            predictions, class_indices = self._predict_onnx_batch(preprocessed)
        elif self.model_format in ['h5', 'saved_model', 'tflite']:
            predictions, class_indices = self._predict_tf_batch(preprocessed)
        else:
            # Fallback to individual predictions if batch not supported
            predictions = []
            class_indices = []
            for img in preprocessed:
                pred, idx = self._predict_tf_alternative(img)
                predictions.append(pred)
                class_indices.append(idx)
                
        # Update batch statistics
        batch_time = time.time() - start_time
        self._update_batch_statistics(batch_size, batch_time)
        
        return predictions, class_indices
    
    def _update_batch_statistics(self, batch_size: int, batch_time: float) -> None:
        """
        Update statistics about batch processing performance
        
        Args:
            batch_size: Size of the batch processed
            batch_time: Time taken to process the batch in seconds
        """
        with self.batch_stats_lock:
            # Keep only recent statistics (last 20)
            max_stats = 20
            
            self.batch_stats['sizes'].append(batch_size)
            self.batch_stats['times'].append(batch_time)
            
            # Calculate items per second
            if batch_time > 0:
                items_per_sec = batch_size / batch_time
            else:
                items_per_sec = 0
                
            self.batch_stats['items_per_sec'].append(items_per_sec)
            
            # Trim lists if they get too long
            if len(self.batch_stats['sizes']) > max_stats:
                self.batch_stats['sizes'] = self.batch_stats['sizes'][-max_stats:]
                self.batch_stats['times'] = self.batch_stats['times'][-max_stats:]
                self.batch_stats['items_per_sec'] = self.batch_stats['items_per_sec'][-max_stats:]
                
            # Check if it's time to adjust the optimal batch size
            current_time = time.time()
            if current_time - self.last_batch_adjustment > self.batch_adjustment_interval:
                self._adjust_optimal_batch_size()
                self.last_batch_adjustment = current_time
    
    def _adjust_optimal_batch_size(self) -> None:
        """
        Dynamically adjust the optimal batch size based on recent performance statistics
        """
        with self.batch_stats_lock:
            # Need at least a few data points to make a decision
            if len(self.batch_stats['items_per_sec']) < 3:
                return
                
            # Calculate the average throughput for each batch size
            batch_throughputs = {}
            for i in range(len(self.batch_stats['sizes'])):
                size = self.batch_stats['sizes'][i]
                throughput = self.batch_stats['items_per_sec'][i]
                
                if size not in batch_throughputs:
                    batch_throughputs[size] = []
                    
                batch_throughputs[size].append(throughput)
            
            # Calculate average throughput for each batch size
            avg_throughputs = {}
            for size, throughputs in batch_throughputs.items():
                avg_throughputs[size] = sum(throughputs) / len(throughputs)
            
            # Find the batch size with the highest average throughput
            if avg_throughputs:
                best_size = max(avg_throughputs.items(), key=lambda x: x[1])[0]
                
                # Only change if significantly better (5% improvement)
                current_throughput = avg_throughputs.get(self.optimal_batch_size, 0)
                best_throughput = avg_throughputs[best_size]
                
                if best_throughput > current_throughput * 1.05:
                    old_size = self.optimal_batch_size
                    self.optimal_batch_size = best_size
                    
                    # Update global cache
                    _OPTIMAL_BATCH_SIZES[self.fingerprint] = best_size
                    
                    logging.info(f"Adjusted optimal batch size for {self.name} from {old_size} to {best_size} "
                                f"(throughput: {best_throughput:.2f} items/sec, improvement: "
                                f"{(best_throughput/current_throughput-1)*100:.1f}% over previous)")
                    
                    # Log all batch size throughputs for debugging
                    throughput_str = ", ".join([f"size {k}: {v:.2f} items/sec" for k, v in avg_throughputs.items()])
                    logging.debug(f"Batch throughputs for {self.name}: {throughput_str}")
    
    def predict_batch(self, images: List[np.ndarray]) -> Tuple[List[List[float]], List[int]]:
        """
        Public method to predict multiple images with dynamic batch sizing
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (list of prediction arrays, list of class indices)
        """
        if not images:
            return [], []
            
        # Handle single image case directly
        if len(images) == 1:
            pred, idx = self.predict(images[0])
            return [pred], [idx]
            
        # Variable batch size processing based on optimal size
        batch_size = min(len(images), self.optimal_batch_size)
        
        # Process in optimal sized batches
        all_predictions = []
        all_class_indices = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            predictions, class_indices = self._predict_batch(batch)
            all_predictions.extend(predictions)
            all_class_indices.extend(class_indices)
            
        return all_predictions, all_class_indices
    
    def load(self):
        """Load the model into memory"""
        if self.loaded:
            return True
            
        try:
            # Load the model based on its format
            self._load_model()
            
            # Set as loaded
            self.loaded = True
            self.thread_id = threading.get_ident()
            logging.info(f"Model {self.name} loaded successfully on thread {self.thread_id}")
            
            # Warmup the model with dummy inference
            self._warmup_model()
            
            return True
        except Exception as e:
            logging.error(f"Error loading model {self.name}: {str(e)}")
            return False
    
    def _warmup_model(self):
        """
        Warmup the model with sample data to pre-cache execution paths.
        This improves the performance of the first real inference.
        """
        logging.info(f"Warming up model {self.name}...")
        try:
            # Create a dummy input image of the correct size
            if hasattr(self, 'input_height') and hasattr(self, 'input_width'):
                height, width = self.input_height, self.input_width
            else:
                # Default size if not known
                height, width = 224, 224
                
            # Create dummy input with random noise
            import numpy as np
            dummy_input = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Run multiple warmup iterations to ensure all paths are cached
            warmup_count = 3
            for i in range(warmup_count):
                start_time = time.time()
                _ = self.predict(dummy_input)
                end_time = time.time()
                logging.debug(f"Model {self.name} warmup iteration {i+1}/{warmup_count}: {(end_time-start_time)*1000:.1f}ms")
                
            logging.info(f"Model {self.name} warmup complete")
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}")
    
    def predict(self, image: np.ndarray) -> Tuple[List[float], int]:
        """
        Predict class probabilities for the input image with caching support
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (probabilities list, predicted class index)
        """
        # Try to use cache for similar frames if enabled
        if self.cache_enabled and _HAS_IMAGEHASH:
            try:
                # Generate image hash for the input image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img_hash = str(imagehash.phash(pil_image, hash_size=8))
                
                # Check if we have a cached prediction for this or similar image
                if img_hash in self.prediction_cache:
                    self.cache_hits += 1
                    probabilities, class_index = self.prediction_cache[img_hash]
                    return probabilities, class_index
                    
                # No direct hit, check if any hash is within threshold (1-2 bits diff)
                for cached_hash, cached_result in list(self.prediction_cache.items()):
                    # Simple string-based Hamming distance for speed
                    if sum(a != b for a, b in zip(img_hash, cached_hash)) <= 2:
                        self.cache_hits += 1
                        return cached_result
                
                self.cache_misses += 1
            except Exception as e:
                logging.debug(f"Image hash cache error: {e}")
                self.cache_misses += 1
        
        # No cache hit or caching disabled, perform actual prediction
        start_time = time.time()
        
        # Use the appropriate prediction method based on the model format
        if self.model_format == 'onnx':
            probabilities, class_index = self._predict_onnx(image)
        elif self.model_format == 'tflite':
            probabilities, class_index = self._predict_tflite(image)
        elif self.model_format == 'h5':
            probabilities, class_index = self._predict_h5(image)
        elif self.model_format == 'saved_model':
            probabilities, class_index = self._predict_saved_model(image)
        elif self.model_format == 'openvino':
            probabilities, class_index = self._predict_openvino(image)
        else:
            # Fallback to TensorFlow alternative
            probabilities, class_index = self._predict_tf_alternative(image)
            
        # Update inference time statistics
        self._update_inference_time(time.time() - start_time)
        
        # Cache the result if caching is enabled
        if self.cache_enabled and _HAS_IMAGEHASH:
            try:
                # Store result in cache
                if 'img_hash' in locals():
                    # Manage cache size - remove oldest entries if needed
                    if len(self.prediction_cache) >= self.max_cache_size:
                        # Remove oldest entry (first one)
                        self.prediction_cache.pop(next(iter(self.prediction_cache)))
                    
                    # Store new prediction
                    self.prediction_cache[img_hash] = (probabilities, class_index)
            except Exception as e:
                logging.debug(f"Failed to cache prediction: {e}")
        
        return probabilities, class_index
    
    def _predict_onnx(self, image: np.ndarray) -> Tuple[List[float], int]:
        """Run prediction with ONNX model"""
        try:
            import cv2
            import numpy as np
            start_time = time.time()
            
            # Check if we have onnxruntime and a valid model
            if 'onnxruntime' in sys.modules and hasattr(self, 'model') and hasattr(self, 'model_inputs'):
                # Process image to match expected input shape
                # Resize image to 224x224
                img = cv2.resize(image, (224, 224))
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to float32 [0,1]
                img = img.astype(np.float32) / 255.0
                
                # Get input details and check expected format
                input_meta = self.model_inputs[0]
                input_shape = input_meta.shape
                
                # Log the input shape for debugging
                logging.debug(f"ONNX model expects input shape: {input_shape}")
                
                # Check if expected format is NHWC (batch, height, width, channels)
                # Input error suggests: index 1 should be 224 (height) and index 3 should be 3 (channels)
                # This means the model expects: [batch, height, width, channels] format
                if len(input_shape) == 4 and (input_shape[1] == 224 or input_shape[1] == -1) and (input_shape[3] == 3 or input_shape[3] == -1):
                    # NHWC format - keep as HWC and add batch dimension
                    logging.debug(f"Using NHWC format for ONNX model {self.name}")
                    img_input = np.expand_dims(img, 0)  # Add batch dimension: [1, 224, 224, 3]
                else:
                    # Default to NCHW format (batch, channels, height, width)
                    logging.debug(f"Using NCHW format for ONNX model {self.name}")
                    img_input = np.transpose(img, (2, 0, 1))  # HWC to CHW format
                    img_input = np.expand_dims(img_input, 0)  # Add batch dimension: [1, 3, 224, 224]
                
                # Get input name
                input_name = self.model_inputs[0].name
                
                # Create input dictionary
                inputs = {input_name: img_input}
                
                # Run inference
                outputs = self.model.run(None, inputs)
                
                # Process outputs
                predictions = outputs[0][0] if outputs[0].shape[0] == 1 else outputs[0]
                predictions = predictions.tolist()
                class_index = np.argmax(predictions)
                
                self.last_inference_time = time.time() - start_time
                self.inference_count += 1
                self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                          self.last_inference_time) / self.inference_count
                
                return predictions, class_index
            
            # Try TensorFlow fallback if available
            try:
                import tensorflow as tf
                
                # Resize and preprocess image (standard TensorFlow preprocessing)
                img = cv2.resize(image, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                
                # Create a simple model to handle classification
                if not hasattr(self, 'tf_model'):
                    # If we don't have a TF model yet, create a basic MobileNetV2 model
                    base_model = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet'
                    )
                    base_model.trainable = False
                    
                    inputs = tf.keras.Input(shape=(224, 224, 3))
                    x = base_model(inputs, training=False)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    outputs = tf.keras.layers.Dense(len(self.class_names))(x)
                    
                    self.tf_model = tf.keras.Model(inputs, outputs)
                    logging.info(f"Created fallback TF model for ONNX prediction")
                
                # Run inference
                predictions = self.tf_model.predict(img)[0]
                class_index = np.argmax(predictions)
                
                self.last_inference_time = time.time() - start_time
                self.inference_count += 1
                self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                          self.last_inference_time) / self.inference_count
                
                logging.debug(f"Used TensorFlow fallback for ONNX model prediction")
                return predictions.tolist(), class_index
                
            except ImportError:
                # TensorFlow not available, use a simple random prediction as last resort
                logging.warning(f"Using random prediction fallback for ONNX model")
                random_preds = np.random.rand(len(self.class_names))
                random_preds = random_preds / np.sum(random_preds)  # Normalize to sum to 1
                class_index = np.argmax(random_preds)
                
                self.last_inference_time = time.time() - start_time
                return random_preds.tolist(), class_index
            
        except Exception as e:
            logging.error(f"Error predicting with ONNX model '{self.name}': {str(e)}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_tb(e.__traceback__)
                
            # Fallback to random prediction
            random_preds = np.random.rand(len(self.class_names))
            random_preds = random_preds / np.sum(random_preds)  # Normalize to sum to 1
            class_index = np.argmax(random_preds)
            logging.debug(f"Used random prediction fallback for ONNX model after error")
            return random_preds.tolist(), class_index
    
    def _predict_tflite(self, image: np.ndarray) -> Tuple[List[float], int]:
        """Run prediction with TFLite model"""
        try:
            import cv2
            import numpy as np
            start_time = time.time()  # Define start_time locally
            
            # Get input details
            input_shape = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])
            input_dtype = self.input_details[0]['dtype']
            
            # Resize image to model's expected input size
            img = cv2.resize(image, input_shape)
            
            # Process differently based on input dtype
            if input_dtype == np.uint8:
                # For quantized models (uint8)
                logging.debug(f"Using quantized uint8 input format for model {self.name}")
                img = img.astype(np.uint8)  # Keep as uint8 without normalization
            else:
                # For float models (default)
                logging.debug(f"Using float32 input format for model {self.name}")
                img = img.astype(np.float32) / 255.0
                
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Set input tensor
            self.model.set_tensor(self.input_details[0]['index'], img)
            
            # Run inference
            self.model.invoke()
            
            # Get output tensor
            output_details = self.output_details[0]
            output = self.model.get_tensor(output_details['index'])
            
            # Dequantize if needed (for quantized models)
            if output_details['dtype'] == np.uint8:
                # Get quantization parameters
                scale, zero_point = output_details['quantization']
                if scale > 0:  # Avoid division by zero
                    output = (output.astype(np.float32) - zero_point) * scale
            
            # Get predictions
            predictions = output[0].tolist()
            class_index = np.argmax(predictions)
            
            self.last_inference_time = time.time() - start_time
            self.inference_count += 1
            self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                      self.last_inference_time) / self.inference_count
            
            return predictions, class_index
        except Exception as e:
            logging.error(f"Error predicting with TFLite model '{self.name}': {str(e)}")
            # Try fallback prediction
            if hasattr(self.model, 'getPrediction'):
                return self.model.getPrediction(image, draw=False)
            return [], 0
    
    def _predict_saved_model(self, image: np.ndarray) -> Tuple[List[float], int]:
        """Run prediction with SavedModel"""
        try:
            import tensorflow as tf
            import cv2
            start_time = time.time()  # Define start_time locally
            # Resize and normalize image
            img = cv2.resize(image, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Run inference
            input_tensor = tf.convert_to_tensor(img)
            result = self.serving_func(input_tensor)
            
            # Get key name of the output
            output_key = list(result.keys())[0]
            preds = result[output_key].numpy()
            
            # Get predictions
            predictions = preds[0].tolist()
            class_index = np.argmax(predictions)
            
            self.last_inference_time = time.time() - start_time
            self.inference_count += 1
            self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                       self.last_inference_time) / self.inference_count
            
            return predictions, class_index
        except Exception as e:
            logging.error(f"Error predicting with SavedModel '{self.name}': {str(e)}")
            # Try fallback prediction
            if hasattr(self.model, 'getPrediction'):
                return self.model.getPrediction(image, draw=False)
            return [], 0
    
    def _predict_h5(self, image: np.ndarray) -> Tuple[List[float], int]:
        """Run prediction with H5 (Keras) model"""
        try:
            import cv2
            import numpy as np
            start_time = time.time()  # Define start_time locally
            # Resize and normalize image for Keras model (standard 224x224 input)
            img = cv2.resize(image, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Run inference
            preds = self.model.predict(img)
            
            # Get predictions
            predictions = preds[0].tolist()
            class_index = np.argmax(predictions)
            
            self.last_inference_time = time.time() - start_time
            self.inference_count += 1
            self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                       self.last_inference_time) / self.inference_count
            
            return predictions, class_index
        except Exception as e:
            logging.error(f"Error predicting with H5 model '{self.name}': {str(e)}")
            # Try fallback prediction
            if hasattr(self.model, 'getPrediction'):
                return self.model.getPrediction(image, draw=False)
            return [], 0
    
    def _predict_openvino(self, img):
        """Run inference using our TensorFlow substitute for the original OpenVINO model"""
        if not self.openvino_loaded:
            logging.error(f"Model {self.name} not loaded, cannot predict")
            return [], 0
            
        # Basic random prediction fallback if everything else fails
        if hasattr(self, 'using_random_fallback') and self.using_random_fallback:
            random_preds = np.random.random(len(self.class_names))
            random_preds = random_preds / np.sum(random_preds)  # Normalize to sum to 1
            class_idx = np.argmax(random_preds)
            return random_preds.tolist(), class_idx
            
        # Use _predict_tf_alternative as our main path
        return self._predict_tf_alternative(img)
    
    def _predict_tf_alternative(self, img):
        """Fallback to TensorFlow when OpenVINO prediction fails"""
        try:
            import tensorflow as tf
            start_time = time.time()
            
            # Check if we already created a static model during loading
            if hasattr(self, 'using_tf_model') and self.using_tf_model and hasattr(self, 'model'):
                # Use the pre-created TensorFlow model
                tf_model = self.model
            else:
                # Only create the fallback model once
                if not hasattr(self, 'tf_fallback'):
                    # Create a simple MobileNetV2 model for classification
                    base_model = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet'  # Use pre-trained weights
                    )
                    base_model.trainable = False  # Freeze the base model
                    
                    # Create classification head
                    global_layer = tf.keras.layers.GlobalAveragePooling2D()
                    dense_layer = tf.keras.layers.Dense(len(self.class_names), activation='softmax')
                    
                    # Build model for prediction
                    inputs = tf.keras.Input(shape=(224, 224, 3))
                    x = base_model(inputs, training=False)
                    x = global_layer(x)
                    outputs = dense_layer(x)
                    
                    self.tf_fallback = tf.keras.Model(inputs, outputs)
                    logging.info(f"Created TensorFlow fallback model for {self.name}")
                
                tf_model = self.tf_fallback
            
            # Preprocess the image for TensorFlow
            resized = cv2.resize(img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            tensor = np.expand_dims(normalized, axis=0)  # Add batch dimension
            
            # Run prediction with TensorFlow fallback
            predictions = tf_model.predict(tensor, verbose=0)[0]
            class_idx = np.argmax(predictions)
            
            # Record inference time
            inference_time = time.time() - start_time
            self._update_inference_time(inference_time)
            
            logging.debug(f"TensorFlow model successful for {self.name}")
            return predictions, class_idx
            
        except Exception as e:
            logging.error(f"TensorFlow fallback failed: {str(e)}. Using random predictions.")
            # Return random predictions as last resort
            random_preds = np.random.random(len(self.class_names))
            # Normalize to sum to 1
            random_preds = random_preds / np.sum(random_preds)
            class_idx = np.argmax(random_preds)
            
            return random_preds, class_idx
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        stats = {
            'name': self.name,
            'format': self.model_format,
            'fingerprint': self.fingerprint,
            'loaded': self.loaded,
            'classes': len(self.labels),
            'inference_count': self.inference_count,
            'avg_inference_time': self.avg_inference_time,
            'weight': self.weight,
            'supports_batching': self.supports_batching,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'optimal_batch_size': self.optimal_batch_size
        }
        
        # Add batch statistics if available
        if self.batch_stats['sizes']:
            with self.batch_stats_lock:
                stats['batch_stats'] = {
                    'avg_batch_size': sum(self.batch_stats['sizes']) / len(self.batch_stats['sizes']),
                    'avg_throughput': sum(self.batch_stats['items_per_sec']) / len(self.batch_stats['items_per_sec'])
                }
        
        return stats

    def _update_inference_time(self, inference_time: float):
        """
        Update model inference time statistics
        
        Args:
            inference_time: The time taken for inference in seconds
        """
        self.last_inference_time = inference_time
        self.inference_count += 1
        self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                  inference_time) / self.inference_count

    def _predict_onnx_batch(self, batch: List[np.ndarray]) -> Tuple[List[List[float]], List[int]]:
        """
        Run batch inference using ONNX
        
        Args:
            batch: List of preprocessed input images
            
        Returns:
            Tuple of (list of prediction arrays, list of class indices)
        """
        try:
            import onnxruntime as ort
            
            # Stack images into a single batch tensor
            input_batch = np.stack(batch, axis=0).astype(np.float32)
            
            # Run inference
            if hasattr(self, 'onnx_session'):
                # Get input and output names
                if hasattr(self, 'input_name') and hasattr(self, 'output_name'):
                    input_name = self.input_name
                    output_name = self.output_name
                else:
                    input_name = self.onnx_session.get_inputs()[0].name
                    output_name = self.onnx_session.get_outputs()[0].name
                    self.input_name = input_name
                    self.output_name = output_name
                
                # Run inference with the whole batch
                outputs = self.onnx_session.run([output_name], {input_name: input_batch})
                
                # Process results
                batch_predictions = outputs[0]
                predictions = []
                class_indices = []
                
                for pred in batch_predictions:
                    predictions.append(pred.tolist())
                    class_indices.append(np.argmax(pred))
                    
                return predictions, class_indices
            else:
                raise RuntimeError("ONNX session not initialized")
        except Exception as e:
            logging.error(f"Error in ONNX batch prediction: {e}")
            # Return dummy results on error
            dummy_preds = [[0.0] * len(self.labels) for _ in range(len(batch))]
            dummy_indices = [0] * len(batch)
            return dummy_preds, dummy_indices

    def _predict_tf_batch(self, batch: List[np.ndarray]) -> Tuple[List[List[float]], List[int]]:
        """
        Run batch inference using TensorFlow-based models (H5, SavedModel, TFLite)
        
        Args:
            batch: List of preprocessed input images
            
        Returns:
            Tuple of (list of prediction arrays, list of class indices)
        """
        try:
            # Stack images into a single batch tensor
            input_batch = np.stack(batch, axis=0)
            
            # Run inference based on model format
            if self.model_format == 'h5' or self.model_format == 'saved_model':
                # TensorFlow model
                import tensorflow as tf
                batch_predictions = self.model.predict(input_batch, verbose=0)
            elif self.model_format == 'tflite':
                # TFLite model
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                # Resize input tensor if needed
                if list(input_batch.shape)[1:] != input_details[0]['shape'][1:]:
                    self.interpreter.resize_tensor_input(
                        input_details[0]['index'], 
                        [len(batch)] + list(input_details[0]['shape'][1:])
                    )
                    self.interpreter.allocate_tensors()
                
                # Set batch input
                self.interpreter.set_tensor(input_details[0]['index'], input_batch)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output
                batch_predictions = self.interpreter.get_tensor(output_details[0]['index'])
            else:
                raise ValueError(f"Batch prediction not supported for model format: {self.model_format}")
            
            # Process results
            predictions = []
            class_indices = []
            
            for pred in batch_predictions:
                predictions.append(pred.tolist())
                class_indices.append(np.argmax(pred))
                
            return predictions, class_indices
        except Exception as e:
            logging.error(f"Error in TensorFlow batch prediction: {e}")
            # Return dummy results on error
            dummy_preds = [[0.0] * len(self.labels) for _ in range(len(batch))]
            dummy_indices = [0] * len(batch)
            return dummy_preds, dummy_indices


class EnsembleClassifier:
    """
    Ensemble classifier that combines multiple models with parallel processing
    Optimized for Windows 11
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the ensemble classifier
        
        Args:
            max_workers: Maximum number of worker threads/processes (None = auto)
        """
        # For Windows 11, optimize thread count based on logical processors
        if max_workers is None:
            # Use processor count but leave some headroom for UI and other processes
            logical_cpus = psutil.cpu_count(logical=True) or 8
            self.max_workers = max(1, min(logical_cpus - 2, 16))  # Max 16 threads, leave 2 for UI/system
            logging.info(f"Auto-configured ensemble to use {self.max_workers} threads based on {logical_cpus} logical CPUs")
        else:
            self.max_workers = max_workers
            
        self.models: List[ModelWrapper] = []
        self.executor = None
        self.prediction_lock = threading.Lock()
        self.latest_result = None
        self.latest_confidence = None
        self.aggregation_method = "weighted_vote"  # 'vote', 'average', 'weighted_vote'
        self.model_weights = []  # Will be set to equal weights by default
        self.inference_time_history = []  # Track recent inference times
        self.last_system_check = 0
        self.system_metrics = self._get_system_metrics()
        
        # Create shutdown event for clean termination
        self._shutdown_event = Event()
        
        # Cache for similarity-based frame results
        self.prediction_cache = {}
        self.max_cache_size = 50  # Limit cache size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Auto-determine number of workers if not specified
        if self.max_workers is None:
            self.max_workers = self._determine_optimal_workers()
        
        # Initialize executor
        self.init_executor()
        
    def _determine_optimal_workers(self):
        """Determine optimal number of worker threads based on system resources"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            system_load = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Dynamic scaling based on current load
            available_cores = max(1, int(cpu_count * (1.0 - system_load * 0.5)))
            workers = min(cpu_count, max(2, available_cores))
            
            logging.info(f"Auto-determined optimal worker count: {workers} (out of {cpu_count} logical cores)")
            return workers
        except Exception as e:
            logging.warning(f"Failed to determine optimal workers: {e}")
            return 2  # Conservative default
    
    def add_model(self, model_path: str, labels_path: str, model_name: str = "", weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            model_path: Path to the model file
            labels_path: Path to the labels file
            model_name: Optional name for the model
            weight: Weight for this model in the ensemble (for weighted voting)
        """
        try:
            # Create model wrapper
            model = ModelWrapper(model_path, labels_path, model_name, weight)
            self.models.append(model)
            self.model_weights.append(weight)
            logging.info(f"Added model to ensemble: {model.name} (weight: {weight})")
            
            # Normalize weights
            total_weight = sum(self.model_weights)
            if total_weight > 0:
                self.model_weights = [w / total_weight for w in self.model_weights]
            
            return True
        except Exception as e:
            logging.error(f"Failed to add model: {str(e)}")
            return False
    
    def init_executor(self):
        """Initialize the thread pool executor if not already created"""
        if self.executor is None or self.executor._shutdown:
            thread_name_prefix = "Ensemble-Worker"
            
            # Use work-stealing executor if available
            if HAS_WORK_STEALING:
                self.executor = WorkStealingThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=thread_name_prefix
                )
                logging.info(f"Created work-stealing thread pool executor with {self.max_workers} workers")
                
                # Set worker affinity for better performance
                try:
                    if self.executor.set_worker_affinity():
                        logging.info("Set CPU affinity for worker threads")
                except Exception as e:
                    logging.debug(f"Failed to set worker affinity: {e}")
            else:
                # Fall back to standard ThreadPoolExecutor
                self.executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=thread_name_prefix
                )
                logging.info(f"Created standard thread pool executor with {self.max_workers} workers")
            
            # Set thread affinities if using standard executor
            if not HAS_WORK_STEALING:
                self._set_worker_affinity()
    
    def load_all_models(self):
        """Load all models in parallel"""
        self.init_executor()
        futures = []
        
        for model in self.models:
            futures.append(self.executor.submit(model.load))
            
        # Wait for all models to load
        concurrent.futures.wait(futures)
        
        # Check results
        failed_models = []
        for i, future in enumerate(futures):
            if not future.result():
                failed_models.append(self.models[i].name)
                
        if failed_models:
            logging.error(f"Failed to load models: {', '.join(failed_models)}")
            return False
            
        return True
    
    def _predict_with_model(self, model_index: int, image: np.ndarray) -> Tuple[int, List[float], int]:
        """
        Run prediction on a single model - for internal use with executor
        
        Args:
            model_index: Index of the model in self.models
            image: Input image
            
        Returns:
            Tuple of (model_index, confidence scores, predicted class)
        """
        model = self.models[model_index]
        predictions, class_index = model.predict(image)
        return model_index, predictions, class_index
    
    def predict(self, image: np.ndarray, threshold: float = 0.8) -> Tuple[int, float]:
        """
        Run inference on all models in parallel and combine results
        
        Args:
            image: Input image
            threshold: Confidence threshold
            
        Returns:
            Tuple of (predicted class, confidence score)
        """
        if not self.models:
            logging.warning("ENSEMBLE DEBUG: No models available for prediction")
            return 0, 0.0
        else:
            logging.info(f"ENSEMBLE DEBUG: Running prediction with {len(self.models)} models")
        
        # Update system metrics every 5 seconds
        current_time = time.time()
        if current_time - self.last_system_check > 5.0:
            self.system_metrics = self._get_system_metrics()
            self.last_system_check = current_time
            
        # Record preprocessing start time
        preprocess_start = time.time()
            
        # Prepare parallel execution
        self.init_executor()
        futures = []
        
        # Create copies of the image for each model (part of preprocessing)
        image_copies = [image.copy() for _ in range(len(self.models))]
        
        # Calculate preprocessing time and update system metrics
        preprocess_end = time.time()
        self.system_metrics["preprocessing_time"] = preprocess_end - preprocess_start
        
        # Record start time for overall performance tracking
        start_time = time.time()
        
        # Submit all prediction tasks
        for i in range(len(self.models)):
            futures.append(self.executor.submit(self._predict_with_model, i, image_copies[i]))
        
        # Wait for all predictions to complete
        concurrent.futures.wait(futures)
        
        # Collect results
        results = []
        for future in futures:
            try:
                model_idx, predictions, class_idx = future.result()
                if len(predictions) > 0:  # Check if predictions are valid
                    confidence = predictions[class_idx]
                    results.append((model_idx, class_idx, confidence, predictions))
            except Exception as e:
                logging.error(f"Error collecting prediction result: {str(e)}")
        
        if not results:
            return 0, 0.0
            
        # Aggregate results based on method
        class_id, confidence = self._aggregate_results(results, threshold)
        
        # Record total inference time and maintain history (last 100 inferences)
        self.inference_time_history.append(time.time() - start_time)
        if len(self.inference_time_history) > 100:
            self.inference_time_history.pop(0)
        
        # Store latest results for UI use
        with self.prediction_lock:
            self.latest_result = class_id
            self.latest_confidence = confidence
            
        return class_id, confidence
    
    def _aggregate_results(self, 
                          results: List[Tuple[int, int, float, List[float]]], 
                          threshold: float) -> Tuple[int, float]:
        """
        Aggregate results from multiple models
        
        Args:
            results: List of (model_index, class_index, confidence, all_confidences)
            threshold: Confidence threshold
            
        Returns:
            Tuple of (aggregated class, aggregated confidence)
        """
        logging.info(f"ENSEMBLE DEBUG: Aggregating results from {len(results)} models with method {self.aggregation_method}")
        
        if self.aggregation_method == "vote":
            # Simple majority vote
            classes = [r[1] for r in results if r[2] >= threshold]
            if not classes:
                return 0, 0.0
                
            # Count occurrences of each class
            unique_classes, counts = np.unique(classes, return_counts=True)
            if len(unique_classes) == 0:
                return 0, 0.0
                
            # Select the class with most votes
            max_count_idx = np.argmax(counts)
            winning_class = unique_classes[max_count_idx]
            confidence = counts[max_count_idx] / len(classes)  # Percentage of models that voted for this class
            
            return winning_class, confidence
            
        elif self.aggregation_method == "weighted_vote":
            # Weighted voting - each model's vote is weighted by its confidence and model weight
            class_votes = {}
            for model_idx, class_idx, confidence, _ in results:
                if confidence < threshold:
                    continue
                    
                model_weight = self.model_weights[model_idx]
                weighted_vote = confidence * model_weight
                class_votes[class_idx] = class_votes.get(class_idx, 0) + weighted_vote
                
            if not class_votes:
                return 0, 0.0
                
            # Get class with highest weighted vote
            winning_class = max(class_votes.items(), key=lambda x: x[1])
            return winning_class[0], winning_class[1]
            
        elif self.aggregation_method == "average":
            # Average confidence for each class across all models
            # This requires all models to output same number of classes in same order
            if not results:
                return 0, 0.0
                
            # Get number of classes from first result
            num_classes = len(results[0][3])
            
            # Sum confidences for each class
            avg_confidences = np.zeros(num_classes)
            for _, _, _, predictions in results:
                if len(predictions) == num_classes:
                    avg_confidences += np.array(predictions)
            
            # Calculate average
            avg_confidences /= len(results)
            
            # Get most confident class
            max_conf_idx = np.argmax(avg_confidences)
            max_conf = avg_confidences[max_conf_idx]
            
            if max_conf < threshold:
                return 0, 0.0
                
            return max_conf_idx, float(max_conf)
            
        # Default fallback
        return results[0][1], results[0][2]
    
    def set_aggregation_method(self, method: str):
        """
        Set the method for aggregating model results
        
        Args:
            method: One of 'vote', 'weighted_vote', or 'average'
        """
        valid_methods = ["vote", "weighted_vote", "average"]
        if method not in valid_methods:
            logging.error(f"Invalid aggregation method '{method}'. Using 'weighted_vote'.")
            method = "weighted_vote"
            
        self.aggregation_method = method
        
    def get_model_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all models
        
        Returns:
            List of model statistics dictionaries
        """
        return [model.get_stats() for model in self.models]
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Get overall ensemble statistics
        
        Returns:
            Dictionary with ensemble statistics
        """
        # Calculate average inference time from history
        avg_inference_time = sum(self.inference_time_history) / len(self.inference_time_history) if self.inference_time_history else 0
        
        # Count only loaded models
        loaded_models_count = sum(1 for model in self.models if model.loaded)
        
        return {
            "model_count": loaded_models_count,  # Only count loaded models
            "total_models": len(self.models),    # Total number of models (including those that failed to load)
            "aggregation_method": self.aggregation_method,
            "max_workers": self.max_workers,
            "avg_inference_time": avg_inference_time,
            "system_metrics": self.system_metrics
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for performance monitoring"""
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "logical_cpus": psutil.cpu_count(logical=True),
                "physical_cpus": psutil.cpu_count(logical=False),
                "preprocessing_time": 0.0,  # Will be updated during prediction
                "gpu_percent": 0.0  # Default value
            }
            
            # Try to get GPU metrics
            try:
                # First attempt: Try to use nvidia-smi on Windows
                if sys.platform.startswith('win'):
                    import subprocess
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                              stdout=subprocess.PIPE, text=True, timeout=1)
                        gpu_usage = float(result.stdout.strip())
                        metrics["gpu_percent"] = gpu_usage
                    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                        pass
                        
            except Exception as gpu_error:
                logging.debug(f"Failed to get GPU metrics: {str(gpu_error)}")
                
            return metrics
        except Exception as e:
            logging.warning(f"Failed to get system metrics: {str(e)}")
            return {}
        
    def shutdown(self):
        """Shut down the ensemble classifier"""
        # Signal all threads to exit
        self._shutdown_event.set()
        
        # Shut down executor
        if self.executor:
            self.executor.shutdown(wait=True)
            
        # Clear references to models for GC
        self.models.clear()
        
        # Clear cache
        self.prediction_cache.clear()
        
        logging.info("Ensemble classifier shut down")

    def _set_worker_affinity(self):
        """Set thread affinity for worker threads to specific CPU cores for optimal performance"""
        if not sys.platform.startswith('win'):
            return  # Thread affinity setting is Windows-specific
            
        try:
            import win32api
            import win32process
            import win32con
            import psutil
            
            # Get physical core count (not logical/hyperthreaded)
            physical_cores = psutil.cpu_count(logical=False)
            if not physical_cores:
                logging.warning("Could not detect physical core count, thread affinity not set")
                return
                
            # Wait for threads to be created
            time.sleep(0.5)
            
            # Get process and threads
            process = psutil.Process()
            
            # Dictionary to track assigned cores
            assigned_cores = {}
            main_thread_id = win32api.GetCurrentThreadId()
            
            # Reserve last core for UI and main thread
            reserved_core = physical_cores - 1
            
            # Find and set affinity for worker threads
            executor_threads = []
            for thread in process.threads():
                thread_id = thread.id
                
                # Try to get thread name via Windows API
                try:
                    thread_handle = None
                    try:
                        thread_handle = win32api.OpenThread(
                            win32con.THREAD_QUERY_INFORMATION | win32con.THREAD_SET_INFORMATION, 
                            False, 
                            thread_id
                        )
                        
                        # Skip main thread
                        if thread_id == main_thread_id:
                            # Set main thread to reserved core
                            mask = 1 << reserved_core
                            try:
                                win32process.SetThreadAffinityMask(thread_handle, mask)
                                logging.debug(f"Set main thread {thread_id} to reserved core {reserved_core}")
                            except Exception as e:
                                logging.debug(f"Failed to set main thread affinity: {e}")
                            continue
                            
                        # Detect worker threads by checking if they're owned by ThreadPoolExecutor
                        if hasattr(self, 'executor') and isinstance(self.executor, ThreadPoolExecutor):
                            # Check if thread belongs to our executor
                            for worker_thread in self.executor._threads:
                                if hasattr(worker_thread, 'ident') and worker_thread.ident == thread_id:
                                    executor_threads.append((thread_id, thread_handle))
                                    break
                    finally:
                        if thread_handle and thread_id not in [t[0] for t in executor_threads]:
                            win32api.CloseHandle(thread_handle)
                except Exception as e:
                    logging.debug(f"Could not access thread {thread_id}: {e}")
            
            # Assign each worker thread to a specific core (distribute evenly)
            # Skip the reserved core for UI
            available_cores = list(range(physical_cores - 1))
            for i, (thread_id, thread_handle) in enumerate(executor_threads):
                try:
                    # Assign to a specific core in round-robin fashion
                    core_index = available_cores[i % len(available_cores)]
                    
                    # Create affinity mask for this core
                    mask = 1 << core_index
                    
                    # Set thread affinity
                    win32process.SetThreadAffinityMask(thread_handle, mask)
                    assigned_cores[thread_id] = core_index
                    logging.info(f"Set worker thread {thread_id} affinity to core {core_index}")
                except Exception as e:
                    logging.debug(f"Could not set affinity for thread {thread_id}: {e}")
                finally:
                    if thread_handle:
                        win32api.CloseHandle(thread_handle)
                        
            if assigned_cores:
                logging.info(f"Assigned {len(assigned_cores)} worker threads to specific CPU cores")
            else:
                logging.debug("No worker threads found to assign CPU affinity")
                
        except ImportError as e:
            logging.debug(f"Required package not available for thread affinity: {e}")
        except Exception as e:
            logging.debug(f"Failed to set thread affinity: {e}")

    def warmup_models(self, sample_image=None):
        """
        Warm up all models with a sample image or dummy data to pre-cache execution paths
        
        Args:
            sample_image: Optional sample image to use for warmup (NumPy array)
                          If None, a dummy input will be created
        """
        logging.info("Starting ensemble model warmup...")
        
        # If no sample image provided, create a dummy one
        if sample_image is None:
            # Create a standard size dummy image
            sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
        with self.model_lock:
            total_models = len(self.models)
            for i, model in enumerate(self.models):
                try:
                    if model.loaded:
                        logging.info(f"Warming up model {i+1}/{total_models}: {model.name}")
                        # Run multiple warmup inferences
                        for j in range(3):
                            start_time = time.time()
                            _ = model.predict(sample_image)
                            end_time = time.time()
                            logging.debug(f"Model {model.name} warmup {j+1}/3: {(end_time-start_time)*1000:.1f}ms")
                except Exception as e:
                    logging.warning(f"Error warming up model {model.name}: {str(e)}")
                    
        # Run one full ensemble prediction to warm up the aggregation logic
        try:
            logging.info("Warming up full ensemble prediction pipeline...")
            start_time = time.time()
            _ = self.predict(sample_image)
            end_time = time.time()
            logging.info(f"Full ensemble warmup complete in {(end_time-start_time)*1000:.1f}ms")
        except Exception as e:
            logging.warning(f"Full ensemble warmup failed: {str(e)}")
            
        logging.info("Ensemble warmup complete")
        return True 

    def _predict_with_model_batch(self, model_index: int, images: List[np.ndarray]) -> Tuple[int, List[Tuple[List[float], int]]]:
        """
        Run batch prediction on a single model - for internal use with executor
        
        Args:
            model_index: Index of the model in self.models
            images: List of input images
            
        Returns:
            Tuple of (model_index, list of (predictions, class_index) tuples)
        """
        model = self.models[model_index]
        batch_results = []
        
        try:
            # Check if model supports batch prediction
            if hasattr(model, 'predict_batch'):
                # Use native batch prediction if available
                predictions_batch, class_indices = model.predict_batch(images)
                for i in range(len(images)):
                    batch_results.append((predictions_batch[i], class_indices[i]))
            else:
                # Fall back to processing images individually
                for image in images:
                    predictions, class_index = model.predict(image)
                    batch_results.append((predictions, class_index))
        except Exception as e:
            logging.error(f"Error in batch prediction for model {model_index}: {e}")
            # Return empty results on error
            batch_results = [([0.0] * model.num_classes, 0) for _ in range(len(images))]
            
        return model_index, batch_results
    
    def _aggregate_batch_results(self, 
                                model_results: List[Tuple[List[float], int]],
                                threshold: float) -> Tuple[int, float]:
        """
        Aggregate results from a single image across multiple models
        Similar to _aggregate_results but with different input format for batch processing
        
        Args:
            model_results: List of (predictions, class_index) for each model
            threshold: Confidence threshold
            
        Returns:
            Tuple of (aggregated class, aggregated confidence)
        """
        # Extract the results in the format expected by _aggregate_results
        formatted_results = []
        for model_idx, (predictions, class_idx) in enumerate(model_results):
            if len(predictions) > 0:
                confidence = predictions[class_idx]
                formatted_results.append((model_idx, class_idx, confidence, predictions))
        
        # Use existing aggregation logic
        return self._aggregate_results(formatted_results, threshold)
    
    def predict_batch(self, images: List[np.ndarray], threshold: float = 0.8) -> List[Tuple[int, float]]:
        """
        Run inference on a batch of images
        
        Args:
            images: List of input images
            threshold: Confidence threshold
            
        Returns:
            List of (predicted class, confidence) tuples
        """
        if not self.models:
            return [(0, 0.0)] * len(images)
        
        if len(images) == 0:
            return []
        
        # Create a separate batch for each model
        # This is memory-inefficient but avoids threading issues with TensorFlow
        batches = []
        
        for i in range(len(self.models)):
            model_batch = []
            for img in images:
                # Make a copy to avoid race conditions
                model_batch.append(img.copy())
            batches.append((i, model_batch))
        
        # Initialize executor if needed
        self.init_executor()
        
        # Submit batch prediction tasks - one task per model
        futures = []
        for model_idx, batch in batches:
            futures.append(self.executor.submit(self._predict_with_model_batch, model_idx, batch))
        
        # Wait for all predictions to complete
        concurrent.futures.wait(futures)
        
        # Collect results
        model_results = []
        for future in futures:
            try:
                model_idx, predictions = future.result()
                model_results.append((model_idx, predictions))
            except Exception as e:
                logging.error(f"Error in batch prediction: {e}")
        
        # No model results
        if not model_results:
            return [(0, 0.0)] * len(images)
        
        # Aggregate results for each image
        batch_results = []
        for img_idx in range(len(images)):
            # Collect predictions for this image from all models
            img_results = []
            for model_idx, predictions in model_results:
                # Make sure we have a prediction for this image
                if img_idx < len(predictions):
                    confidences, class_idx = predictions[img_idx]
                    confidence = confidences[class_idx]
                    img_results.append((model_idx, class_idx, confidence, confidences))
            
            # Aggregate results for this image
            if img_results:
                class_id, confidence = self._aggregate_results(img_results, threshold)
                batch_results.append((class_id, confidence))
            else:
                batch_results.append((0, 0.0))
        
        return batch_results