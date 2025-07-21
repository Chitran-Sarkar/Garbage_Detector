#utils.py
import os
import sys
import tkinter as tk
import logging
import threading
import time
import queue
import gc
import platform
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ctypes
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, TypeVar, Generic
import cv2
import weakref
from collections import defaultdict, deque

# Load Windows-specific modules when available
if platform.system() == 'Windows':
    try:
        import win32process
        import win32api
        import win32con
        import win32security
        HAS_WIN32API = True
    except ImportError:
        HAS_WIN32API = False
else:
    HAS_WIN32API = False

# Check for NumPy with SIMD support
HAS_NUMPY_SIMD = hasattr(np, '__config__') and any(x in str(np.__config__.show()) for x in ['AVX', 'SSE'])

# Import performance monitoring tools if available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def redirect_stdout_if_needed():
    """Redirect stdout/stderr to null if not initialized"""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')


def resource_path(relative_path):
    """Get absolute resource path for both development and bundled app"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        
    path = os.path.join(base_path, relative_path)
    return os.path.normpath(path)


def get_screen_scale(orig_width, orig_height):
    """Get screen dimensions and calculate scaling factors"""
    try:
        root = tk.Tk()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return width, height, width / orig_width, height / orig_height
    except Exception as e:
        logging.error(f"Error getting screen scale: {e}")
        return 1920, 1080, 1.0, 1.0


class MemoryManager:
    """Memory management with page locking for critical buffers to prevent disk swapping"""
    def __init__(self, max_locked_memory_mb=100):
        self.locked_buffers = {}
        self.buffer_pools = {}  # Dict of size category -> list of buffers
        self.buffer_pool_usage = {}  # Track when each pool was last used
        self.lock = threading.RLock()
        self.can_lock_memory = self._check_memory_lock_capability()
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30.0  # Check for idle buffers every 30 seconds
        
        # Memory limits for safety
        self.max_locked_memory_mb = max_locked_memory_mb
        self.current_locked_memory_bytes = 0
        
        # Leak detection - track buffer usage 
        self.buffer_allocation_stack = {}
        self.buffer_access_counts = {}
        self.allocation_count = 0
        self.release_count = 0
        
        # Initialize GPU pinned memory if available
        self.has_cuda = False
        self.cuda_context = None
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                self.has_cuda = True
                logging.info("CUDA support detected for memory management")
                if not cv2.cuda.getCudaEnabledDeviceCount():
                    cv2.cuda.setDevice(0)
            except Exception as e:
                logging.warning(f"Error initializing CUDA for memory management: {e}")
                self.has_cuda = False
        
        # Small object pools for frequently allocated objects
        self.small_object_pools = {}
        self.small_object_pool_locks = {}
        self.small_object_pool_stats = defaultdict(lambda: {"created": 0, "reused": 0})
        
        # Start background cleanup thread
        self._start_cleanup_thread()
        
    def _check_memory_lock_capability(self) -> bool:
        """Check if process can lock memory pages"""
        if not HAS_WIN32API:
            return False
            
        try:
            process = win32api.GetCurrentProcess()
            token = win32security.OpenProcessToken(
                process, 
                win32security.TOKEN_QUERY | win32security.TOKEN_ADJUST_PRIVILEGES
            )
            
            lock_memory_privilege = win32security.LookupPrivilegeValue(
                None, "SeLockMemoryPrivilege"
            )
            
            new_privileges = [(lock_memory_privilege, win32security.SE_PRIVILEGE_ENABLED)]
            win32security.AdjustTokenPrivileges(token, False, new_privileges)
            
            return win32api.GetLastError() == 0
        except Exception as e:
            logging.debug(f"Memory lock capability check failed: {e}")
            return False
    
    def _start_cleanup_thread(self):
        """Start background thread to periodically cleanup unused buffers"""
        self.cleanup_running = True
        
        def cleanup_worker():
            while self.cleanup_running:
                try:
                    time.sleep(5.0)  # Check every 5 seconds
                    current_time = time.time()
                    
                    if current_time - self.last_cleanup_time < self.cleanup_interval:
                        continue
                        
                    self._cleanup_idle_buffers()
                    self.last_cleanup_time = current_time
                except Exception as e:
                    logging.debug(f"Error in memory cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(
            target=cleanup_worker, 
            name="MemoryCleanupThread",
            daemon=True
        )
        cleanup_thread.start()
    
    def _cleanup_idle_buffers(self):
        """Release memory from buffer pools that haven't been used recently"""
        with self.lock:
            current_time = time.time()
            idle_threshold = 60.0  # 60 seconds of non-use
            
            pools_to_clean = []
            for size_key, last_used in list(self.buffer_pool_usage.items()):
                if current_time - last_used > idle_threshold:
                    pools_to_clean.append(size_key)
            
            buffers_released = 0
            memory_released = 0
            
            for size_key in pools_to_clean:
                if size_key in self.buffer_pools:
                    pool = self.buffer_pools[size_key]
                    for buffer in pool:
                        memory_released += buffer.nbytes
                        buffers_released += 1
                    
                    self.buffer_pools[size_key] = []
                    del self.buffer_pool_usage[size_key]
            
            if buffers_released > 0:
                logging.debug(f"Released {buffers_released} idle buffers, freeing {memory_released / (1024*1024):.2f} MB")
                
                if memory_released > 50 * 1024 * 1024:  # > 50MB
                    gc.collect()
    
    def _get_size_category(self, shape, dtype):
        """Categorize buffer by size for more efficient pooling"""
        element_size = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        buffer_size = total_elements * element_size
        
        if buffer_size <= 4096:  # 4KB
            size_category = "tiny"
        elif buffer_size <= 65536:  # 64KB
            size_category = "small"
        elif buffer_size <= 1048576:  # 1MB
            size_category = "medium"
        elif buffer_size <= 16777216:  # 16MB
            size_category = "large"
        else:
            size_category = "huge"
            
        if len(shape) >= 2 and shape[0] < 2000 and shape[1] < 2000:
            return f"{size_category}_{shape[0]}x{shape[1]}"
        else:
            return size_category
    
    def lock_memory(self, buffer, key):
        """Lock a memory buffer to prevent it from being swapped to disk (Windows only)"""
        if not self.can_lock_memory:
            return buffer
            
        if key in self.locked_buffers:
            # Already locked with this key
            return buffer
            
        buffer_size_bytes = buffer.nbytes
        if self.current_locked_memory_bytes + buffer_size_bytes > self.max_locked_memory_mb * 1024 * 1024:
            logging.warning(f"Memory lock limit reached ({self.max_locked_memory_mb} MB), not locking buffer with key {key}")
            return buffer
            
        try:
            buffer_addr = buffer.__array_interface__['data'][0]
            
            buffer_size = buffer.nbytes
            
            if sys.platform.startswith('win'):
                try:
                    import win32con
                    import win32api
                    
                    win32api.VirtualLock(buffer_addr, buffer_size)
                    
                    self.locked_buffers[key] = {
                        'addr': buffer_addr,
                        'size': buffer_size,
                        'buffer': buffer,
                        'allocation_stack': traceback.format_stack()
                    }
                    
                    self.current_locked_memory_bytes += buffer_size
                    
                    self.allocation_count += 1
                    
                    logging.debug(f"Locked {buffer_size/1024:.1f} KB in memory for key '{key}'")
                    
                    try:
                        import app_loop
                        if hasattr(app_loop, 'memory_tracking'):
                            app_loop.memory_tracking["active_buffers"].add(buffer)
                    except (ImportError, AttributeError):
                        pass
                        
                    return buffer
                except ImportError:
                    return buffer
            
            return buffer
        except Exception as e:
            logging.warning(f"Failed to lock memory: {e}")
            return buffer
    
    def unlock_memory(self, key: str) -> bool:
        """Unlock a previously locked memory buffer"""
        if not self.can_lock_memory or not HAS_WIN32API:
            return False
            
        with self.lock:
            if key not in self.locked_buffers:
                return False
                
            buffer_info = self.locked_buffers[key]
            
            try:
                result = ctypes.windll.kernel32.VirtualUnlock(
                    ctypes.c_void_p(buffer_info['addr']),
                    ctypes.c_size_t(buffer_info['size'])
                )
                
                del self.locked_buffers[key]
                
                self.current_locked_memory_bytes -= buffer_info['size']
                
                return result != 0
            except Exception as e:
                logging.debug(f"Failed to unlock memory for '{key}': {e}")
                return False
    
    def get_buffer(self, shape: tuple, dtype=np.uint8, key: str = None) -> np.ndarray:
        """Get a pre-allocated buffer of the specified shape and type"""
        size_category = self._get_size_category(shape, dtype)
        
        with self.lock:
            self.buffer_pool_usage[size_category] = time.time()
            
            pool = self.buffer_pools.get(size_category, [])
            
            if pool:
                buffer = pool.pop()
                
                if buffer.shape != shape or buffer.dtype != dtype:
                    try:
                        if buffer.size >= np.prod(shape):
                            buffer = buffer.ravel()[:np.prod(shape)].reshape(shape).astype(dtype)
                        else:
                            buffer = self._allocate_new_buffer(shape, dtype)
                    except Exception:
                        buffer = self._allocate_new_buffer(shape, dtype)
                
                buffer.fill(0)
            else:
                buffer = self._allocate_new_buffer(shape, dtype)
                
            if key and self.can_lock_memory:
                buffer = self.lock_memory(buffer, key)
                
            return buffer
    
    def _allocate_new_buffer(self, shape, dtype):
        """Create a new buffer, using pinned memory if available"""
        if self.has_cuda:
            try:
                buffer = cv2.cuda.registerPageLocked(np.zeros(shape, dtype=dtype))
                return buffer
            except Exception as e:
                logging.debug(f"Failed to allocate pinned memory: {e}")
        
        return np.zeros(shape, dtype=dtype)
    
    def release_buffer(self, buffer: np.ndarray, key: str = None) -> bool:
        """Return a buffer to the pool for reuse"""
        if buffer is None:
            return False
            
        if key and key in self.locked_buffers:
            self.unlock_memory(key)
        
        size_category = self._get_size_category(buffer.shape, buffer.dtype)
        
        with self.lock:
            self.buffer_pool_usage[size_category] = time.time()
            
            max_pool_size = 20  # Default max buffers per category
            
            if 'tiny' in size_category or 'small' in size_category:
                max_pool_size = 50  # Keep more small buffers
            elif 'huge' in size_category:
                max_pool_size = 5   # Keep fewer large buffers
                
            if len(self.buffer_pools[size_category]) < max_pool_size:
                self.buffer_pools[size_category].append(buffer)
                return True
                
        return False
    
    def cleanup(self) -> None:
        """Release all locked memory and clear buffer pools"""
        self.cleanup_running = False  # Stop cleanup thread
        
        with self.lock:
            for key in list(self.locked_buffers.keys()):
                self.unlock_memory(key)
                
            for pool in self.buffer_pools.values():
                pool.clear()
            self.buffer_pools.clear()
            self.buffer_pool_usage.clear()
            
            if self.has_cuda and self.cuda_context:
                try:
                    cv2.cuda.releaseMemPool()
                    cv2.cuda.resetDevice()
                except Exception:
                    pass
            
            gc.collect()

    def create_tensor_pool(self, shape, dtype=np.float32, count=5):
        """Create a pool of pre-allocated tensors for model inference"""
        pool_key = f"tensor_{shape}_{dtype.__name__}"
        
        with self.lock:
            if pool_key not in self.buffer_pools:
                self.buffer_pools[pool_key] = []
                
                for _ in range(count):
                    if self.has_cuda and dtype in (np.float32, np.float16, np.int8):
                        try:
                            import pycuda.driver as cuda
                            tensor = cuda.pagelocked_empty(shape, dtype)
                        except (ImportError, Exception):
                            tensor = np.zeros(shape, dtype=dtype)
                    else:
                        tensor = np.zeros(shape, dtype=dtype)
                        
                    self.buffer_pools[pool_key].append(tensor)
                    
                self.buffer_pool_usage[pool_key] = time.time()
                
        return pool_key

    def get_tensor(self, pool_key):
        """Get a tensor from the pool"""
        with self.lock:
            if pool_key in self.buffer_pools and self.buffer_pools[pool_key]:
                self.buffer_pool_usage[pool_key] = time.time()
                return self.buffer_pools[pool_key].pop()
            
        return None

    def return_tensor(self, tensor, pool_key):
        """Return a tensor to the pool when done with it"""
        with self.lock:
            if pool_key in self.buffer_pools:
                self.buffer_pool_usage[pool_key] = time.time()
                self.buffer_pools[pool_key].append(tensor)
                return True
                
        return False

    def get_shared_array(self, shape, dtype=np.uint8, name=None):
        """Get a shared memory array that can be accessed across processes"""
        try:
            from shared_memory import create_shared_buffer
            
            if name is None:
                import uuid
                name = f"shm_{uuid.uuid4().hex[:8]}"
                
            buffer = create_shared_buffer(name=name, shape=shape, dtype=dtype)
            
            return buffer.get_array(), name
        except (ImportError, Exception) as e:
            logging.warning(f"Failed to create shared memory array: {e}")
            return np.zeros(shape, dtype=dtype), None

    def get_pinned_memory_buffer(self, shape, dtype=np.uint8):
        """Get a CUDA pinned memory buffer for zero-copy transfers"""
        if not self.has_cuda:
            return np.zeros(shape, dtype=dtype)
        
        try:
            import pycuda.driver as cuda
            
            if not cuda.Context.get_current():
                cuda.init()
                ctx = cuda.Device(0).make_context()
                ctx.pop()
            
            buffer = cuda.pagelocked_empty(shape, dtype)
            return buffer
        except (ImportError, Exception) as e:
            logging.debug(f"Failed to allocate pinned memory: {e}")
            return np.zeros(shape, dtype=dtype)

    def get_memory_stats(self):
        """Get memory usage statistics for monitoring"""
        with self.lock:
            stats = {
                'locked_buffers_count': len(self.locked_buffers),
                'locked_memory_mb': self.current_locked_memory_bytes / (1024 * 1024),
                'buffer_pools_count': len(self.buffer_pools),
                'total_pool_buffers': sum(len(pool) for pool in self.buffer_pools.values()),
                'allocation_count': self.allocation_count,
                'release_count': self.release_count,
                'unreleased_count': self.allocation_count - self.release_count,
                'memory_limits': {
                    'max_locked_mb': self.max_locked_memory_mb
                }
            }
            return stats

    def create_small_object_pool(self, pool_id: str, factory_func: Callable, max_size: int = 100,
                               reset_func: Optional[Callable] = None) -> str:
        """
        Create a pool for small, frequently allocated objects like dicts, lists, etc.
        
        Args:
            pool_id: Unique identifier for this pool
            factory_func: Function that creates a new instance of the object
            max_size: Maximum number of objects to keep in the pool
            reset_func: Optional function to reset/clear an object before reuse
            
        Returns:
            Pool identifier
        """
        with self.lock:
            if pool_id in self.small_object_pools:
                logging.warning(f"Small object pool '{pool_id}' already exists")
                return pool_id
                
            self.small_object_pools[pool_id] = deque(maxlen=max_size)
            self.small_object_pool_locks[pool_id] = threading.RLock()
            
            # Store the factory and reset functions
            self.small_object_pools[f"{pool_id}_factory"] = factory_func
            self.small_object_pools[f"{pool_id}_reset"] = reset_func
            
            logging.debug(f"Created small object pool '{pool_id}' with max size {max_size}")
            return pool_id
            
    def get_small_object(self, pool_id: str) -> Any:
        """
        Get an object from the specified pool, or create a new one if the pool is empty
        
        Args:
            pool_id: The pool identifier
            
        Returns:
            A pooled object, ready for use
        """
        if pool_id not in self.small_object_pools:
            raise ValueError(f"Small object pool '{pool_id}' does not exist")
            
        lock = self.small_object_pool_locks[pool_id]
        factory_func = self.small_object_pools.get(f"{pool_id}_factory")
        
        with lock:
            if not self.small_object_pools[pool_id]:
                # Pool is empty, create a new object
                obj = factory_func()
                self.small_object_pool_stats[pool_id]["created"] += 1
                return obj
                
            # Get an object from the pool
            obj = self.small_object_pools[pool_id].pop()
            self.small_object_pool_stats[pool_id]["reused"] += 1
            
            # Reset the object if a reset function is provided
            reset_func = self.small_object_pools.get(f"{pool_id}_reset")
            if reset_func is not None:
                reset_func(obj)
                
            return obj
            
    def return_small_object(self, obj: Any, pool_id: str) -> bool:
        """
        Return an object to its pool for later reuse
        
        Args:
            obj: The object to return to the pool
            pool_id: The pool identifier
            
        Returns:
            True if the object was returned to the pool, False otherwise
        """
        if pool_id not in self.small_object_pools:
            return False
            
        lock = self.small_object_pool_locks[pool_id]
        
        with lock:
            # Only add if we haven't reached the maximum size
            if len(self.small_object_pools[pool_id]) < self.small_object_pools[pool_id].maxlen:
                self.small_object_pools[pool_id].append(obj)
                return True
                
        return False
        
    def get_small_object_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about small object pools
        
        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            stats = {}
            for pool_id in self.small_object_pools:
                if isinstance(pool_id, str) and not pool_id.endswith("_factory") and not pool_id.endswith("_reset"):
                    pool_stats = self.small_object_pool_stats[pool_id].copy()
                    pool_stats["size"] = len(self.small_object_pools[pool_id])
                    pool_stats["max_size"] = self.small_object_pools[pool_id].maxlen
                    stats[pool_id] = pool_stats
            return stats


class PriorityTaskScheduler:
    """Task scheduler with priority levels for optimizing processing of critical vs non-critical tasks"""
    HIGH = 0
    NORMAL = 1
    LOW = 2
    
    def __init__(self, high_workers=1, normal_workers=2, low_workers=1):
        self.running = True
        self.lock = threading.RLock()
        
        self.task_queues = {
            self.HIGH: queue.PriorityQueue(),
            self.NORMAL: queue.PriorityQueue(),
            self.LOW: queue.PriorityQueue()
        }
        
        self.thread_pools = {
            self.HIGH: self._create_thread_pool(high_workers, "HighPriority"),
            self.NORMAL: self._create_thread_pool(normal_workers, "NormalPriority"),
            self.LOW: self._create_thread_pool(low_workers, "LowPriority")
        }
        
        self.task_counter = 0
        
        logging.info(f"Priority task scheduler initialized with {high_workers}/{normal_workers}/{low_workers} workers")
    
    def _create_thread_pool(self, count, name_prefix):
        """Create a pool of worker threads for a specific priority level"""
        threads = []
        for i in range(count):
            thread = threading.Thread(
                target=self._worker_thread,
                name=f"{name_prefix}-Worker-{i}",
                args=(name_prefix,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        return threads
    
    def _worker_thread(self, priority_name):
        """Worker thread that processes tasks from the priority queues"""
        if HAS_WIN32API:
            try:
                thread_handle = win32api.GetCurrentThread()
                if "High" in priority_name:
                    win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_ABOVE_NORMAL)
                elif "Low" in priority_name:
                    win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_BELOW_NORMAL)
            except Exception as e:
                logging.debug(f"Could not set thread priority: {e}")

        priority_map = {
            "HighPriority": self.HIGH,
            "NormalPriority": self.NORMAL, 
            "LowPriority": self.LOW
        }
        
        priority_level = priority_map.get(priority_name, self.NORMAL)
        task_queue = self.task_queues[priority_level]
        
        while self.running:
            try:
                if priority_level > self.HIGH and not self.task_queues[self.HIGH].empty():
                    time.sleep(0.01)
                    continue
                    
                try:
                    _, task, args, kwargs, callback = task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                try:
                    start_time = time.time()
                    result = task(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    if elapsed > 0.1:  # More than 100ms
                        logging.debug(f"Slow task {task.__name__} ({priority_name}): {elapsed*1000:.1f}ms")
                    
                    if callback and self.running:
                        try:
                            callback(result)
                        except Exception as callback_error:
                            logging.error(f"Error in callback for task {task.__name__}: {callback_error}")
                            traceback.print_exc()
                except Exception as task_error:
                    logging.error(f"Error in {priority_name} task {task.__name__}: {task_error}")
                    traceback.print_exc()
                finally:
                    task_queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in {priority_name} worker thread: {e}")
                time.sleep(0.5)
    
    def schedule_task(self, task, priority=NORMAL, callback=None, *args, **kwargs):
        """Schedule a task to be executed with the specified priority"""
        with self.lock:
            if not self.running:
                return False
                
            self.task_counter += 1
            
            if priority not in [self.HIGH, self.NORMAL, self.LOW]:
                priority = self.NORMAL
                
            try:
                self.task_queues[priority].put((self.task_counter, task, args, kwargs, callback))
                return True
            except Exception as e:
                logging.error(f"Error scheduling task: {e}")
                return False
    
    def shutdown(self, wait=True):
        """Shutdown the scheduler and all thread pools"""
        with self.lock:
            if not self.running:
                return  # Already shut down
                
            self.running = False
            
        if wait:
            for priority, task_queue in self.task_queues.items():
                try:
                    task_queue.join()
                except Exception as e:
                    logging.error(f"Error waiting for priority {priority} tasks: {e}")
                    
        for priority, task_queue in self.task_queues.items():
            try:
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                        task_queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                logging.error(f"Error clearing task queue: {e}")
                
        gc.collect()
        logging.info("Priority task scheduler shutdown complete")


class BackgroundTaskScheduler:
    """A scheduler for running non-critical tasks in the background at lower priority"""
    
    def __init__(self, max_workers=2, name="BGTaskScheduler"):
        self.task_queue = queue.Queue()
        self.max_workers = max(1, max_workers)  # Ensure at least 1 worker
        self.workers = []
        self.running = True
        self.name = name
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, 
                                         thread_name_prefix=f"{name}-Worker")
        self.futures = []
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"{name}-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logging.info(f"Started {self.max_workers} background task workers")
    
    def _worker_thread(self):
        """Worker thread that processes tasks from the queue"""
        if HAS_WIN32API:
            try:
                thread_handle = win32api.GetCurrentThread()
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_BELOW_NORMAL)
                logging.debug(f"Set {threading.current_thread().name} to lower priority")
            except Exception as e:
                logging.debug(f"Could not lower thread priority: {e}")
            
        while self.running:
            try:
                try:
                    task, args, kwargs, callback = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue  # No tasks available, continue waiting
                
                try:
                    result = task(*args, **kwargs)
                    
                    if callback and self.running:
                        try:
                            callback(result)
                        except Exception as callback_error:
                            logging.error(f"Error in callback for task {task.__name__}: {callback_error}")
                            traceback.print_exc()
                except Exception as task_error:
                    logging.error(f"Error in background task {task.__name__}: {task_error}")
                    traceback.print_exc()
                finally:
                    self.task_queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in worker thread: {e}")
                time.sleep(0.5)  # Brief pause before continuing
    
    def schedule_task(self, task, callback=None, *args, **kwargs):
        """Schedule a task to be executed in the background"""
        with self.lock:
            if self.running:
                try:
                    self.task_queue.put((task, args, kwargs, callback))
                    return True
                except Exception as e:
                    logging.error(f"Error scheduling task: {e}")
        return False
    
    def shutdown(self, wait=True):
        """Shutdown the scheduler and stop all worker threads"""
        with self.lock:
            if not self.running:
                return  # Already shut down
                
            self.running = False
            
        if wait:
            try:
                self.task_queue.join()
            except Exception as e:
                logging.error(f"Error waiting for tasks to complete: {e}")
                
        try:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Error clearing task queue: {e}")
            
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=False)
            except Exception as e:
                logging.error(f"Error shutting down executor: {e}")
        
        gc.collect()
            
        logging.info(f"Background task scheduler {self.name} shutdown")


class HardwareMonitor:
    """Monitors hardware status including thermal throttling, CPU/GPU utilization, and memory usage"""
    
    def __init__(self, check_interval=5.0, callback=None):
        self.check_interval = max(1.0, check_interval)  # Minimum 1 second interval
        self.callback = callback
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        self.last_status = {}
        self.throttling_detected = False
        self.thermal_throttling = False
        self.power_throttling = False
        self.memory_pressure = False
        self.metrics_history = []
        
        self.has_nvidia_gpu = False
        self.has_amd_gpu = False
        self.has_intel_gpu = False
        
        # Check for OpenHardwareMonitor folder
        if platform.system() == 'Windows':
            try:
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                ohm_folder = os.path.join(script_dir, "OpenHardwareMonitor")
                ohm_exe = os.path.join(ohm_folder, "OpenHardwareMonitor.exe")
                
                if os.path.isdir(ohm_folder):
                    logging.info(f"OpenHardwareMonitor folder found at: {ohm_folder}")
                    if os.path.isfile(ohm_exe):
                        logging.info(f"OpenHardwareMonitor.exe found at: {ohm_exe}")
                    else:
                        logging.warning(f"OpenHardwareMonitor folder exists but exe not found at: {ohm_exe}")
                else:
                    logging.debug("OpenHardwareMonitor folder not found in script directory")
            except Exception as e:
                logging.debug(f"Error checking for OpenHardwareMonitor: {e}")
        
        self._detect_gpus()
        
    def _detect_gpus(self):
        """Detect available GPUs in system"""
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    cuda.init()
                    device_count = cuda.Device.count()
                    if device_count > 0:
                        self.has_nvidia_gpu = True
                        logging.info(f"Detected {device_count} NVIDIA GPU(s)")
                        device = cuda.Device(0)
                        name = device.name()
                        logging.info(f"Primary GPU: {name}")
                except ImportError:
                    logging.debug("pycuda not available, cannot detect NVIDIA GPUs")
                except Exception as e:
                    logging.debug(f"Error detecting NVIDIA GPUs: {e}")
                
            if platform.system() == 'Windows':
                try:
                    import winreg
                    amd_key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, amd_key_path) as key:
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, subkey_name) as subkey:
                                    try:
                                        provider_name = winreg.QueryValueEx(subkey, "ProviderName")[0]
                                        if "AMD" in provider_name:
                                            self.has_amd_gpu = True
                                            logging.info("Detected AMD GPU")
                                            break
                                    except:
                                        pass
                                i += 1
                            except WindowsError:
                                break
                except Exception as e:
                    logging.debug(f"Error detecting AMD GPUs: {e}")
                
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if "Intel" in info.get("vendor_id", ""):
                    self.has_intel_gpu = True
                    logging.info("Detected Intel integrated GPU")
            except ImportError:
                if platform.system() == 'Windows':
                    try:
                        import winreg
                        intel_key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, intel_key_path) as key:
                            i = 0
                            while True:
                                try:
                                    subkey_name = winreg.EnumKey(key, i)
                                    with winreg.OpenKey(key, subkey_name) as subkey:
                                        try:
                                            provider_name = winreg.QueryValueEx(subkey, "ProviderName")[0]
                                            if "Intel" in provider_name:
                                                self.has_intel_gpu = True
                                                logging.info("Detected Intel GPU")
                                                break
                                        except:
                                            pass
                                    i += 1
                                except WindowsError:
                                    break
                    except Exception as e:
                        logging.debug(f"Error detecting Intel GPUs: {e}")
        except Exception as e:
            logging.error(f"Error in GPU detection: {e}")
            
    def _init_openhardwaremonitor(self):
        """Initialize OpenHardwareMonitor if available"""
        if platform.system() != 'Windows':
            return False
            
        try:
            import os
            import subprocess
            import wmi
            
            # Check if OpenHardwareMonitor namespace is already available
            try:
                w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                w.Sensor()
                logging.info("OpenHardwareMonitor is already running")
                return True
            except Exception:
                logging.debug("OpenHardwareMonitor namespace not available")
            
            # Try to find and start OpenHardwareMonitor
            # First check in the same directory as the main script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Prioritize paths, with the script directory first
            ohm_paths = [
                os.path.join(script_dir, "OpenHardwareMonitor", "OpenHardwareMonitor.exe"),
                os.path.join(script_dir, "OpenHardwareMonitor.exe"),
                os.path.expanduser("~\\Downloads\\OpenHardwareMonitor\\OpenHardwareMonitor.exe"),
                "C:\\Program Files\\OpenHardwareMonitor\\OpenHardwareMonitor.exe",
                "C:\\Program Files (x86)\\OpenHardwareMonitor\\OpenHardwareMonitor.exe"
            ]
            
            for path in ohm_paths:
                if os.path.exists(path):
                    try:
                        logging.info(f"Found OpenHardwareMonitor at: {path}")
                        # Start OpenHardwareMonitor minimized and hidden
                        subprocess.Popen([path, "/minimized"], 
                                        shell=True, 
                                        creationflags=subprocess.CREATE_NO_WINDOW,
                                        start_new_session=True)
                        logging.info(f"Started OpenHardwareMonitor from {path}")
                        
                        # Wait a moment for OHM to initialize
                        import time
                        time.sleep(2)
                        
                        # Check if it's now available
                        try:
                            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                            w.Sensor()
                            return True
                        except Exception:
                            logging.warning("OpenHardwareMonitor started but WMI namespace not available")
                            continue
                    except Exception as e:
                        logging.warning(f"Failed to start OpenHardwareMonitor: {e}")
            
            logging.warning("OpenHardwareMonitor not found. CPU temperature monitoring will be limited")
            return False
        except Exception as e:
            logging.error(f"Error initializing OpenHardwareMonitor: {e}")
            return False

    def start(self):
        """Start the hardware monitoring thread"""
        with self.lock:
            if self.running:
                return  # Already running
            
            # Try to initialize OpenHardwareMonitor for better CPU temperature monitoring
            if platform.system() == 'Windows':
                self._init_openhardwaremonitor()
                
            self.running = True
            self.thread = threading.Thread(target=self._monitor_thread, 
                                         daemon=True, 
                                         name="HardwareMonitorThread")
            self.thread.start()
            logging.info("Hardware monitoring started")
            return True
    
    def stop(self):
        """Stop the hardware monitoring thread"""
        with self.lock:
            if not self.running:
                return  # Already stopped
                
            self.running = False
            
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=2.0)
            except Exception as e:
                logging.error(f"Error waiting for hardware monitor thread to terminate: {e}")
                
        logging.info("Hardware monitoring stopped")
    
    def _monitor_thread(self):
        """Thread function for monitoring hardware status"""
        if not HAS_PSUTIL:
            logging.warning("psutil not available, hardware monitoring disabled")
            monitoring_enabled = False
        else:
            monitoring_enabled = True
            
        nvidia_smi_available = False
        try:
            if self.has_nvidia_gpu:
                import py3nvml.py3nvml as nvml
                nvml.nvmlInit()
                nvidia_smi_available = True
                logging.info("NVIDIA GPU monitoring enabled")
        except ImportError:
            logging.debug("py3nvml not available, NVIDIA GPU monitoring disabled")
        except Exception as e:
            logging.debug(f"Error initializing NVIDIA monitoring: {e}")
        
        while self.running and monitoring_enabled:
            try:
                status = self._collect_metrics(nvidia_smi_available)
                
                self._detect_throttling(status)
                
                self.last_status = status
                
                if self.callback:
                    try:
                        self.callback(status)
                    except Exception as callback_e:
                        logging.error(f"Error in hardware monitor callback: {callback_e}")
                    
                self.metrics_history.append(status)
                if len(self.metrics_history) > 60:
                    self.metrics_history.pop(0)
                    
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in hardware monitoring: {e}")
                time.sleep(self.check_interval)
                
        if nvidia_smi_available:
            try:
                import py3nvml.py3nvml as nvml
                nvml.nvmlShutdown()
                logging.debug("NVIDIA monitoring shut down")
            except Exception as e:
                logging.debug(f"Error shutting down NVIDIA monitoring: {e}")
    
    def _collect_metrics(self, nvidia_smi_available):
        """Collect hardware metrics from system"""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": 0,
            "memory_percent": 0,
            "gpu_percent": 0,
            "gpu_memory_percent": 0,
            "gpu_temperature": 0,
            "cpu_temperature": 0
        }
        
        try:
            import psutil
            import platform
            
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # CPU temperature collection
            # For Windows, try to use WMI with OpenHardwareMonitor
            if platform.system() == 'Windows':
                try:
                    import wmi
                    # First try with OpenHardwareMonitor namespace
                    try:
                        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                        temperature_sensors = w.Sensor()
                        cpu_temps = []
                        
                        for sensor in temperature_sensors:
                            if sensor.SensorType == 'Temperature' and 'CPU' in sensor.Name:
                                cpu_temps.append(float(sensor.Value))
                        
                        if cpu_temps:
                            metrics["cpu_temperature"] = sum(cpu_temps) / len(cpu_temps)
                            logging.debug(f"CPU temperature from OpenHardwareMonitor: {metrics['cpu_temperature']:.1f}°C")
                    except Exception as ohm_error:
                        logging.debug(f"OpenHardwareMonitor WMI error: {ohm_error}")
                        
                        # Try to initialize OHM if it's not running
                        if "not found" in str(ohm_error) or "found no instances" in str(ohm_error).lower():
                            if hasattr(self, '_init_openhardwaremonitor'):
                                logging.info("Attempting to start OpenHardwareMonitor...")
                                if self._init_openhardwaremonitor():
                                    # Retry with OpenHardwareMonitor now that it's running
                                    try:
                                        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                                        temperature_sensors = w.Sensor()
                                        cpu_temps = []
                                        
                                        for sensor in temperature_sensors:
                                            if sensor.SensorType == 'Temperature' and 'CPU' in sensor.Name:
                                                cpu_temps.append(float(sensor.Value))
                                        
                                        if cpu_temps:
                                            metrics["cpu_temperature"] = sum(cpu_temps) / len(cpu_temps)
                                            logging.debug(f"CPU temperature after starting OpenHardwareMonitor: {metrics['cpu_temperature']:.1f}°C")
                                            return metrics
                                    except Exception as retry_error:
                                        logging.debug(f"Failed to get temperature after starting OpenHardwareMonitor: {retry_error}")
                        
                        # Try with MSAcpi_ThermalZoneTemperature as fallback
                        try:
                            w = wmi.WMI(namespace="root\\wmi")
                            temperature_info = w.MSAcpi_ThermalZoneTemperature()
                            if temperature_info:
                                # Convert tenths of degrees Kelvin to Celsius
                                temp_kelvin = float(temperature_info[0].CurrentTemperature) / 10.0
                                metrics["cpu_temperature"] = temp_kelvin - 273.15
                                logging.debug(f"CPU temperature from ACPI: {metrics['cpu_temperature']:.1f}°C")
                        except Exception as wmi_error:
                            logging.debug(f"ACPI WMI error: {wmi_error}")
                            
                            # If WMI methods fail, try traditional approach
                            if hasattr(psutil, 'sensors_temperatures'):
                                try:
                                    temps = psutil.sensors_temperatures()
                                    if temps and 'coretemp' in temps:
                                        metrics["cpu_temperature"] = max(temp.current for temp in temps.get('coretemp', []))
                                        logging.debug(f"CPU temperature from psutil: {metrics['cpu_temperature']:.1f}°C")
                                except Exception as psutil_error:
                                    logging.debug(f"psutil temperature error: {psutil_error}")
                except ImportError:
                    logging.debug("WMI module not available")
                    # Try psutil if available on Windows
                    if hasattr(psutil, 'sensors_temperatures'):
                        try:
                            temps = psutil.sensors_temperatures()
                            if temps and 'coretemp' in temps:
                                metrics["cpu_temperature"] = max(temp.current for temp in temps.get('coretemp', []))
                        except Exception as e:
                            logging.debug(f"Error getting temperatures via psutil: {e}")
            else:
                # For non-Windows platforms, use psutil if available
                if hasattr(psutil, 'sensors_temperatures'):
                    try:
                        temps = psutil.sensors_temperatures()
                        if temps and 'coretemp' in temps:
                            metrics["cpu_temperature"] = max(temp.current for temp in temps.get('coretemp', []))
                    except Exception as e:
                        logging.debug(f"Error getting temperatures via psutil: {e}")
            
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            
            try:
                disk = psutil.disk_usage('/')
                metrics["disk_percent"] = disk.percent
            except Exception:
                pass
                
            try:
                process = psutil.Process()
                metrics["process_cpu_percent"] = process.cpu_percent(interval=0)
                metrics["process_memory_percent"] = process.memory_percent()
                metrics["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
            
            if nvidia_smi_available and self.has_nvidia_gpu:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["gpu_percent"] = util.gpu
                    except nvml.NVMLError:
                        pass
                    
                    try:
                        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        total_mem = mem_info.total
                        used_mem = mem_info.used
                        if total_mem > 0:
                            metrics["gpu_memory_percent"] = (used_mem / total_mem) * 100
                            metrics["gpu_memory_total_mb"] = total_mem / (1024 * 1024)
                            metrics["gpu_memory_used_mb"] = used_mem / (1024 * 1024)
                    except nvml.NVMLError:
                        pass
                    
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        metrics["gpu_temperature"] = temp
                    except nvml.NVMLError:
                        pass
                    
                    try:
                        throttle_reasons = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        metrics["gpu_throttling"] = throttle_reasons != 0
                        metrics["gpu_thermal_throttling"] = (
                            throttle_reasons & nvml.nvmlClocksThrottleReasonGpuIdle or
                            throttle_reasons & nvml.nvmlClocksThrottleReasonHwSlowdown or
                            throttle_reasons & nvml.nvmlClocksThrottleReasonHwThermalSlowdown
                        )
                    except nvml.NVMLError:
                        pass
                except Exception as e:
                    logging.debug(f"Error collecting NVIDIA GPU metrics: {e}")
            
            if platform.system() == 'Windows' and self.has_amd_gpu:
                try:
                    import wmi
                    w = wmi.WMI(namespace="root\\CIMV2")
                    for gpu in w.Win32_VideoController():
                        if "AMD" in gpu.Name:
                            metrics["gpu_name"] = gpu.Name
                            metrics["gpu_driver_version"] = gpu.DriverVersion
                            break
                except ImportError:
                    pass
                except Exception as e:
                    logging.debug(f"Error collecting AMD GPU metrics: {e}")
            
            if platform.system() == 'Windows' and self.has_intel_gpu:
                try:
                    import wmi
                    w = wmi.WMI(namespace="root\\CIMV2")
                    for gpu in w.Win32_VideoController():
                        if "Intel" in gpu.Name:
                            metrics["gpu_name"] = gpu.Name
                            metrics["gpu_driver_version"] = gpu.DriverVersion
                            break
                except ImportError:
                    pass
                except Exception as e:
                    logging.debug(f"Error collecting Intel GPU metrics: {e}")
                    
        except ImportError:
            logging.debug("psutil not available, cannot collect detailed metrics")
        except Exception as e:
            logging.error(f"Error collecting hardware metrics: {e}")
            
        return metrics
        
    def _detect_throttling(self, status):
        """Detect hardware throttling conditions based on collected metrics"""
        try:
            was_throttling = self.throttling_detected
            
            if status.get("cpu_temperature", 0) > 80:
                self.thermal_throttling = True
            elif status.get("cpu_temperature", 0) < 75:
                self.thermal_throttling = False
                
            if status.get("gpu_temperature", 0) > 80:
                self.thermal_throttling = True
            elif status.get("gpu_temperature", 0) < 75 and not self.thermal_throttling:
                self.thermal_throttling = False
                
            high_cpu = status.get("cpu_percent", 0) > 90
            if high_cpu and len(self.metrics_history) >= 3:
                sustained_high_cpu = all(m.get("cpu_percent", 0) > 85 for m in self.metrics_history[-3:])
                if sustained_high_cpu:
                    self.power_throttling = True
            elif status.get("cpu_percent", 0) < 75:
                self.power_throttling = False
                
            if status.get("memory_available_gb", 4) < 1.0:
                self.memory_pressure = True
            elif status.get("memory_available_gb", 4) > 1.5:
                self.memory_pressure = False
                
            self.throttling_detected = (self.thermal_throttling or 
                                      self.power_throttling or 
                                      self.memory_pressure)
                                      
            if self.throttling_detected != was_throttling:
                if self.throttling_detected:
                    reasons = []
                    if self.thermal_throttling:
                        reasons.append("thermal")
                    if self.power_throttling:
                        reasons.append("power")
                    if self.memory_pressure:
                        reasons.append("memory")
                    logging.warning(f"Hardware throttling detected: {', '.join(reasons)}")
                else:
                    logging.info("Hardware throttling condition cleared")
        except Exception as e:
            logging.error(f"Error detecting throttling: {e}")
    
    def get_status(self):
        """Get the current hardware status"""
        with self.lock:
            status = self.last_status.copy()
            status.update({
                "throttling_detected": self.throttling_detected,
                "thermal_throttling": self.thermal_throttling,
                "power_throttling": self.power_throttling,
                "memory_pressure": self.memory_pressure
            })
            return status
    
    def should_reduce_workload(self):
        """Check if workload should be reduced due to throttling"""
        with self.lock:
            return self.throttling_detected
    
    def get_recommended_parameters(self):
        """Get recommended parameters based on current hardware status"""
        with self.lock:
            params = {
                "skip_frames": 0,
                "reduce_resolution": False,
                "use_simpler_model": False
            }
            
            if self.thermal_throttling:
                params["skip_frames"] = 2
                params["reduce_resolution"] = True
                params["use_simpler_model"] = True
            elif self.power_throttling:
                params["skip_frames"] = 1
                params["reduce_resolution"] = True
            elif self.memory_pressure:
                params["use_simpler_model"] = True
                
            return params


class PerformanceMonitor:
    """Monitor system performance metrics over time to detect degradation and provide recommendations"""
    def __init__(self, window_size=30, check_interval=5.0):
        """Initialize performance monitor"""
        self.window_size = max(5, window_size)
        self.check_interval = max(1.0, check_interval)
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        self.metrics_history = []
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_usage = []
        self.fps_history = []
        self.disk_io_history = []
        
        self.current_status = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_percent': 0,
            'inference_time': 0,
            'fps': 0,
            'disk_io': 0,
            'throttling_detected': False,
            'system_load': 'normal'  # 'normal', 'high', 'critical'
        }
        
        self.last_update = time.time()
        self.degradation_detected = False
        self.optimization_recommendations = []
        
        self.status_callbacks = []
        
    def start(self):
        """Start performance monitoring thread"""
        with self.lock:
            if self.running:
                return False
                
            self.running = True
            self.thread = threading.Thread(
                target=self._monitor_thread,
                daemon=True,
                name="PerformanceMonitorThread"
            )
            self.thread.start()
            logging.info("Performance monitoring started")
            return True
            
    def stop(self):
        """Stop performance monitoring thread"""
        with self.lock:
            if not self.running:
                return False
                
            self.running = False
            
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=2.0)
            except Exception as e:
                logging.error(f"Error waiting for performance monitor thread: {e}")
                
        logging.info("Performance monitoring stopped")
        return True
    
    def add_status_callback(self, callback):
        """Add a callback to be notified of status changes"""
        with self.lock:
            self.status_callbacks.append(callback)
            
    def register_callback(self, callback):
        """Alias for add_status_callback for compatibility"""
        return self.add_status_callback(callback)
            
    def remove_status_callback(self, callback):
        """Remove a status callback"""
        with self.lock:
            if callback in self.status_callbacks:
                self.status_callbacks.remove(callback)
    
    def record_inference_time(self, inference_time):
        """Record inference time for model predictions"""
        with self.lock:
            self.inference_times.append(inference_time)
            
            while len(self.inference_times) > self.window_size:
                self.inference_times.pop(0)
                
            self.current_status['inference_time'] = sum(self.inference_times) / len(self.inference_times)
            
    def record_fps(self, fps):
        """Record frames per second for UI rendering"""
        with self.lock:
            self.fps_history.append(fps)
            
            while len(self.fps_history) > self.window_size:
                self.fps_history.pop(0)
                
            self.current_status['fps'] = sum(self.fps_history) / len(self.fps_history)
    
    def _monitor_thread(self):
        """Thread function for monitoring performance metrics"""
        if not HAS_PSUTIL:
            logging.warning("psutil not available, performance monitoring limited")
            
        nvidia_smi_available = False
        try:
            if sys.platform.startswith('win'):
                import subprocess
                try:
                    subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1)
                    nvidia_smi_available = True
                    logging.info("NVIDIA GPU monitoring available")
                except (subprocess.SubprocessError, FileNotFoundError):
                    nvidia_smi_available = False
        except Exception:
            nvidia_smi_available = False
            
        while self.running:
            try:
                self._collect_metrics(nvidia_smi_available)
                
                self._analyze_performance()
                
                self._notify_status_callbacks()
                
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in performance monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self, nvidia_smi_available):
        """Collect current system performance metrics"""
        metrics = {}
        
        if HAS_PSUTIL:
            try:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(metrics['cpu_percent'])
                while len(self.cpu_usage) > self.window_size:
                    self.cpu_usage.pop(0)
                
                memory = psutil.virtual_memory()
                metrics['memory_percent'] = memory.percent
                self.memory_usage.append(metrics['memory_percent'])
                while len(self.memory_usage) > self.window_size:
                    self.memory_usage.pop(0)
                
                try:
                    io_counters = psutil.disk_io_counters()
                    if io_counters:
                        current_time = time.time()
                        current_read = io_counters.read_bytes
                        current_write = io_counters.write_bytes
                        
                        if hasattr(self, '_last_io_time'):
                            time_diff = current_time - self._last_io_time
                            if time_diff > 0:
                                read_rate = (current_read - self._last_read_bytes) / time_diff / (1024 * 1024)
                                write_rate = (current_write - self._last_write_bytes) / time_diff / (1024 * 1024)
                                
                                io_rate = read_rate + write_rate
                                metrics['disk_io'] = io_rate
                                self.disk_io_history.append(io_rate)
                                while len(self.disk_io_history) > self.window_size:
                                    self.disk_io_history.pop(0)
                        
                        self._last_io_time = current_time
                        self._last_read_bytes = current_read
                        self._last_write_bytes = current_write
                except Exception:
                    pass
            except Exception as e:
                logging.debug(f"Error collecting CPU/memory metrics: {e}")
        
        if nvidia_smi_available:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                     stdout=subprocess.PIPE, text=True, timeout=1)
                try:
                    gpu_percent = float(result.stdout.strip())
                    metrics['gpu_percent'] = gpu_percent
                    self.gpu_usage.append(gpu_percent)
                    while len(self.gpu_usage) > self.window_size:
                        self.gpu_usage.pop(0)
                except (ValueError, IndexError):
                    pass
            except Exception:
                pass
        
        with self.lock:
            self.metrics_history.append(metrics)
            while len(self.metrics_history) > self.window_size:
                self.metrics_history.pop(0)
            
            for key, value in metrics.items():
                self.current_status[key] = value
    
    def _analyze_performance(self):
        """Analyze performance metrics to detect degradation"""
        with self.lock:
            if len(self.metrics_history) < 3:
                return  # Not enough data yet
                
            cpu_avg = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            mem_avg = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            
            system_load = max(cpu_avg, mem_avg)
            
            if system_load > 85:
                self.current_status['system_load'] = 'critical'
            elif system_load > 70:
                self.current_status['system_load'] = 'high'
            else:
                self.current_status['system_load'] = 'normal'
                
            throttling = False
            
            if len(self.inference_times) >= 3:
                half = len(self.inference_times) // 2
                first_half_avg = sum(self.inference_times[:half]) / half
                second_half_avg = sum(self.inference_times[half:]) / (len(self.inference_times) - half)
                
                if second_half_avg > first_half_avg * 1.2:
                    throttling = True
                    
            if len(self.fps_history) >= 3:
                half = len(self.fps_history) // 2
                first_half_avg = sum(self.fps_history[:half]) / half
                second_half_avg = sum(self.fps_history[half:]) / (len(self.fps_history) - half)
                
                if second_half_avg < first_half_avg * 0.8:
                    throttling = True
            
            if self.cpu_usage and all(usage > 90 for usage in self.cpu_usage[-3:]):
                throttling = True
                
            self.current_status['throttling_detected'] = throttling
            self.degradation_detected = throttling
            
            self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        recommendations = []
        
        self.optimization_recommendations = []
        
        if self.cpu_usage and sum(self.cpu_usage) / len(self.cpu_usage) > 80:
            recommendations.append({
                'component': 'cpu',
                'severity': 'high',
                'action': 'reduce_processing',
                'description': 'Reduce frame processing rate or use lower resolution'
            })
            
        if self.memory_usage and sum(self.memory_usage) / len(self.memory_usage) > 85:
            recommendations.append({
                'component': 'memory',
                'severity': 'high',
                'action': 'reduce_memory',
                'description': 'Reduce image buffer size or disable caching'
            })
            
        if self.disk_io_history and sum(self.disk_io_history) / len(self.disk_io_history) > 50:
            recommendations.append({
                'component': 'disk',
                'severity': 'medium',
                'action': 'reduce_io',
                'description': 'Reduce logging frequency or file operations'
            })
            
        if self.inference_times and sum(self.inference_times) / len(self.inference_times) > 0.1:
            recommendations.append({
                'component': 'model',
                'severity': 'medium',
                'action': 'optimize_model',
                'description': 'Use lighter model or enable frame skipping'
            })
            
        if self.fps_history and sum(self.fps_history) / len(self.fps_history) < 15:
            recommendations.append({
                'component': 'rendering',
                'severity': 'medium',
                'action': 'optimize_ui',
                'description': 'Reduce UI complexity or rendering frequency'
            })
            
        self.optimization_recommendations = recommendations
    
    def _notify_status_callbacks(self):
        """Notify registered callbacks of status updates"""
        with self.lock:
            current_status = self.current_status.copy()
            current_status['recommendations'] = self.optimization_recommendations
            
            for callback in self.status_callbacks:
                try:
                    callback(current_status)
                except Exception as e:
                    logging.error(f"Error in performance status callback: {e}")
    
    def get_status(self):
        """Get current performance status and recommendations"""
        with self.lock:
            status = self.current_status.copy()
            status['recommendations'] = self.optimization_recommendations
            return status
    
    def should_reduce_workload(self):
        """Check if workload should be reduced based on current status"""
        with self.lock:
            return self.degradation_detected or self.current_status.get('system_load') == 'critical'
    
    def get_recommendations(self):
        """Get current optimization recommendations"""
        with self.lock:
            return self.optimization_recommendations.copy()
