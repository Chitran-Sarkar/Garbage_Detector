#!/usr/bin/env python
# shared_memory.py - Shared memory implementation for inter-process communication

import logging
import os
import sys
import numpy as np
import threading
import time
import platform
import tempfile
import uuid
import mmap
import ctypes
import struct
import pickle
import hashlib
from typing import Optional, Tuple, Dict, Any, List, Union

# Windows-specific imports for memory mapping
if platform.system() == 'Windows':
    try:
        import win32api
        import win32con
        import win32file
        HAS_WIN32API = True
    except ImportError:
        HAS_WIN32API = False
else:
    HAS_WIN32API = False

class SharedMemoryBuffer:
    """
    A shared memory buffer for efficient data sharing between processes.
    Uses memory mapping for zero-copy operations.
    """
    
    def __init__(self, name=None, size=None, shape=None, dtype=np.uint8, create=True):
        """
        Initialize a shared memory buffer.
        
        Args:
            name: Unique name for the shared memory segment (auto-generated if None)
            size: Size of the buffer in bytes (required if create=True)
            shape: Shape of the numpy array to be stored (required if create=True)
            dtype: Data type of the numpy array
            create: Whether to create a new shared memory segment or open existing
        """
        self.name = name or f"wastemgr_shm_{uuid.uuid4().hex}"
        self.dtype = dtype
        self.shape = shape
        self.buffer = None
        self.mmap_obj = None
        self.lock = threading.RLock()
        self.file_handle = None
        self.is_owner = create
        
        # Size calculation
        if create:
            if shape is None:
                raise ValueError("Shape must be provided when creating shared memory")
                
            self.shape = tuple(shape)
            self.size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            
            if size is not None and size > self.size:
                self.size = size  # Use the larger size if explicitly provided
        else:
            # When opening existing buffer, size will be determined from the file
            if name is None:
                raise ValueError("Name must be provided when opening existing shared memory")
            self.size = size
        
        # Initialize the shared memory
        self._init_shared_memory()
    
    def _init_shared_memory(self):
        """Initialize the shared memory segment based on platform"""
        try:
            if platform.system() == 'Windows':
                self._init_windows_shared_memory()
            else:
                self._init_posix_shared_memory()
        except Exception as e:
            logging.error(f"Failed to initialize shared memory: {e}")
            raise
    
    def _init_windows_shared_memory(self):
        """Initialize shared memory using Windows memory-mapped files"""
        if not HAS_WIN32API:
            raise ImportError("pywin32 is required for Windows shared memory")
        
        try:
            if self.is_owner:
                # Create a new memory-mapped file
                self.file_handle = win32file.CreateFileMapping(
                    win32file.INVALID_HANDLE_VALUE,  # Use paging file
                    None,  # Default security
                    win32con.PAGE_READWRITE,  # Read/write access
                    0,  # Maximum size high (0 for size < 4GB)
                    self.size,  # Maximum size low
                    self.name  # Name of the mapping
                )
                
                if not self.file_handle:
                    raise RuntimeError(f"Failed to create file mapping: {win32api.GetLastError()}")
                
                # Check if the mapping already existed
                if win32api.GetLastError() == win32con.ERROR_ALREADY_EXISTS:
                    logging.warning(f"Shared memory segment {self.name} already exists")
                    
                logging.info(f"Created shared memory segment {self.name} with size {self.size} bytes")
            else:
                # Open existing memory-mapped file
                self.file_handle = win32file.OpenFileMapping(
                    win32con.FILE_MAP_ALL_ACCESS,  # Read/write access
                    False,  # Don't inherit handle
                    self.name  # Name of the mapping
                )
                
                if not self.file_handle:
                    raise RuntimeError(f"Failed to open file mapping {self.name}: {win32api.GetLastError()}")
                
                logging.info(f"Opened existing shared memory segment {self.name}")
            
            # Map the file into memory
            ptr = win32file.MapViewOfFile(
                self.file_handle,
                win32con.FILE_MAP_ALL_ACCESS,  # Read/write access
                0,  # Offset high
                0,  # Offset low
                self.size  # Number of bytes to map
            )
            
            if not ptr:
                raise RuntimeError(f"Failed to map view of file: {win32api.GetLastError()}")
            
            # Create a memory map object
            self.mmap_obj = mmap.mmap(-1, self.size, tagname=self.name)
            
            # Create numpy array from shared memory
            if self.shape is not None:
                self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.mmap_obj)
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing Windows shared memory: {e}")
    
    def _init_posix_shared_memory(self):
        """Initialize shared memory using POSIX shared memory (for Unix-like systems)"""
        try:
            # Use the temp directory for shared memory
            temp_dir = tempfile.gettempdir()
            self.shm_path = os.path.join(temp_dir, self.name)
            
            if self.is_owner:
                # Create new file with required size
                with open(self.shm_path, 'wb') as f:
                    f.write(b'\0' * self.size)
                    
                logging.info(f"Created shared memory file at {self.shm_path} with size {self.size} bytes")
            else:
                if not os.path.exists(self.shm_path):
                    raise FileNotFoundError(f"Shared memory file {self.shm_path} not found")
                    
                # When opening, determine size from the file
                if self.size is None:
                    self.size = os.path.getsize(self.shm_path)
                
                logging.info(f"Opened existing shared memory file at {self.shm_path}")
            
            # Memory map the file
            fd = os.open(self.shm_path, os.O_RDWR)
            self.mmap_obj = mmap.mmap(fd, self.size, mmap.MAP_SHARED)
            os.close(fd)
            
            # Create numpy array from shared memory
            if self.shape is not None:
                self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.mmap_obj)
                
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing POSIX shared memory: {e}")
    
    def get_array(self) -> np.ndarray:
        """Get the numpy array backed by shared memory"""
        with self.lock:
            if self.buffer is None:
                raise RuntimeError("Shared memory buffer not initialized")
            return self.buffer
    
    def copy_from_array(self, arr: np.ndarray) -> bool:
        """Copy data from a numpy array into the shared memory buffer"""
        with self.lock:
            if self.buffer is None:
                raise RuntimeError("Shared memory buffer not initialized")
                
            if arr.shape != self.shape:
                raise ValueError(f"Array shape {arr.shape} does not match buffer shape {self.shape}")
                
            # Copy array data to shared memory
            np.copyto(self.buffer, arr)
            return True
    
    def copy_to_array(self, arr: np.ndarray) -> bool:
        """Copy data from the shared memory buffer to a numpy array"""
        with self.lock:
            if self.buffer is None:
                raise RuntimeError("Shared memory buffer not initialized")
                
            if arr.shape != self.shape:
                raise ValueError(f"Array shape {arr.shape} does not match buffer shape {self.shape}")
                
            # Copy shared memory data to array
            np.copyto(arr, self.buffer)
            return True
    
    def cleanup(self):
        """Clean up shared memory resources"""
        with self.lock:
            # Clear references to numpy array
            self.buffer = None
            
            # Close memory map
            if self.mmap_obj is not None:
                try:
                    self.mmap_obj.close()
                except Exception as e:
                    logging.debug(f"Error closing memory map: {e}")
                self.mmap_obj = None
            
            # Windows-specific cleanup
            if platform.system() == 'Windows' and HAS_WIN32API:
                if self.file_handle is not None:
                    try:
                        win32file.CloseHandle(self.file_handle)
                    except Exception as e:
                        logging.debug(f"Error closing file handle: {e}")
                    self.file_handle = None
            
            # POSIX-specific cleanup
            elif self.is_owner and platform.system() != 'Windows':
                try:
                    if os.path.exists(self.shm_path):
                        os.unlink(self.shm_path)
                except Exception as e:
                    logging.debug(f"Error removing shared memory file: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class SharedFrameBuffer:
    """
    A specialized shared memory buffer for camera frames with metadata.
    Implements a ring buffer structure for producer-consumer pattern.
    """
    
    def __init__(self, buffer_size=3, width=1280, height=720, channels=3, dtype=np.uint8):
        """
        Initialize a shared frame buffer with multiple slots.
        
        Args:
            buffer_size: Number of frame slots in the ring buffer
            width: Frame width in pixels
            height: Frame height in pixels
            channels: Number of color channels (3 for RGB/BGR)
            dtype: Data type of the frame pixels
        """
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        
        # Calculate frame size in bytes
        self.frame_size = width * height * channels * np.dtype(dtype).itemsize
        
        # Metadata size for each frame (timestamp, frame number, status flags)
        self.metadata_size = 64  # Reserve 64 bytes for metadata
        
        # Total size required for the entire buffer
        total_size = (self.frame_size + self.metadata_size) * buffer_size
        
        # Calculate shape for the entire buffer
        self.buffer_shape = (buffer_size, height, width, channels)
        
        # Create shared memory buffer
        self.name = f"frame_buffer_{uuid.uuid4().hex[:8]}"
        self.shm = SharedMemoryBuffer(name=self.name, size=total_size, 
                                      shape=self.buffer_shape, dtype=dtype)
        
        # Create separate metadata buffer
        self.metadata_name = f"metadata_{uuid.uuid4().hex[:8]}"
        self.metadata_shape = (buffer_size, 16)  # 16 elements per frame
        self.metadata_shm = SharedMemoryBuffer(name=self.metadata_name,
                                          shape=self.metadata_shape, dtype=np.float64)
        
        # Indexes for ring buffer
        self.write_index = 0
        self.read_index = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logging.info(f"Created shared frame buffer with {buffer_size} slots, " + 
                     f"frame size: {width}x{height}x{channels}")
    
    def write_frame(self, frame: np.ndarray, timestamp: float, frame_number: int) -> int:
        """
        Write a frame to the buffer.
        
        Args:
            frame: Input frame (height x width x channels)
            timestamp: Frame timestamp
            frame_number: Frame sequence number
            
        Returns:
            Index where the frame was written
        """
        with self.lock:
            # Get the buffer arrays
            buffer_array = self.shm.get_array()
            metadata_array = self.metadata_shm.get_array()
            
            # Get current write position
            write_pos = self.write_index
            
            # Write the frame data
            buffer_array[write_pos] = frame
            
            # Write metadata
            metadata_array[write_pos, 0] = timestamp
            metadata_array[write_pos, 1] = float(frame_number)
            metadata_array[write_pos, 2] = 1.0  # Frame valid flag
            
            # Update write index (circular buffer)
            self.write_index = (self.write_index + 1) % self.buffer_size
            
            return write_pos
    
    def read_frame(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read the next available frame from the buffer.
        
        Returns:
            Tuple of (frame, metadata_dict)
        """
        with self.lock:
            # Get the buffer arrays
            buffer_array = self.shm.get_array()
            metadata_array = self.metadata_shm.get_array()
            
            # Check if there's a valid frame at the read position
            if metadata_array[self.read_index, 2] != 1.0:
                return None, None
            
            # Get the frame data (make a copy to avoid shared memory issues)
            frame = buffer_array[self.read_index].copy()
            
            # Get metadata
            timestamp = metadata_array[self.read_index, 0]
            frame_number = int(metadata_array[self.read_index, 1])
            
            # Mark frame as read
            metadata_array[self.read_index, 2] = 0.0
            
            # Update read index (circular buffer)
            self.read_index = (self.read_index + 1) % self.buffer_size
            
            metadata = {
                'timestamp': timestamp,
                'frame_number': frame_number
            }
            
            return frame, metadata
    
    def peek_latest_frame(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get the most recent frame without advancing the read pointer.
        
        Returns:
            Tuple of (frame, metadata_dict)
        """
        with self.lock:
            # Get the buffer arrays
            buffer_array = self.shm.get_array()
            metadata_array = self.metadata_shm.get_array()
            
            # Latest frame is at (write_index - 1)
            latest_idx = (self.write_index - 1) % self.buffer_size
            
            # Check if there's a valid frame
            if metadata_array[latest_idx, 2] != 1.0:
                return None, None
            
            # Get the frame data (make a copy to avoid shared memory issues)
            frame = buffer_array[latest_idx].copy()
            
            # Get metadata
            timestamp = metadata_array[latest_idx, 0]
            frame_number = int(metadata_array[latest_idx, 1])
            
            metadata = {
                'timestamp': timestamp,
                'frame_number': frame_number
            }
            
            return frame, metadata
    
    def get_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        with self.lock:
            return {
                'buffer_size': self.buffer_size,
                'frame_dimensions': (self.height, self.width, self.channels),
                'write_index': self.write_index,
                'read_index': self.read_index,
                'frame_size_bytes': self.frame_size,
                'total_size_bytes': self.shm.size
            }
    
    def cleanup(self):
        """Clean up shared memory resources"""
        self.shm.cleanup()
        self.metadata_shm.cleanup()
        
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class LargeSharedMemoryBuffer:
    """
    An optimized shared memory buffer for larger data structures.
    Uses memory mapping with advanced features for efficient handling of large arrays and objects.
    
    Features:
    - Optimized for larger data structures (>100MB)
    - Support for chunked access to avoid memory spikes
    - Metadata section for storing information about the data
    - Support for serialized objects in addition to numpy arrays
    """
    
    def __init__(self, name=None, size_mb=None, create=True):
        """
        Initialize a large shared memory buffer.
        
        Args:
            name: Unique name for the shared memory segment (auto-generated if None)
            size_mb: Size of the buffer in megabytes (required if create=True)
            create: Whether to create a new shared memory segment or open existing
        """
        self.name = name or f"wastemgr_large_shm_{uuid.uuid4().hex}"
        self.is_owner = create
        self.lock = threading.RLock()
        self.mmap_obj = None
        self.file_handle = None
        self.header_size = 4096  # Reserved space for metadata
        
        # Metadata structure in the header:
        # - Magic number (8 bytes)
        # - Format version (4 bytes)
        # - Data type code (4 bytes)
        # - Data size (8 bytes)
        # - Shape info (variable)
        # - Checksum (32 bytes)
        # - Timestamp (8 bytes)
        
        self.magic = b'LGSHMBUF'
        self.version = 1
        
        # Initialize size
        if create:
            if size_mb is None:
                raise ValueError("Size must be provided when creating large shared memory")
            self.size = int(size_mb * 1024 * 1024)  # Convert MB to bytes
        else:
            if name is None:
                raise ValueError("Name must be provided when opening existing shared memory")
            self.size = None  # Will be determined from the file
        
        # Initialize the shared memory
        self._init_shared_memory()
    
    def _init_shared_memory(self):
        """Initialize the shared memory segment based on platform"""
        try:
            if platform.system() == 'Windows':
                self._init_windows_shared_memory()
            else:
                self._init_posix_shared_memory()
        except Exception as e:
            logging.error(f"Failed to initialize large shared memory: {e}")
            raise
    
    def _init_windows_shared_memory(self):
        """Initialize shared memory using Windows memory-mapped files"""
        if not HAS_WIN32API:
            raise ImportError("pywin32 is required for Windows shared memory")
        
        try:
            if self.is_owner:
                # Create a new memory-mapped file
                self.file_handle = win32file.CreateFileMapping(
                    win32file.INVALID_HANDLE_VALUE,  # Use paging file
                    None,  # Default security
                    win32con.PAGE_READWRITE,  # Read/write access
                    0,  # Maximum size high (0 for size < 4GB)
                    self.size,  # Maximum size low
                    self.name  # Name of the mapping
                )
                
                if not self.file_handle:
                    raise RuntimeError(f"Failed to create file mapping: {win32api.GetLastError()}")
                
                # Check if the mapping already existed
                if win32api.GetLastError() == win32con.ERROR_ALREADY_EXISTS:
                    logging.warning(f"Large shared memory segment {self.name} already exists")
                    
                logging.info(f"Created large shared memory segment {self.name} with size {self.size} bytes")
                
                # Initialize header
                self._init_header()
            else:
                # Open existing memory-mapped file
                self.file_handle = win32file.OpenFileMapping(
                    win32con.FILE_MAP_ALL_ACCESS,  # Read/write access
                    False,  # Don't inherit handle
                    self.name  # Name of the mapping
                )
                
                if not self.file_handle:
                    raise RuntimeError(f"Failed to open file mapping {self.name}: {win32api.GetLastError()}")
                
                logging.info(f"Opened existing large shared memory segment {self.name}")
            
            # Map the file into memory
            ptr = win32file.MapViewOfFile(
                self.file_handle,
                win32con.FILE_MAP_ALL_ACCESS,  # Read/write access
                0,  # Offset high
                0,  # Offset low
                0  # Map the entire file
            )
            
            if not ptr:
                raise RuntimeError(f"Failed to map view of file: {win32api.GetLastError()}")
            
            # Create a memory map object (size will be auto-detected if opening existing)
            self.mmap_obj = mmap.mmap(-1, 0 if self.size is None else self.size, tagname=self.name)
            
            if self.size is None:
                # Determine size from existing mapping
                self.size = self.mmap_obj.size()
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing Windows large shared memory: {e}")
    
    def _init_posix_shared_memory(self):
        """Initialize shared memory using POSIX shared memory (for Unix-like systems)"""
        try:
            # Use the temp directory for shared memory
            temp_dir = tempfile.gettempdir()
            self.shm_path = os.path.join(temp_dir, self.name)
            
            if self.is_owner:
                # Create new file with required size
                with open(self.shm_path, 'wb') as f:
                    f.write(b'\0' * self.size)
                    
                logging.info(f"Created large shared memory file at {self.shm_path} with size {self.size} bytes")
                
                # Open for memory mapping
                fd = os.open(self.shm_path, os.O_RDWR)
                self.mmap_obj = mmap.mmap(fd, self.size, mmap.MAP_SHARED)
                os.close(fd)
                
                # Initialize header
                self._init_header()
            else:
                if not os.path.exists(self.shm_path):
                    raise FileNotFoundError(f"Large shared memory file {self.shm_path} not found")
                    
                # When opening, determine size from the file
                self.size = os.path.getsize(self.shm_path)
                logging.info(f"Opened existing large shared memory file at {self.shm_path} with size {self.size}")
                
                # Memory map the file
                fd = os.open(self.shm_path, os.O_RDWR)
                self.mmap_obj = mmap.mmap(fd, self.size, mmap.MAP_SHARED)
                os.close(fd)
                
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing POSIX large shared memory: {e}")
    
    def _init_header(self):
        """Initialize the header section with metadata"""
        with self.lock:
            # Seek to the beginning
            self.mmap_obj.seek(0)
            
            # Write magic number
            self.mmap_obj.write(self.magic)
            
            # Write version
            self.mmap_obj.write(struct.pack('!I', self.version))
            
            # Write zeros for the rest of the header
            self.mmap_obj.write(b'\0' * (self.header_size - 12))
    
    def _update_header_metadata(self, data_type_code, data_size, shape=None):
        """Update the header with metadata about the stored data"""
        with self.lock:
            # Seek past magic and version
            self.mmap_obj.seek(12)
            
            # Write data type code
            self.mmap_obj.write(struct.pack('!I', data_type_code))
            
            # Write data size
            self.mmap_obj.write(struct.pack('!Q', data_size))
            
            # Write shape info if provided
            if shape is not None:
                shape_str = ','.join(str(dim) for dim in shape)
                shape_bytes = shape_str.encode('utf-8')
                self.mmap_obj.write(struct.pack('!H', len(shape_bytes)))
                self.mmap_obj.write(shape_bytes)
            else:
                self.mmap_obj.write(struct.pack('!H', 0))  # No shape info
                
            # Update timestamp
            self.mmap_obj.seek(self.header_size - 8)
            self.mmap_obj.write(struct.pack('!d', time.time()))
    
    def _compute_checksum(self, data):
        """Compute MD5 checksum of data"""
        if isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest().encode('utf-8')
        else:
            return hashlib.md5(data).hexdigest().encode('utf-8')
    
    def _read_header_metadata(self):
        """Read metadata from the header"""
        with self.lock:
            # Seek to the beginning
            self.mmap_obj.seek(0)
            
            # Read and verify magic number
            magic = self.mmap_obj.read(8)
            if magic != self.magic:
                raise ValueError(f"Invalid magic number in large shared memory header: {magic}")
            
            # Read version
            version = struct.unpack('!I', self.mmap_obj.read(4))[0]
            if version != self.version:
                logging.warning(f"Version mismatch in large shared memory header: expected {self.version}, got {version}")
            
            # Read data type code
            data_type_code = struct.unpack('!I', self.mmap_obj.read(4))[0]
            
            # Read data size
            data_size = struct.unpack('!Q', self.mmap_obj.read(8))[0]
            
            # Read shape info
            shape_len = struct.unpack('!H', self.mmap_obj.read(2))[0]
            shape = None
            if shape_len > 0:
                shape_str = self.mmap_obj.read(shape_len).decode('utf-8')
                if shape_str:
                    shape = tuple(int(dim) for dim in shape_str.split(','))
            
            # Read timestamp
            self.mmap_obj.seek(self.header_size - 8)
            timestamp = struct.unpack('!d', self.mmap_obj.read(8))[0]
            
            return {
                'data_type_code': data_type_code,
                'data_size': data_size,
                'shape': shape,
                'timestamp': timestamp
            }
    
    def write_array(self, arr: np.ndarray) -> bool:
        """Write a numpy array to shared memory with optimized handling for large arrays"""
        with self.lock:
            if self.mmap_obj is None:
                raise RuntimeError("Shared memory buffer not initialized")
            
            data_size = arr.nbytes
            if data_size + self.header_size > self.size:
                raise ValueError(f"Array size ({data_size} bytes) exceeds available space ({self.size - self.header_size} bytes)")
            
            # Update header with metadata
            data_type_code = 1  # Code for numpy array
            self._update_header_metadata(data_type_code, data_size, arr.shape)
            
            # Compute and store checksum
            checksum = self._compute_checksum(arr)
            self.mmap_obj.seek(self.header_size - 40)
            self.mmap_obj.write(checksum)
            
            # Write array data after header
            self.mmap_obj.seek(self.header_size)
            self.mmap_obj.write(arr.tobytes())
            
            return True
    
    def read_array(self, dtype=None, shape=None) -> np.ndarray:
        """Read a numpy array from shared memory with verification"""
        with self.lock:
            if self.mmap_obj is None:
                raise RuntimeError("Shared memory buffer not initialized")
            
            # Read metadata from header
            metadata = self._read_header_metadata()
            
            if metadata['data_type_code'] != 1:
                raise TypeError(f"Stored data is not a numpy array (type code: {metadata['data_type_code']})")
            
            # Use provided dtype/shape or from metadata
            if dtype is None:
                # We don't store dtype in metadata, so this is a limitation
                dtype = np.uint8
                logging.warning("No dtype provided for read_array, assuming uint8")
                
            if shape is None:
                if metadata['shape'] is None:
                    raise ValueError("Shape information missing from shared memory and not provided")
                shape = metadata['shape']
            
            # Read data
            self.mmap_obj.seek(self.header_size)
            data = self.mmap_obj.read(metadata['data_size'])
            
            # Read stored checksum
            self.mmap_obj.seek(self.header_size - 40)
            stored_checksum = self.mmap_obj.read(32)
            
            # Verify checksum
            actual_checksum = hashlib.md5(data).hexdigest().encode('utf-8')
            if stored_checksum != actual_checksum:
                logging.warning("Checksum mismatch in shared memory data")
            
            # Create array from buffer
            arr = np.frombuffer(data, dtype=dtype).reshape(shape)
            return arr
    
    def write_object(self, obj: Any) -> bool:
        """Write a Python object to shared memory using pickle serialization"""
        with self.lock:
            if self.mmap_obj is None:
                raise RuntimeError("Shared memory buffer not initialized")
            
            # Serialize object
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            data_size = len(data)
            
            if data_size + self.header_size > self.size:
                raise ValueError(f"Serialized object size ({data_size} bytes) exceeds available space ({self.size - self.header_size} bytes)")
            
            # Update header with metadata
            data_type_code = 2  # Code for pickled object
            self._update_header_metadata(data_type_code, data_size)
            
            # Compute and store checksum
            checksum = self._compute_checksum(data)
            self.mmap_obj.seek(self.header_size - 40)
            self.mmap_obj.write(checksum)
            
            # Write serialized data after header
            self.mmap_obj.seek(self.header_size)
            self.mmap_obj.write(data)
            
            return True
    
    def read_object(self) -> Any:
        """Read a Python object from shared memory"""
        with self.lock:
            if self.mmap_obj is None:
                raise RuntimeError("Shared memory buffer not initialized")
            
            # Read metadata from header
            metadata = self._read_header_metadata()
            
            if metadata['data_type_code'] != 2:
                raise TypeError(f"Stored data is not a pickled object (type code: {metadata['data_type_code']})")
            
            # Read data
            self.mmap_obj.seek(self.header_size)
            data = self.mmap_obj.read(metadata['data_size'])
            
            # Read stored checksum
            self.mmap_obj.seek(self.header_size - 40)
            stored_checksum = self.mmap_obj.read(32)
            
            # Verify checksum
            actual_checksum = hashlib.md5(data).hexdigest().encode('utf-8')
            if stored_checksum != actual_checksum:
                logging.warning("Checksum mismatch in shared memory data")
            
            # Deserialize object
            obj = pickle.loads(data)
            return obj
    
    def get_chunks(self, chunk_size_mb=10, offset=0, limit=None):
        """
        Generator that yields chunks of the shared memory buffer.
        Useful for processing very large arrays without loading entire data into memory.
        
        Args:
            chunk_size_mb: Size of each chunk in megabytes
            offset: Starting offset in bytes (relative to data section, after header)
            limit: Maximum number of bytes to read (None for all available data)
            
        Yields:
            Chunks of bytes from the shared memory buffer
        """
        chunk_size = chunk_size_mb * 1024 * 1024
        
        with self.lock:
            if self.mmap_obj is None:
                raise RuntimeError("Shared memory buffer not initialized")
            
            # Read metadata to determine data size
            metadata = self._read_header_metadata()
            data_size = metadata['data_size']
            
            # Apply limit if specified
            if limit is not None:
                data_size = min(data_size, limit)
            
            # Start from the data section
            pos = self.header_size + offset
            remaining = data_size - offset
            
            while remaining > 0:
                # Calculate size of this chunk
                size = min(chunk_size, remaining)
                
                # Read chunk
                self.mmap_obj.seek(pos)
                chunk = self.mmap_obj.read(size)
                
                yield chunk
                
                # Update position and remaining bytes
                pos += size
                remaining -= size
    
    def cleanup(self):
        """Clean up shared memory resources"""
        with self.lock:
            # Close memory map
            if self.mmap_obj is not None:
                try:
                    self.mmap_obj.close()
                except Exception as e:
                    logging.debug(f"Error closing memory map: {e}")
                self.mmap_obj = None
            
            # Windows-specific cleanup
            if platform.system() == 'Windows' and HAS_WIN32API:
                if self.file_handle is not None:
                    try:
                        win32file.CloseHandle(self.file_handle)
                    except Exception as e:
                        logging.debug(f"Error closing file handle: {e}")
                    self.file_handle = None
            
            # POSIX-specific cleanup
            elif self.is_owner and platform.system() != 'Windows':
                try:
                    if hasattr(self, 'shm_path') and os.path.exists(self.shm_path):
                        os.unlink(self.shm_path)
                except Exception as e:
                    logging.debug(f"Error removing shared memory file: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Helper functions for external access
_SHARED_BUFFERS = {}

def create_shared_buffer(name, size=None, shape=None, dtype=np.uint8):
    """Create a shared memory buffer and register it globally"""
    buffer = SharedMemoryBuffer(name=name, size=size, shape=shape, dtype=dtype)
    _SHARED_BUFFERS[name] = buffer
    return buffer

def get_shared_buffer(name):
    """Get a registered shared memory buffer by name"""
    if name in _SHARED_BUFFERS:
        return _SHARED_BUFFERS[name]
    
    # Try to open existing buffer
    try:
        buffer = SharedMemoryBuffer(name=name, create=False)
        _SHARED_BUFFERS[name] = buffer
        return buffer
    except Exception as e:
        logging.error(f"Failed to get shared buffer {name}: {e}")
        return None

def cleanup_shared_buffers():
    """Clean up all registered shared memory buffers"""
    for name, buffer in list(_SHARED_BUFFERS.items()):
        try:
            buffer.cleanup()
        except Exception as e:
            logging.debug(f"Error cleaning up shared buffer {name}: {e}")
    _SHARED_BUFFERS.clear() 