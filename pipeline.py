#!/usr/bin/env python
# pipeline.py - Parallel processing pipeline implementation with work stealing

import threading
import queue
import time
import logging
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Any, Callable, Optional, Tuple, Union, Deque
import platform
import sys
import os
import gc
import traceback

# Windows-specific imports
if platform.system() == 'Windows':
    try:
        import win32api
        import win32process
        import win32con
        HAS_WIN32API = True
    except ImportError:
        HAS_WIN32API = False
else:
    HAS_WIN32API = False

# Import psutil for system monitoring if available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class PipelineStage:
    """
    A single stage in the processing pipeline that processes input data and produces output.
    Stages are connected to form a pipeline where output of one stage becomes input for the next.
    """
    
    def __init__(self, name: str, process_func: Callable, 
                 max_batch_size: int = 1, 
                 priority: int = 0,
                 timeout: float = 0.1,
                 cpu_affinity: Optional[List[int]] = None):
        """
        Initialize a pipeline stage
        
        Args:
            name: Stage name for identification
            process_func: Function to process data (input -> output)
            max_batch_size: Maximum number of items to process in a batch
            priority: CPU priority (higher value = higher priority, 0=normal)
            timeout: Timeout for queue operations
            cpu_affinity: List of CPU core IDs this stage should run on (None for no affinity)
        """
        self.name = name
        self.process_func = process_func
        self.max_batch_size = max(1, max_batch_size)
        self.priority = priority
        self.timeout = timeout
        self.cpu_affinity = cpu_affinity
        
        # Input and output queues
        self.input_queue = queue.Queue()
        self.output_queue = None  # Set by pipeline when connected
        
        # Statistics and status
        self.items_processed = 0
        self.batch_count = 0
        self.processing_times = deque(maxlen=100)  # Last 100 processing times
        self.queue_times = deque(maxlen=100)  # Last 100 queue waiting times
        self.stage_active = False
        self.running = False
        self.thread = None
        self.last_batch_time = 0
        
        # Identification
        self.id = id(self)
        self.thread_id = None
        
        # Backpressure mechanism
        self.backpressure_enabled = True
        self.max_queue_size = 100  # Maximum size before applying backpressure
        self.current_pressure = 0.0  # 0.0 = no pressure, 1.0 = max pressure
        self.wait_on_queue_full = True  # Slow down producer if consumer can't keep up
        
    def connect_output(self, next_stage_input_queue: queue.Queue) -> None:
        """Connect this stage's output to the next stage's input"""
        self.output_queue = next_stage_input_queue
        
    def start(self) -> None:
        """Start the stage processing thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._process_loop,
            name=f"Pipeline-{self.name}",
            daemon=True
        )
        self.thread.start()
        
    def stop(self) -> None:
        """Stop the stage processing thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
    def _process_loop(self) -> None:
        """Main processing loop for this stage"""
        # Record thread ID for management
        self.thread_id = threading.get_ident()
        
        # Set thread priority and affinity if supported
        self._set_thread_priority()
        self._set_thread_affinity()
        
        logging.info(f"Pipeline stage '{self.name}' started")
        
        while self.running:
            try:
                # Try to get a batch of items from the input queue
                batch = self._get_batch()
                
                if not batch:
                    time.sleep(0.001)  # Short sleep to prevent CPU spinning
                    continue
                
                # Mark as active during processing
                self.stage_active = True
                batch_size = len(batch)
                
                # Process the batch and measure time
                start_time = time.time()
                
                if batch_size == 1:
                    # Process single item
                    input_data, timestamp, seq_id = batch[0]
                    try:
                        result = self.process_func(input_data)
                        if result is not None and self.output_queue is not None:
                            self.output_queue.put((result, timestamp, seq_id))
                    except Exception as e:
                        logging.error(f"Error in pipeline stage '{self.name}': {e}")
                        traceback.print_exc()
                else:
                    # Process batch
                    input_batch = [item[0] for item in batch]
                    timestamps = [item[1] for item in batch]
                    seq_ids = [item[2] for item in batch]
                    
                    try:
                        results = self.process_func(input_batch)
                        if results is not None and self.output_queue is not None:
                            # Handle different result types
                            if isinstance(results, list) and len(results) == batch_size:
                                # Matching output batch size
                                for res, ts, sid in zip(results, timestamps, seq_ids):
                                    self.output_queue.put((res, ts, sid))
                            else:
                                # Single result for whole batch
                                self.output_queue.put((results, timestamps[0], seq_ids[0]))
                    except Exception as e:
                        logging.error(f"Error in pipeline stage '{self.name}' batch processing: {e}")
                        traceback.print_exc()
                
                # Record statistics
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                self.items_processed += batch_size
                self.batch_count += 1
                self.last_batch_time = time.time()
                
                # Update backpressure metrics
                self._update_backpressure()
                
                # Mark as inactive after processing
                self.stage_active = False
                
            except Exception as e:
                logging.error(f"Unexpected error in pipeline stage '{self.name}': {e}")
                self.stage_active = False
                time.sleep(0.1)  # Sleep longer on error
    
    def _update_backpressure(self) -> None:
        """Update backpressure metrics to help with flow control"""
        if not self.backpressure_enabled:
            return
            
        try:
            # Calculate current queue size as percentage of max
            queue_size = self.input_queue.qsize()
            queue_pressure = min(1.0, queue_size / self.max_queue_size)
            
            # Calculate processing pressure based on recent processing times
            if self.processing_times:
                avg_process_time = sum(self.processing_times) / len(self.processing_times)
                process_pressure = min(1.0, avg_process_time / 0.1)  # Normalize to 0-1 range (100ms as reference)
            else:
                process_pressure = 0.0
                
            # Combine pressures (70% weight to queue size, 30% to processing time)
            self.current_pressure = (0.7 * queue_pressure) + (0.3 * process_pressure)
        except Exception as e:
            logging.debug(f"Error updating backpressure: {e}")
    
    def _get_batch(self) -> List[Tuple[Any, float, int]]:
        """Get a batch of items from the input queue with timeout and backpressure awareness"""
        batch = []
        
        # Adjust timeout based on backpressure
        timeout = self.timeout
        if self.current_pressure > 0.8:
            # Increase timeout when under heavy pressure to allow system to recover
            timeout *= 2
        
        try:
            # Try to get first item with timeout
            first_item = self.input_queue.get(timeout=timeout)
            batch.append(first_item)
            
            # Try to get more items without blocking (up to max_batch_size)
            # Dynamic batch size based on backpressure
            current_max_batch = self.max_batch_size
            if self.current_pressure > 0.5:
                # Increase batch size when under pressure to process backlog more efficiently
                current_max_batch = min(current_max_batch * 2, 32)
            
            while len(batch) < current_max_batch:
                try:
                    item = self.input_queue.get_nowait()
                    batch.append(item)
                except queue.Empty:
                    break
                    
        except queue.Empty:
            # First get timed out - return empty batch
            return []
            
        return batch
    
    def _set_thread_priority(self) -> None:
        """Set thread priority based on the stage priority"""
        if not HAS_WIN32API:
            return
            
        try:
            thread_handle = win32api.GetCurrentThread()
            
            if self.priority > 1:
                # Higher than normal priority
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_ABOVE_NORMAL)
            elif self.priority > 0:
                # Normal priority but slightly higher
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_NORMAL)
            elif self.priority < 0:
                # Below normal priority
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_BELOW_NORMAL)
            else:
                # Normal priority
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_NORMAL)
        except Exception as e:
            logging.debug(f"Could not set thread priority: {e}")
    
    def _set_thread_affinity(self) -> None:
        """Set thread affinity to specific CPU cores if specified"""
        if not self.cpu_affinity or not HAS_PSUTIL:
            return
            
        try:
            # Get current process
            process = psutil.Process()
            
            # Get current thread ID
            if HAS_WIN32API:
                thread_id = win32api.GetCurrentThreadId()
            else:
                thread_id = self.thread_id
                
            # Set affinity for this thread
            if thread_id:
                for thread in process.threads():
                    if thread.id == thread_id:
                        # Convert affinity list to affinity mask
                        mask = 0
                        for core in self.cpu_affinity:
                            mask |= (1 << core)
                            
                        if HAS_WIN32API:
                            thread_handle = win32api.OpenThread(
                                win32con.THREAD_SET_INFORMATION, 
                                False, 
                                thread_id
                            )
                            win32process.SetThreadAffinityMask(thread_handle, mask)
                            win32api.CloseHandle(thread_handle)
                        else:
                            # Linux/macOS approach - not directly supported by psutil
                            # Would need to use os.sched_setaffinity but it requires thread PID
                            logging.debug(f"Thread affinity not fully supported on this platform")
                            
                        logging.debug(f"Set thread affinity for pipeline stage '{self.name}' to cores {self.cpu_affinity}")
                        break
        except Exception as e:
            logging.debug(f"Could not set thread affinity: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this stage"""
        avg_process_time = 0
        if self.processing_times:
            avg_process_time = sum(self.processing_times) / len(self.processing_times)
            
        try:
            queue_size = self.input_queue.qsize()
        except:
            queue_size = -1
            
        return {
            'name': self.name,
            'items_processed': self.items_processed,
            'batch_count': self.batch_count,
            'avg_process_time': avg_process_time,
            'active': self.stage_active,
            'max_batch_size': self.max_batch_size,
            'input_queue_size': queue_size,
            'priority': self.priority,
            'pressure': self.current_pressure
        }
        
    def put(self, data: Any, timestamp: float = None, seq_id: int = None) -> bool:
        """
        Add an item to this stage's input queue
        
        Args:
            data: The data to process
            timestamp: Optional timestamp for the data
            seq_id: Optional sequence ID for the data
            
        Returns:
            True if item was added to queue, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
            
        if self.wait_on_queue_full:
            # Blocking put with timeout
            try:
                self.input_queue.put((data, timestamp, seq_id), timeout=1.0)
                return True
            except queue.Full:
                return False
        else:
            # Non-blocking put
            try:
                self.input_queue.put_nowait((data, timestamp, seq_id))
                return True
            except queue.Full:
                return False


class Pipeline:
    """
    A processing pipeline with multiple stages running in parallel.
    Includes work stealing for load balancing between stages.
    """
    
    def __init__(self, name: str = "ProcessingPipeline", 
                 enable_work_stealing: bool = True, 
                 enable_batch_adjustment: bool = True, 
                 batch_adjustment_interval: float = 5.0,
                 enable_backpressure: bool = True,
                 cpu_distribution: Optional[Dict[str, List[int]]] = None):
        """
        Initialize a processing pipeline
        
        Args:
            name: Pipeline name for identification
            enable_work_stealing: Whether to enable work stealing between stages
            enable_batch_adjustment: Whether to enable dynamic batch size adjustment
            batch_adjustment_interval: How often to adjust batch sizes (seconds)
            enable_backpressure: Whether to enable backpressure mechanisms
            cpu_distribution: Optional mapping of stage names to CPU cores for affinity
        """
        self.name = name
        self.stages = []
        self.lock = threading.RLock()
        self.running = False
        self.sequence_counter = 0
        self.enable_work_stealing = enable_work_stealing
        self.enable_batch_adjustment = enable_batch_adjustment
        self.batch_adjustment_interval = batch_adjustment_interval
        self.enable_backpressure = enable_backpressure
        self.cpu_distribution = cpu_distribution or {}
        
        # Work stealing variables
        self._work_stealing_thread = None
        self._work_stealing_interval = 0.01  # 10ms check interval
        self._work_stealing_active = False
        self._last_steal_time = 0
        self._steal_count = 0
        self._steal_stats = defaultdict(int)
        
        # Lock-free work stealing using thread-local memory
        self._work_stealing_data = threading.local()
        
        # Batch adjustment variables
        self._batch_adjustment_thread = None
        self._last_batch_adjustment = 0
        
        # Backpressure variables
        self._backpressure_threshold = 0.8  # Threshold for applying backpressure
        self._backpressure_stats = defaultdict(float)
        
        # Determine optimal settings based on available CPUs
        self._detect_core_distribution()
        
    def _detect_core_distribution(self) -> None:
        """Determine optimal CPU core distribution based on system configuration"""
        if not HAS_PSUTIL or self.cpu_distribution:
            return  # Already specified or not available
            
        try:
            # Get physical core count and logical processor count
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            
            if not physical_cores or not logical_cores:
                return
                
            # If we have multiple physical cores, distribute workload
            if physical_cores >= 2:
                self.cpu_distribution = {
                    'default': list(range(logical_cores)),  # Default is all cores
                    'critical': list(range(physical_cores)),  # Critical stages on physical cores
                    'io': [physical_cores + i for i in range(min(2, logical_cores - physical_cores))],  # I/O on some logical cores
                    'compute': list(range(physical_cores))  # Compute-intensive on physical cores
                }
                
                logging.info(f"Detected {physical_cores} physical cores, {logical_cores} logical cores. "
                            f"Configured automatic CPU distribution.")
        except Exception as e:
            logging.debug(f"Could not determine optimal CPU distribution: {e}")
        
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline
        
        Args:
            stage: The PipelineStage to add
        """
        with self.lock:
            # Set CPU affinity based on stage name if specified
            if not stage.cpu_affinity:
                for pattern, cores in self.cpu_distribution.items():
                    if pattern in stage.name.lower() or pattern == 'default':
                        stage.cpu_affinity = cores
                        break
            
            # Enable backpressure if requested
            if self.enable_backpressure:
                stage.backpressure_enabled = True
            
            # Connect to previous stage if this isn't the first stage
            if self.stages:
                stage.connect_output(self.stages[-1].input_queue)
                
            self.stages.append(stage)
            
            logging.info(f"Added stage '{stage.name}' to pipeline '{self.name}'"
                        f"{' with CPU affinity: ' + str(stage.cpu_affinity) if stage.cpu_affinity else ''}")
    
    def create_stage(self, name: str, process_func: Callable, 
                     max_batch_size: int = 1, 
                     priority: int = 0,
                     cpu_affinity: Optional[List[int]] = None) -> PipelineStage:
        """
        Create and add a new stage to the pipeline
        
        Args:
            name: Stage name
            process_func: Processing function
            max_batch_size: Maximum batch size
            priority: Thread priority
            cpu_affinity: List of CPU cores this stage should run on
            
        Returns:
            The created PipelineStage
        """
        # Create the stage
        stage = PipelineStage(
            name=name,
            process_func=process_func,
            max_batch_size=max_batch_size,
            priority=priority,
            cpu_affinity=cpu_affinity
        )
        
        # Add it to the pipeline
        self.add_stage(stage)
        
        return stage
    
    def start(self) -> None:
        """Start the pipeline processing"""
        if self.running:
            return
            
        with self.lock:
            self.running = True
            
            # Start all stages
            for stage in self.stages:
                stage.start()
                
            logging.info(f"Started pipeline '{self.name}' with {len(self.stages)} stages")
            
            # Start work stealing if enabled
            if self.enable_work_stealing and len(self.stages) > 1:
                self._start_work_stealing()
                
            # Start batch adjustment if enabled
            if self.enable_batch_adjustment:
                self._start_batch_adjustment()
    
    def stop(self) -> None:
        """Stop the pipeline processing"""
        if not self.running:
            return
            
        with self.lock:
            self.running = False
            
            # Stop all stages
            for stage in self.stages:
                stage.stop()
                
            logging.info(f"Stopped pipeline '{self.name}'")
            
            # Wait for work stealing thread to finish
            if self._work_stealing_thread and self._work_stealing_thread.is_alive():
                self._work_stealing_thread.join(timeout=2.0)
                
            # Wait for batch adjustment thread to finish
            if self._batch_adjustment_thread and self._batch_adjustment_thread.is_alive():
                self._batch_adjustment_thread.join(timeout=2.0)
    
    def put(self, data: Any, timestamp: float = None, seq_id: int = None) -> bool:
        """
        Add data to the pipeline for processing
        
        Args:
            data: Data to process
            timestamp: Optional timestamp (defaults to current time)
            seq_id: Optional sequence ID (auto-generated if None)
            
        Returns:
            True if data was added, False otherwise
        """
        if not self.running or not self.stages:
            return False
            
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            if seq_id is None:
                self.sequence_counter += 1
                seq_id = self.sequence_counter
                
        return self.stages[0].put(data, timestamp, seq_id)
    
    def get(self, timeout: float = 0.1) -> Tuple[Any, float, int]:
        """
        Get processed data from the pipeline output
        
        Args:
            timeout: How long to wait for output
            
        Returns:
            Tuple of (data, timestamp, sequence_id) or None if timeout
        """
        if not self.running or not self.stages:
            return None
            
        try:
            # Get from the output queue of the last stage
            return self.stages[-1].output_queue.get(timeout=timeout)
        except (queue.Empty, AttributeError):
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all pipeline stages"""
        stats = {
            'name': self.name,
            'num_stages': len(self.stages),
            'running': self.running,
            'work_stealing': self.enable_work_stealing,
            'work_stealing_stats': dict(self._steal_stats),
            'backpressure': self.enable_backpressure,
            'backpressure_stats': dict(self._backpressure_stats),
            'stages': [stage.get_stats() for stage in self.stages]
        }
        return stats
    
    def _start_work_stealing(self) -> None:
        """Start the work stealing thread"""
        self._work_stealing_thread = threading.Thread(
            target=self._work_stealing_loop,
            name=f"{self.name}-WorkStealing",
            daemon=True
        )
        self._work_stealing_thread.start()
    
    def _work_stealing_loop(self) -> None:
        """Main loop for work stealing between stages"""
        while self.running:
            try:
                self._balance_work()
                time.sleep(self._work_stealing_interval)
            except Exception as e:
                logging.error(f"Error in work stealing thread: {e}")
                time.sleep(1.0)  # Sleep longer on error
    
    def _balance_work(self) -> None:
        """
        Balance work between pipeline stages
        This is the optimized work stealing algorithm that redistributes work from overloaded stages
        with fewer lock contentions
        """
        # Skip if fewer than 2 stages
        if len(self.stages) < 2:
            return
        
        # Avoid frequent work stealing operations
        current_time = time.time()
        if current_time - self._last_steal_time < 0.05:  # At most 20 steals per second
            return
            
        # Mark that work stealing is active to prevent concurrent steals
        if self._work_stealing_active:
            return
            
        self._work_stealing_active = True
        try:
            # Get queue sizes for all stages with a single pass
            queue_sizes = []
            total_work = 0
            max_work = 0
            max_stage_idx = -1
            
            for i, stage in enumerate(self.stages):
                try:
                    size = stage.input_queue.qsize()
                    queue_sizes.append((i, stage, size))
                    total_work += size
                    
                    # Track maximum work stage
                    if size > max_work:
                        max_work = size
                        max_stage_idx = i
                except Exception:
                    queue_sizes.append((i, stage, 0))
            
            # If no significant work, nothing to steal
            if total_work < 5 or max_work < 3:
                return
                
            # Calculate ideal work distribution
            avg_work = total_work / len(self.stages)
            
            # Sort stages by queue size (ascending)
            queue_sizes.sort(key=lambda x: x[2])
            
            # Only steal if there's a significant imbalance
            # Donor = stage with most work, recipient = stage with least work
            donor_idx, donor_stage, donor_size = queue_sizes[-1]
            recipient_idx, recipient_stage, recipient_size = queue_sizes[0]
            
            # Don't steal if donor and recipient are adjacent stages - that's normal pipeline flow
            if abs(donor_idx - recipient_idx) <= 1:
                # Check if there's another suitable recipient that's not adjacent
                for i in range(1, min(3, len(queue_sizes))):
                    alt_idx, alt_stage, alt_size = queue_sizes[i]
                    if abs(donor_idx - alt_idx) > 1:
                        recipient_idx, recipient_stage, recipient_size = alt_idx, alt_stage, alt_size
                        break
                else:
                    # No suitable recipient found
                    return
                    
            # Check if imbalance is significant enough to justify stealing
            imbalance = donor_size - recipient_size
            if imbalance < 3 or donor_size < avg_work * 1.5 or recipient_size > avg_work * 0.5:
                return
            
            # Calculate how many items to steal (up to half of the imbalance)
            steal_count = min(10, max(1, imbalance // 2))
            
            # Create thread-local container for items to steal
            if not hasattr(self._work_stealing_data, 'buffer'):
                self._work_stealing_data.buffer = []
            buffer = self._work_stealing_data.buffer
            buffer.clear()
            
            # Steal items with minimal locking
            stolen_items = 0
            for _ in range(steal_count):
                try:
                    item = donor_stage.input_queue.get_nowait()
                    buffer.append(item)
                    stolen_items += 1
                except queue.Empty:
                    break
            
            # Put stolen items into recipient queue
            for item in buffer:
                try:
                    recipient_stage.input_queue.put_nowait(item)
                except queue.Full:
                    # If recipient queue is full, put item back to donor
                    try:
                        donor_stage.input_queue.put_nowait(item)
                        stolen_items -= 1
                    except queue.Full:
                        # If donor queue is also full, this item is lost
                        # This should rarely happen, and the sequence ID allows
                        # the system to detect missing items if needed
                        logging.warning(f"Item lost during work stealing - both queues full")
            
            if stolen_items > 0:
                # Record statistics
                self._steal_count += stolen_items
                self._steal_stats[f"{donor_stage.name}->{recipient_stage.name}"] += stolen_items
                self._last_steal_time = current_time
                
                if stolen_items >= 3:
                    logging.debug(f"Work stealing: moved {stolen_items} items from '{donor_stage.name}' to '{recipient_stage.name}'")
        
        except Exception as e:
            logging.debug(f"Error during work stealing: {e}")
        finally:
            self._work_stealing_active = False

    def _start_batch_adjustment(self) -> None:
        """Start the batch size adjustment thread"""
        self._batch_adjustment_thread = threading.Thread(
            target=self._batch_adjustment_loop,
            name=f"{self.name}-BatchAdjust",
            daemon=True
        )
        self._batch_adjustment_thread.start()
        
    def _batch_adjustment_loop(self) -> None:
        """Main loop for adjusting batch sizes based on load"""
        while self.running:
            try:
                self._adjust_batch_sizes()
                time.sleep(self.batch_adjustment_interval)
            except Exception as e:
                logging.error(f"Error in batch adjustment thread: {e}")
                time.sleep(2.0)  # Sleep longer on error
                
    def _adjust_batch_sizes(self) -> None:
        """Dynamically adjust batch sizes based on real-time performance metrics"""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get system metrics
            cpu_load = psutil.cpu_percent(interval=None) / 100.0  # 0.0 to 1.0
            
            # Memory pressure as a factor (0.0 to 1.0)
            memory = psutil.virtual_memory()
            memory_pressure = memory.percent / 100.0
            
            # Combine CPU and memory into overall system load
            system_load = 0.7 * cpu_load + 0.3 * memory_pressure
            
            # Gather pipeline performance metrics
            pipeline_stats = self.get_stats()
            stage_stats = pipeline_stats.get('stages', [])
            
            # Calculate average processing times and queue depths
            avg_process_times = {}
            queue_depths = {}
            pressures = {}
            
            for stage in stage_stats:
                name = stage.get('name', '')
                avg_process_times[name] = stage.get('avg_process_time', 0)
                queue_depths[name] = stage.get('input_queue_size', 0)
                pressures[name] = stage.get('pressure', 0)
            
            # Update backpressure statistics
            if self.enable_backpressure:
                for name, pressure in pressures.items():
                    self._backpressure_stats[name] = pressure
                    
                # Calculate overall pipeline pressure
                if pressures:
                    self._backpressure_stats['overall'] = sum(pressures.values()) / len(pressures)
            
            # Check if we should apply backpressure
            self._update_backpressure(pressures)
            
            # Identify bottlenecks
            bottlenecks = []
            for stage in self.stages:
                stage_name = stage.name
                process_time = avg_process_times.get(stage_name, 0)
                queue_depth = queue_depths.get(stage_name, 0)
                
                # Mark as bottleneck if:
                # 1. Long processing time compared to others
                # 2. Deep input queue
                # 3. High pressure
                if (process_time > 0.01 and queue_depth > 5) or pressures.get(stage_name, 0) > 0.7:
                    bottlenecks.append((stage, process_time, queue_depth))
            
            # Prioritize adjustments based on severity
            bottlenecks.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            # Adjust batch sizes based on real-time conditions
            for stage in self.stages:
                current_batch_size = stage.max_batch_size
                stage_name = stage.name
                
                # Skip if not enough statistics yet
                if stage_name not in avg_process_times:
                    continue
                    
                process_time = avg_process_times[stage_name]
                queue_depth = queue_depths.get(stage_name, 0)
                pressure = pressures.get(stage_name, 0)
                
                # Default to no change
                new_batch_size = current_batch_size
                
                # Factor 1: System load
                if system_load > 0.85:  # Very high system load
                    # Reduce batch sizes to maintain responsiveness
                    load_factor = 0.8
                elif system_load > 0.7:  # High system load
                    load_factor = 0.9
                elif system_load < 0.3:  # Low system load
                    load_factor = 1.2
                elif system_load < 0.5:  # Moderate system load
                    load_factor = 1.1
                else:
                    load_factor = 1.0
                
                # Factor 2: Queue depth
                if queue_depth > 20:  # Very deep queue
                    depth_factor = 1.3  # Increase batch size to process backlog
                elif queue_depth > 10:
                    depth_factor = 1.2
                elif queue_depth < 2 and current_batch_size > 1:
                    depth_factor = 0.9  # Decrease batch size for low queue depth
                else:
                    depth_factor = 1.0
                
                # Factor 3: Processing time 
                if process_time < 0.001 and current_batch_size < 8:
                    # Very fast processing, can increase batch size
                    time_factor = 1.2
                elif process_time > 0.05 and current_batch_size > 1:
                    # Slow processing, decrease batch size
                    time_factor = 0.9
                else:
                    time_factor = 1.0
                
                # Factor 4: Stage pressure
                if pressure > 0.8:
                    pressure_factor = 0.8  # Reduce batch size under high pressure
                elif pressure > 0.6:
                    pressure_factor = 0.9
                elif pressure < 0.2 and current_batch_size < 8:
                    pressure_factor = 1.1  # Increase batch size under low pressure
                else:
                    pressure_factor = 1.0
                
                # Combine all factors with appropriate weights
                adjustment_factor = (
                    load_factor * 0.3 +
                    depth_factor * 0.3 +
                    time_factor * 0.2 +
                    pressure_factor * 0.2
                )
                
                # Apply adjustment
                new_batch_size = max(1, min(32, int(current_batch_size * adjustment_factor + 0.5)))
                
                # Don't change batch size too drastically
                if new_batch_size != current_batch_size:
                    # Limit changes to at most +/- 50%
                    if new_batch_size > current_batch_size:
                        new_batch_size = min(new_batch_size, int(current_batch_size * 1.5 + 0.5))
                    else:
                        new_batch_size = max(new_batch_size, max(1, int(current_batch_size * 0.75)))
                    
                    # Update batch size
                    stage.max_batch_size = new_batch_size
                    logging.debug(f"Adjusted batch size for {stage_name}: {current_batch_size} -> {new_batch_size}")
            
            # Record adjustment time
            self._last_batch_adjustment = time.time()
        
        except Exception as e:
            logging.error(f"Error adjusting batch sizes: {e}")
            
    def _update_backpressure(self, pressures: Dict[str, float]) -> None:
        """Update backpressure settings based on current pipeline state"""
        if not self.enable_backpressure:
            return
            
        try:
            # Calculate overall pipeline pressure
            if not pressures:
                return
                
            overall_pressure = sum(pressures.values()) / len(pressures)
            
            # Find stages under high pressure
            high_pressure_stages = []
            for stage in self.stages:
                stage_pressure = pressures.get(stage.name, 0)
                if stage_pressure > self._backpressure_threshold:
                    high_pressure_stages.append((stage, stage_pressure))
            
            # Sort by pressure (highest first)
            high_pressure_stages.sort(key=lambda x: x[1], reverse=True)
            
            # Apply backpressure mechanisms
            if high_pressure_stages:
                # Update earliest stages to wait on queue full
                for i, stage in enumerate(self.stages):
                    # First stage should always use queue limiting to prevent unbounded memory growth
                    if i == 0:
                        stage.wait_on_queue_full = True
                    else:
                        # Enable queue limiting on stages that feed into high pressure stages
                        for high_pressure_stage, _ in high_pressure_stages:
                            if i == self.stages.index(high_pressure_stage) - 1:
                                stage.wait_on_queue_full = True
                                break
                        else:
                            # For other stages, use blocking only under very high overall pressure
                            stage.wait_on_queue_full = overall_pressure > 0.9
            else:
                # No high pressure stages, keep first stage blocking always
                for i, stage in enumerate(self.stages):
                    stage.wait_on_queue_full = (i == 0) or (overall_pressure > 0.9)
        
        except Exception as e:
            logging.debug(f"Error updating backpressure: {e}")


class StreamProcessor:
    """
    Specialized pipeline for camera frame processing with frame skipping
    and adaptive processing based on system load.
    """
    
    def __init__(self, name: str = "FrameProcessor", max_queue_size: int = 10):
        """
        Initialize a stream processor
        
        Args:
            name: Processor name for identification
            max_queue_size: Maximum size of internal queues
        """
        self.name = name
        self.max_queue_size = max_queue_size
        self.pipeline = Pipeline(name, enable_work_stealing=True)
        self.running = False
        self.frame_count = 0
        self.last_output_time = 0
        self.last_input_time = 0
        self.frame_skip = 0  # How many frames to skip (adaptive)
        self.frames_to_skip = 0  # Counter for current skipping
        self.lock = threading.RLock()
        self.output_callbacks = []
        
    def add_stage(self, name: str, process_func: Callable, 
                 max_batch_size: int = 1, 
                 priority: int = 0) -> None:
        """Add a processing stage to the pipeline"""
        self.pipeline.create_stage(name, process_func, max_batch_size, priority)
        
    def start(self) -> None:
        """Start the stream processor"""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.pipeline.start()
            
            # Start the output handling thread
            self.output_thread = threading.Thread(
                target=self._handle_output,
                name=f"{self.name}-Output",
                daemon=True
            )
            self.output_thread.start()
            
            logging.info(f"Stream processor '{self.name}' started")
            
    def stop(self) -> None:
        """Stop the stream processor"""
        with self.lock:
            if not self.running:
                return
                
            self.running = False
            self.pipeline.stop()
            
            # Stop output thread
            if hasattr(self, 'output_thread') and self.output_thread.is_alive():
                self.output_thread.join(timeout=2.0)
                
            logging.info(f"Stream processor '{self.name}' stopped")
            
    def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Process a frame through the pipeline
        
        Args:
            frame: Input frame
            metadata: Optional metadata dict
            
        Returns:
            True if frame was accepted, False if skipped or queue full
        """
        if not self.running:
            return False
            
        with self.lock:
            self.frame_count += 1
            self.last_input_time = time.time()
            
            # Check if we should skip this frame
            if self.frames_to_skip > 0:
                self.frames_to_skip -= 1
                return False
                
            # Reset the skip counter
            self.frames_to_skip = self.frame_skip
            
            # Use sequence number from metadata or generate a new one
            seq_id = metadata.get('frame_number', self.frame_count) if metadata else self.frame_count
            timestamp = metadata.get('timestamp', self.last_input_time) if metadata else self.last_input_time
            
            # Queue the frame for processing
            return self.pipeline.put(frame, timestamp, seq_id)
            
    def _handle_output(self) -> None:
        """Thread that handles pipeline output and invokes callbacks"""
        while self.running:
            try:
                # Get result from pipeline with timeout
                result = self.pipeline.get(timeout=0.1)
                
                if result:
                    output_data, timestamp, seq_id = result
                    self.last_output_time = time.time()
                    
                    # Compute pipeline latency
                    latency = self.last_output_time - timestamp
                    
                    # Call all registered callbacks with the result
                    for callback in self.output_callbacks:
                        try:
                            callback(output_data, {'timestamp': timestamp, 
                                                  'seq_id': seq_id,
                                                  'latency': latency})
                        except Exception as e:
                            logging.error(f"Error in output callback: {e}")
                            
            except Exception as e:
                logging.error(f"Error handling pipeline output: {e}")
                time.sleep(0.1)
                
    def add_output_callback(self, callback: Callable[[Any, Dict[str, Any]], None]) -> None:
        """
        Add a callback to handle processed output
        
        Args:
            callback: Function to call with results (data, metadata)
        """
        if callback not in self.output_callbacks:
            self.output_callbacks.append(callback)
            
    def remove_output_callback(self, callback: Callable) -> None:
        """Remove a previously registered output callback"""
        if callback in self.output_callbacks:
            self.output_callbacks.remove(callback)
            
    def set_frame_skip(self, skip_frames: int) -> None:
        """
        Set how many frames to skip between processing
        
        Args:
            skip_frames: Number of frames to skip (0 = process all frames)
        """
        with self.lock:
            self.frame_skip = max(0, skip_frames)
            self.frames_to_skip = 0  # Reset skip counter
            
    def adjust_skip_rate(self, latency: float, target_latency: float = 0.1) -> None:
        """
        Automatically adjust frame skip rate based on processing latency
        
        Args:
            latency: Current processing latency
            target_latency: Target latency to maintain
        """
        with self.lock:
            if latency > target_latency * 1.5:
                # Latency too high, skip more frames
                self.frame_skip = min(self.frame_skip + 1, 5)
            elif latency < target_latency * 0.8 and self.frame_skip > 0:
                # Latency low enough, can process more frames
                self.frame_skip = max(self.frame_skip - 1, 0)
                
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stream processor"""
        stats = {
            'name': self.name,
            'running': self.running,
            'frame_count': self.frame_count,
            'frame_skip': self.frame_skip,
            'latency': (self.last_output_time - self.last_input_time) if self.last_output_time else 0,
            'pipeline': self.pipeline.get_stats() if self.pipeline else {}
        }
        return stats 