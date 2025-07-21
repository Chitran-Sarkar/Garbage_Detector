#!/usr/bin/env python
# adaptive_thread_pool.py - Work-stealing thread pool implementation

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import logging
import os
import platform
import sys
import traceback
from typing import List, Dict, Any, Callable, Optional, Tuple, Union, Deque
from collections import deque
import multiprocessing

# Import Windows-specific modules if needed
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

class WorkStealingThreadPoolExecutor(ThreadPoolExecutor):
    """
    A ThreadPoolExecutor implementation with work stealing capability.
    This executor maintains a separate work queue for each worker thread,
    allowing idle threads to steal work from busy ones.
    """
    
    def __init__(self, max_workers=None, thread_name_prefix="",
                initializer=None, initargs=(), **kwargs):
        """Initialize the work-stealing thread pool executor."""
        max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix,
                        initializer=initializer, initargs=initargs)
                        
        # Create a work queue for each worker
        self._worker_queues = [queue.SimpleQueue() for _ in range(max_workers)]
        
        # Queue to track idle workers
        self._idle_workers = queue.Queue()
        
        # Track which worker processed which task
        self._task_to_worker = {}
        self._task_to_worker_lock = threading.RLock()
        
        # Set up worker threads
        self._threads = []
        self._shutdown_lock = threading.RLock()
        self._shutdown = False
        self._work_stealing_enabled = True
        
        # Track queue sizes for load balancing
        self._queue_sizes = [0] * max_workers
        self._queue_size_lock = threading.RLock()
        
        # Create and start worker threads
        for i in range(max_workers):
            thread = threading.Thread(
                name=f"{thread_name_prefix or 'Thread'}-{i}",
                target=self._worker_thread,
                args=(i,),
                daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _worker_thread(self, worker_id):
        """Worker thread that processes tasks from its queue and steals when idle."""
        # Set thread priority if supported
        if HAS_WIN32API:
            try:
                thread_handle = win32api.GetCurrentThread()
                # Set to slightly above normal priority for worker threads
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_ABOVE_NORMAL)
            except Exception as e:
                logging.debug(f"Could not set thread priority: {e}")
                
        # Record thread initialization
        logging.debug(f"Worker thread {worker_id} started")
        
        # Get our own work queue
        work_queue = self._worker_queues[worker_id]
        
        while True:
            work_item = None
            
            # Try to get work from our own queue first
            try:
                work_item = work_queue.get(block=False)
                work_item_source = worker_id
            except queue.Empty:
                # Mark as idle
                self._idle_workers.put(worker_id)
                
                # Try to steal work from other busy workers
                if self._work_stealing_enabled:
                    work_item, work_item_source = self._steal_work(worker_id)
                
                # If no work was stolen, wait for work to be assigned
                if work_item is None:
                    try:
                        work_item = work_queue.get(block=True, timeout=0.1)
                        work_item_source = worker_id
                    except queue.Empty:
                        # If we're shutting down, exit the loop
                        if self._shutdown:
                            break
                        continue
                        
            # If we're shutting down, exit the loop
            if self._shutdown:
                # Put back any pending work if possible
                if work_item is not None:
                    try:
                        work_queue.put(work_item)
                    except Exception:
                        pass
                break
                
            # Update queue size tracking
            with self._queue_size_lock:
                self._queue_sizes[work_item_source] -= 1
                
            # Process the work item
            if work_item is not None:
                # Get the Future and the function to execute
                future, fn, args, kwargs = work_item
                
                # Execute the function and set the result
                if not future.set_running_or_notify_cancel():
                    continue
                
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    future.set_exception(exc)
                    # Clear any references to the exception to avoid reference cycles
                    self = None
                else:
                    future.set_result(result)
                    
        logging.debug(f"Worker thread {worker_id} exiting")
                    
    def _steal_work(self, worker_id):
        """Attempt to steal work from other busy workers."""
        # Find the busiest worker
        busiest_worker = -1
        highest_load = 0
        
        with self._queue_size_lock:
            # Skip our own queue
            for i, size in enumerate(self._queue_sizes):
                if i != worker_id and size > highest_load:
                    highest_load = size
                    busiest_worker = i
                    
            # Don't steal unless there's significant imbalance
            if highest_load <= 1:
                return None, -1
                
        # Attempt to steal from the busiest worker
        if busiest_worker >= 0:
            try:
                work = self._worker_queues[busiest_worker].get(block=False)
                logging.debug(f"Worker {worker_id} stole work from worker {busiest_worker}")
                return work, busiest_worker
            except queue.Empty:
                pass
                
        return None, -1
        
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
                
            future = concurrent.futures.Future()
            work_item = (future, fn, args, kwargs)
            
            # Choose the best worker to handle this task
            worker_id = self._choose_worker()
            
            # Assign the task to the chosen worker
            self._worker_queues[worker_id].put(work_item)
            
            # Track which worker got this task
            with self._task_to_worker_lock:
                self._task_to_worker[future] = worker_id
                
            # Update queue size tracking
            with self._queue_size_lock:
                self._queue_sizes[worker_id] += 1
                
            return future
            
    def _choose_worker(self):
        """Choose the best worker to handle a new task."""
        # Try to get an idle worker first
        try:
            return self._idle_workers.get(block=False)
        except queue.Empty:
            pass
            
        # If no idle workers, choose the least busy one
        least_busy = 0
        min_queue_size = float('inf')
        
        with self._queue_size_lock:
            for i, size in enumerate(self._queue_sizes):
                if size < min_queue_size:
                    min_queue_size = size
                    least_busy = i
                    
        return least_busy
        
    def shutdown(self, wait=True, cancel_futures=False):
        """Shut down the executor."""
        with self._shutdown_lock:
            self._shutdown = True
            
        if cancel_futures:
            # Cancel all pending futures
            with self._task_to_worker_lock:
                for future, worker_id in list(self._task_to_worker.items()):
                    future.cancel()
                    
        if wait:
            for thread in self._threads:
                thread.join(timeout=2.0)
                
    def set_worker_affinity(self):
        """Set CPU affinity for worker threads (Windows only)."""
        if not HAS_WIN32API:
            return False
            
        try:
            # Get CPU count
            cpu_count = os.cpu_count()
            if not cpu_count or cpu_count <= 1:
                return False
                
            # Distribute workers across available CPUs
            for i, thread in enumerate(self._threads):
                try:
                    # Assign to a specific core in round-robin fashion
                    # Reserve Core 0 for system and UI threads
                    core = 1 + (i % (cpu_count - 1))
                    
                    # Set affinity for the thread
                    thread_id = thread.ident
                    if thread_id:
                        thread_handle = win32api.OpenThread(
                            win32con.THREAD_SET_INFORMATION | win32con.THREAD_QUERY_INFORMATION,
                            False,
                            thread_id
                        )
                        
                        # Create affinity mask for this core
                        mask = 1 << core
                        
                        # Set thread affinity
                        win32process.SetThreadAffinityMask(thread_handle, mask)
                        win32api.CloseHandle(thread_handle)
                        
                        logging.debug(f"Worker thread {i} assigned to CPU core {core}")
                except Exception as e:
                    logging.debug(f"Error setting affinity for thread {i}: {e}")
                    
            return True
        except Exception as e:
            logging.error(f"Error setting thread affinities: {e}")
            return False
            
    def enable_work_stealing(self, enabled=True):
        """Enable or disable work stealing between threads."""
        self._work_stealing_enabled = enabled
        return self._work_stealing_enabled 