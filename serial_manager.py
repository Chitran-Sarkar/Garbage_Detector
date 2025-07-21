import serial
import threading
import time
import logging
from collections import deque

class SerialManager:
    def __init__(self, port, baudrate=57600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.read_thread = None
        self.running = False
        self.line_queue = deque(maxlen=100)
        self.lock = threading.Lock()

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.running = True
            self.read_thread = threading.Thread(target=self._read_lines, daemon=True)
            self.read_thread.start()
            logging.info(f"SerialManager: Opened port {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            logging.error(f"SerialManager: Failed to open port {self.port}: {e}")
            return False

    def close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            logging.info("SerialManager: Closed serial port.")

    def _read_lines(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    with self.lock:
                        self.line_queue.append(line)
            except Exception as e:
                logging.error(f"SerialManager: Error reading line: {e}")
                time.sleep(0.1)

    def get_latest_line(self, prefix=None):
        with self.lock:
            for line in reversed(self.line_queue):
                if prefix is None or line.startswith(prefix):
                    return line
        return None

    def send_command(self, cmd):
        try:
            if self.ser and self.ser.is_open:
                self.ser.write((cmd.strip() + '\n').encode())
                logging.info(f"SerialManager: Sent command: {cmd}")
                return True
        except Exception as e:
            logging.error(f"SerialManager: Failed to send command '{cmd}': {e}")
        return False 