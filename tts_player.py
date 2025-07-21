import win32com.client
import threading
import logging
import queue
import time
from config import BIN_NAMES

class TTSPlayer:
    """
    Text-to-speech player using Windows SAPI.
    Provides asynchronous speech capabilities optimized for Windows.
    """
    
    def __init__(self):
        self.speaker = None
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.running = False
        self.initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize the TTS engine"""
        try:
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.speaker.Rate = 0  # -10 to 10, 0 is default
            self.speaker.Volume = 100  # 0 to 100
            
            # Start the speech worker thread
            self.running = True
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
            self.initialized = True
            logging.info("TTS system initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize TTS system: {e}")
            self.initialized = False
    
    def _speech_worker(self):
        """Worker thread that processes speech requests from the queue"""
        while self.running:
            try:
                # Get text from queue with timeout to allow checking running flag
                text = self.speech_queue.get(timeout=0.5)
                if self.speaker and text:
                    self.speaker.Speak(text, 1)  # SVSFlagsAsync = 1
                self.speech_queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                logging.error(f"TTS error in worker thread: {e}")
                time.sleep(0.1)  # Prevent tight loop if there's an error
    
    def speak(self, text):
        """
        Add text to the speech queue for asynchronous playback.
        Returns immediately without blocking.
        """
        if not self.initialized:
            logging.debug("TTS not initialized, ignoring speech request")
            return False
            
        try:
            self.speech_queue.put(text)
            return True
        except Exception as e:
            logging.error(f"Error queueing speech: {e}")
            return False
    
    def speak_bin_open(self, bin_index):
        """Speak notification that a specific bin is opening"""
        bin_name = BIN_NAMES.get(bin_index, f"bin {bin_index}")
        return self.speak(f"Opening {bin_name}")
    
    def speak_bin_close(self, bin_index):
        """Speak notification that a specific bin is closing"""
        bin_name = BIN_NAMES.get(bin_index, f"bin {bin_index}")
        return self.speak(f"Closing {bin_name}")
    
    def cleanup(self):
        """Clean up resources before shutdown"""
        self.running = False
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
        self.speaker = None
        logging.info("TTS system cleaned up")

# Create a singleton instance
tts_player = TTSPlayer() 