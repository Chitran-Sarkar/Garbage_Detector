import logging
import time
from serial_manager import SerialManager
from tts_player import tts_player

class ServoController:
    """Controls multiple servos via SerialManager (custom protocol), with per-bin cooldown."""
    def __init__(self, serial_manager, num_bins=4, cooldown_seconds=10):
        self.serial_manager = serial_manager
        self.num_bins = num_bins
        self.cooldown_seconds = cooldown_seconds
        self.last_open_time = {i: 0 for i in range(num_bins)}

    def can_open_bin(self, bin_index):
        now = time.time()
        last_time = self.last_open_time.get(bin_index, 0)
        can_open = (now - last_time) > self.cooldown_seconds
        logging.info(f"ServoController: can_open_bin({bin_index}) = {can_open} (last_open={last_time}, now={now})")
        return can_open

    def open_lid(self, bin_index):
        if self.can_open_bin(bin_index):
            result = self.serial_manager.send_command(f'OPEN:{bin_index}')
            if result:
                self.last_open_time[bin_index] = time.time()
                logging.info(f"ServoController: Sent OPEN:{bin_index} command to servo.")
                # Add TTS feedback for bin opening
                if tts_player and tts_player.initialized:
                    tts_player.speak_bin_open(bin_index)
            else:
                logging.error(f"ServoController: Failed to send OPEN:{bin_index} command.")
            return result
        else:
            logging.info(f"ServoController: Bin {bin_index} is in cooldown.")
            return False

    def close_lid(self, bin_index):
        result = self.serial_manager.send_command(f'CLOSE:{bin_index}')
        if result:
            logging.info(f"ServoController: Sent CLOSE:{bin_index} command to servo.")
            # Add TTS feedback for bin closing
            if tts_player and tts_player.initialized:
                tts_player.speak_bin_close(bin_index)
        else:
            logging.error(f"ServoController: Failed to send CLOSE:{bin_index} command.")
        return result

    def is_connected(self):
        return self.serial_manager.ser and self.serial_manager.ser.is_open

    def get_servo_status(self):
        # Optionally parse status from serial if Arduino sends it
        return None

    def cleanup(self):
        self.serial_manager.close() 