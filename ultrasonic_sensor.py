import logging
from serial_manager import SerialManager

class UltrasonicSensor:
    def __init__(self, serial_manager, threshold_distance=30):
        self.serial_manager = serial_manager
        self.threshold_distance = threshold_distance
        self.last_distance = None

    def get_distance(self):
        line = self.serial_manager.get_latest_line(prefix='DIST:')
        if line:
            try:
                value = float(line.split(':')[1])
                self.last_distance = value
                logging.info(f"UltrasonicSensor: Read distance {value} cm from serial.")
                return value
            except Exception as e:
                logging.error(f"UltrasonicSensor: Failed to parse distance from line '{line}': {e}")
        return self.last_distance

    def is_object_detected(self):
        distance = self.get_distance()
        detected = distance is not None and distance < self.threshold_distance
        logging.info(f"UltrasonicSensor: is_object_detected = {detected} (distance={distance})")
        return detected 