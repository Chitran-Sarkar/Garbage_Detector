# config.py

# UI Layout
UI_LAYOUT = {
    "bg_width": 1920,
    "bg_height": 1080,
    "camera_pos": (43, 289),
    "camera_size": (885, 520),
    "waste_pos": (1210, 260),
    "arrow_pos": (1280, 590),
    "bin_pos": (1205, 700)
}

# Class and bin configuration
WASTE_CLASSES = {
    0: {
        "name": "Nothing",
        "description": "No waste detected",
        "bin_index": None
    },
    1: {
        "name": "Bottle",
        "description": "Plastic bottle",
        "bin_index": 0  # Recyclable waste bin
    },
    2: {
        "name": "Disinfectant",
        "description": "Disinfectant bottle",
        "bin_index": 1  # Hazardous waste bin
    },
    3: {
        "name": "Phone",
        "description": "Mobile phone",
        "bin_index": 1  # Hazardous waste bin
    },
    4: {
        "name": "Apple",
        "description": "Apple core or scraps",
        "bin_index": 2  # Food waste bin
    },
    5: {
        "name": "Eggplant",
        "description": "Eggplant",
        "bin_index": 2  # Food waste bin
    },
    6: {
        "name": "Ceramic bowl",
        "description": "Ceramic bowl",
        "bin_index": 3  # Residual waste bin
    },
    7: {
        "name": "Pen bag",
        "description": "Pen bag",
        "bin_index": 3  # Residual waste bin
    }
}

# Waste bin configuration
WASTE_BINS = {
    0: {
        "name": "Recyclable waste bin",
        "description": "For recyclable materials like plastic, glass, paper, etc.",
        "servo_pin": 9,
        "color": (0, 255, 0)  # Green
    },
    1: {
        "name": "Hazardous waste bin",
        "description": "For hazardous materials like batteries, chemicals, etc.",
        "servo_pin": 10,
        "color": (0, 0, 255)  # Red
    },
    2: {
        "name": "Food waste bin",
        "description": "For food waste and organic materials",
        "servo_pin": 11,
        "color": (0, 165, 255)  # Orange
    },
    3: {
        "name": "Residual waste bin",
        "description": "For non-recyclable and non-hazardous waste",
        "servo_pin": 12,
        "color": (128, 128, 128)  # Gray
    }
}

# Servo angles
SERVO_CONFIG = {
    "open_angle": 90,
    "closed_angle": 0,
    "open_time": 3.0,  # How long to keep bin open in seconds
    "move_time": 0.5   # Time to move servo to position
}

# Display configuration
DISPLAY_CONFIG = {
    "show_labels": True,  # Enable text overlay with class names
    "show_confidence": True,  # Show confidence scores
    "show_fps": True,  # Show FPS counter
    "font_scale": 0.6,  # Text size
    "fullscreen": True,  # Start in fullscreen mode
    "show_performance_metrics": True  # Show performance metrics
}

# Ultrasonic sensor configuration
ULTRASONIC_CONFIG = {
    "trig_pin": 7,  # Trigger pin
    "echo_pin": 8,  # Echo pin
    "analog_pin": 0,  # Analog pin for Firmata
    "threshold_distance": 30,  # cm
    "reading_interval": 0.1,  # seconds
    "max_distance": 400,  # cm
    "baud_rate": 57600  # Match existing Firmata baud rate
}

# For backward compatibility
ORIG_BG_WIDTH = UI_LAYOUT["bg_width"]
ORIG_BG_HEIGHT = UI_LAYOUT["bg_height"]
CAMERA_FEED_POS = UI_LAYOUT["camera_pos"]
CAMERA_FEED_SIZE = UI_LAYOUT["camera_size"]
WASTE_POS = UI_LAYOUT["waste_pos"]
ARROW_POS = UI_LAYOUT["arrow_pos"]
BIN_POS = UI_LAYOUT["bin_pos"]

# Generate the original CLASS_MAPPING from WASTE_CLASSES
CLASS_MAPPING = {class_id: waste_info["bin_index"] for class_id, waste_info in WASTE_CLASSES.items()}

# Generate SERVO_PINS from WASTE_BINS
SERVO_PINS = {bin_id: bin_info["servo_pin"] for bin_id, bin_info in WASTE_BINS.items()}

# For backward compatibility
SERVO_OPEN_ANGLE = SERVO_CONFIG["open_angle"]
SERVO_CLOSED_ANGLE = SERVO_CONFIG["closed_angle"]
DISPLAY_LABELS = DISPLAY_CONFIG["show_labels"]

# Generate BIN_NAMES from WASTE_BINS
BIN_NAMES = {bin_id: bin_info["name"] for bin_id, bin_info in WASTE_BINS.items()}

# Generate CLASS_NAMES from WASTE_CLASSES
CLASS_NAMES = {class_id: waste_info["name"] for class_id, waste_info in WASTE_CLASSES.items()}
