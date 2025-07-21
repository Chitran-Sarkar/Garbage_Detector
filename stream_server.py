from flask import Flask, Response, render_template
import cv2
import numpy as np
from ui_renderer import draw_ui
import logging
import threading
import queue
import time

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=2)  # Buffer for frames
is_streaming = False

def generate_frames():
    """Generator function for streaming frames"""
    while is_streaming:
        try:
            # Get frame from queue with timeout
            frame = frame_queue.get(timeout=1.0)
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    # Convert to bytes and yield
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            continue

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_stream_server(host='0.0.0.0', port=5000):
    """Start the Flask server"""
    global is_streaming
    is_streaming = True
    app.run(host=host, port=port, debug=False, threaded=True)

def stop_stream_server():
    """Stop the Flask server"""
    global is_streaming
    is_streaming = False

def update_frame(frame):
    """Update the frame in the queue"""
    try:
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # If queue is full, remove old frame and add new one
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(frame)
    except Exception as e:
        logging.error(f"Error updating frame: {e}")

if __name__ == '__main__':
    # Example usage
    import cv2
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_stream_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Example: Capture from webcam and stream
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                update_frame(frame)
            time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        stop_stream_server()
        cap.release() 