# camera_manager.py
import cv2
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, src, resolution=(320, 240)):
        self.src = src
        self.resolution = resolution
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        if not self.cap.isOpened():
            logger.error(f"Error: Could not open video source {self.src}")
        else:
            logger.info(f"Video source {self.src} opened successfully")

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    logger.error(f"Failed to grab frame from source: {self.src}")
                    self.frame = None
                    break
        logger.info(f"Exiting update loop for source: {self.src}")

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                logger.error(f"No frame to read from source: {self.src}")
                return None

    def stop(self):
        self.stopped = True
        self.cap.release()

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.lock = threading.Lock()

    def add_camera(self, camera_id, src):
        with self.lock:
            if camera_id not in self.cameras:
                self.cameras[camera_id] = VideoStream(src).start()
                logger.info(f"Camera {camera_id} added successfully")
            else:
                logger.info(f"Camera {camera_id} already exists")

    def get_frame(self, camera_id):
        with self.lock:
            if camera_id in self.cameras:
                frame = self.cameras[camera_id].read()
                if frame is not None:
                    return frame
                else:
                    logger.error(f"No frame available for camera {camera_id}")
                    return None
            else:
                logger.error(f"Camera {camera_id} not found")
                return None

    def remove_camera(self, camera_id):
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()
                del self.cameras[camera_id]
                logger.info(f"Camera {camera_id} removed successfully")
            else:
                logger.error(f"Camera {camera_id} not found")
