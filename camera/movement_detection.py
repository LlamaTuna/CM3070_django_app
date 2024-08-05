import cv2
import numpy as np
from threading import Thread, Lock

class MovementDetection:
    def __init__(self, frame_skip_interval=2):
        self.previous_frame = None
        self.frame_skip_interval = frame_skip_interval
        self.frame_count = 0
        self.lock = Lock()

    def detect_movement(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip_interval != 0:
            return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        with self.lock:
            if self.previous_frame is None:
                self.previous_frame = gray
                return False, None

            frame_diff = cv2.absdiff(self.previous_frame, gray)
            self.previous_frame = gray

        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            return True, (x, y, w, h)

        return False, None

    def start_detection(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        self.detection_thread = Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def _detection_loop(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            self.detect_movement(frame)

    def stop_detection(self):
        if self.video.isOpened():
            self.video.release()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join()

