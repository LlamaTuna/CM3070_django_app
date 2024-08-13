import os
import django
from django.conf import settings
import cv2
import numpy as np
from collections import deque

# Set up Django environment manually
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
django.setup()

class ObjectClassifier:
    def __init__(self, buffer_size=5, confidence_threshold=0.2, nms_threshold=0.4):
        # Load class names
        self.classes = self._load_class_names()

        # Path to the YOLOv8 ONNX model
        model_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov10n.onnx')

        try:
            print("Loading YOLOv8 ONNX model from:", model_path)
            self.net = cv2.dnn.readNetFromONNX(model_path)
            print("YOLOv8 model loaded successfully")
        except cv2.error as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.net = None  # Ensure that self.net is defined even in case of error
