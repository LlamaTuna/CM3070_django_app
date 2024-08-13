import os
import django
from django.conf import settings
import cv2

# Set up Django environment manually
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
django.setup()

cfg_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov3-tiny.cfg')
weights_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov3-tiny.weights')

try:
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    print("Model loaded successfully")
except cv2.error as e:
    print(f"Error loading model: {e}")
