# utils.py
import os
import logging
from django.conf import settings
from .models import Face
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

log_lock = threading.Lock()
logs = []

def reconcile_faces():
    faces = Face.objects.all()
    faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')

    for face in faces:
        image_path = os.path.join(settings.MEDIA_ROOT, face.image.name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}. Deleting record from database.")
            face.delete()

def log_event(event):
    global logs
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {event}"
    with log_lock:
        logs.append(log_entry)
    print("log event call", log_entry)  # Debug statement

def get_logs(request):
    global logs
    with log_lock:
        log_data = logs[-100:]  # Get the last 100 log entries
    print("Fetching logs:", log_data)  # Debug statement
    return JsonResponse({'logs': log_data})