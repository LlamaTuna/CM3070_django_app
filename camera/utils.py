# utils.py
import os
import logging
from django.conf import settings
from .models import Face

logger = logging.getLogger(__name__)

def reconcile_faces():
    faces = Face.objects.all()
    faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')

    for face in faces:
        image_path = os.path.join(settings.MEDIA_ROOT, face.image.name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}. Deleting record from database.")
            face.delete()
