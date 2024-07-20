import threading
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import login
from .models import Face, Event
from .forms import TagFaceForm, CustomUserCreationForm, UploadFaceForm
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import euclidean
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
from datetime import datetime, timezone, timedelta
from .utils import reconcile_faces
import pytz
import logging
from .video_camera import VideoCamera

# Check if the script is running a management command
import sys
is_management_command = len(sys.argv) > 1 and sys.argv[1] in ['makemigrations', 'migrate', 'createsuperuser', 'collectstatic']

if not is_management_command:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
    from tensorflow.keras.models import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

log_lock = threading.Lock()
logs = []

# Global variable to hold the camera instance
camera_instance = None

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


# Initialize the camera processing
def start_camera():
    global camera_instance
    if camera_instance is None:
        camera_instance = VideoCamera()

# Initialize the camera processing
def start_camera():
    global camera_instance
    if camera_instance is None:
        camera_instance = VideoCamera()

if not is_management_command:
    camera_thread = threading.Thread(target=start_camera)
    camera_thread.daemon = True
    camera_thread.start()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(camera_instance),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'camera/index.html')

@login_required
def list_faces(request):
    reconcile_faces()  # Reconcile the database with the actual images
    faces = Face.objects.filter(tagged=False)
    return render(request, 'camera/list_faces.html', {'faces': faces})

@login_required
def tag_face(request, face_id):
    face = Face.objects.get(id=face_id)
    if request.method == 'POST':
        form = TagFaceForm(request.POST, request.FILES, instance=face)
        if form.is_valid():
            form.save()
            known_faces_path = settings.KNOWN_FACES_DIR
            print("Known Faces Path:", known_faces_path)  # Debug statement
            if not os.path.exists(known_faces_path):
                os.makedirs(known_faces_path)
            new_path = os.path.join(known_faces_path, os.path.basename(face.image.path))
            print("New Path:", new_path)  # Debug statement
            os.rename(face.image.path, new_path)
            face.image.name = os.path.join('known_faces', os.path.basename(new_path))
            face.tagged = True
            face.save()
            return redirect('list_faces')
    else:
        form = TagFaceForm(instance=face)
    return render(request, 'camera/tag_face.html', {'form': form})


@staff_member_required
def admin_view(request):
    return render(request, 'camera/admin.html')

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = CustomUserCreationForm()
    return render(request, 'camera/register.html', {'form': form})

@login_required
def upload_face(request):
    if request.method == 'POST':
        form = UploadFaceForm(request.POST, request.FILES)
        if form.is_valid():
            face = form.save(commit=False)
            # Process the uploaded image
            image_file = request.FILES['image']
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                form.add_error('image', 'Image not valid. Please upload a valid image file.')
            else:
                # Detect and crop the face
                detector = MTCNN()
                faces = detector.detect_faces(image)
                if faces:
                    x, y, width, height = faces[0]['box']
                    cropped_face = image[y:y + height, x:x + width]

                    # Ensure the known_faces directory exists
                    known_faces_dir = settings.KNOWN_FACES_DIR
                    if not os.path.exists(known_faces_dir):
                        os.makedirs(known_faces_dir)

                    # Create a unique filename for the cropped face
                    filename = f"{face.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cropped_path = os.path.join(known_faces_dir, filename)
                    
                    # Debugging statements
                    print(f"Known Faces Directory: {known_faces_dir}")
                    print(f"Filename: {filename}")
                    print(f"Cropped Path: {cropped_path}")

                    # Save cropped face directly to known_faces directory
                    cv2.imwrite(cropped_path, cropped_face)

                    # Update face image path and save the record
                    face.image.name = os.path.join('known_faces', filename)
                    face.tagged = True
                    face.save()

                    return redirect('list_faces')
                else:
                    form.add_error('image', 'No face detected in the image. Please upload a different image.')
    else:
        form = UploadFaceForm()
    return render(request, 'camera/upload_face.html', {'form': form})
