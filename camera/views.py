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
from .forms import EmailSettingsForm, UserSettingsForm
from .models import EmailSettings

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
    """
    Logs an event with a timestamp.

    Args:
        event (str): Description of the event.
    """
    global logs
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {event}"
    with log_lock:
        logs.append(log_entry)
    print("log event call", log_entry)  # Debug statement

def get_logs(request):
    """
    Returns the last 100 log entries as a JSON response.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response containing the last 100 log entries.
    """
    global logs
    with log_lock:
        log_data = logs[-100:]  # Get the last 100 log entries
    print("Fetching logs:", log_data)  # Debug statement
    return JsonResponse({'logs': log_data})

def initialize_camera(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = VideoCamera(request=request)
        if camera_instance.video is None:
            camera_instance = None
            print("Failed to initialize camera.")
        else:
            print("Camera initialized successfully.")

# Initialize the camera processing
# Remove the threading context here as request argument is not available in this context

def gen(camera):
    """
    Generator function to yield frames from the camera.

    Args:
        camera (VideoCamera): The camera instance.

    Yields:
        bytes: JPEG-encoded frame.
    """
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    global camera_instance
    if camera_instance is None:
        initialize_camera(request)
    return StreamingHttpResponse(gen(camera_instance),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    """
    Renders the index page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered index page.
    """
    return render(request, 'camera/index.html')

@login_required
def list_faces(request):
    """
    Lists all untagged faces.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered list_faces page with untagged faces.
    """
    reconcile_faces()  # Reconcile the database with the actual images
    faces = Face.objects.filter(tagged=False)
    return render(request, 'camera/list_faces.html', {'faces': faces})

@login_required
def tag_face(request, face_id):
    """
    Tags a face with the provided details.

    Args:
        request (HttpRequest): The HTTP request object.
        face_id (int): The ID of the face to be tagged.

    Returns:
        HttpResponse: The rendered tag_face page with the form or a redirect to list_faces.
    """
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
    """
    Renders the admin view page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered admin view page.
    """
    return render(request, 'camera/admin.html')

def register(request):
    """
    Handles user registration.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered register page with the form or a redirect to index.
    """
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
    """
    Handles the upload of a new face image.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered upload_face page with the form or a redirect to list_faces.
    """
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

@login_required
def email_settings(request):
    try:
        email_settings = EmailSettings.objects.get(user=request.user)
    except EmailSettings.DoesNotExist:
        email_settings = None

    if request.method == 'POST':
        form = EmailSettingsForm(request.POST, instance=email_settings)
        if form.is_valid():
            email_settings = form.save(commit=False)
            email_settings.user = request.user
            email_settings.save()
            print("Email settings saved:", email_settings.__dict__)  # Debug statement
            return redirect('email_settings')
    else:
        form = EmailSettingsForm(instance=email_settings)
    
    return render(request, 'camera/email_settings.html', {'form': form})

@login_required
def user_settings(request):
    if request.method == 'POST':
        form = UserSettingsForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('user_settings')
    else:
        form = UserSettingsForm(instance=request.user)
    return render(request, 'camera/user_settings.html', {'form': form})
