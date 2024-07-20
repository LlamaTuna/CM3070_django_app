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

# # Global variable to hold the camera instance
# camera_instance = None
# class VideoCamera:
#     def __init__(self, resolution=(320, 240)):
#         self.video = cv2.VideoCapture(0)
#         if not self.video.isOpened():
#             print("Error: Could not open video device.")
#             self.video = None
#         else:
#             self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#             self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
#         self.previous_frame = None
#         self.detector = MTCNN()

#         if not is_management_command:
#             self._initialize_model()

#         self.known_faces_features = []
#         self.known_faces_labels = []
#         self.load_known_faces()
#         self.detection_log = {}
#         self.detection_interval = 5  # seconds
#         self.alert_interval = 30  # 30 seconds
#         self.alert_buffer = []
#         self.frame_buffer = []  # Buffer to store frames for each detected event
#         self.running_buffer = []  # Running buffer to store frames continuously
#         self.last_alert_time = time.time()
#         self.frame_skip_interval = 2
#         self.frame_count = 0  # Frame counter to skip frames
#         self.lock = threading.Lock()
#         self.frames = []  # Initialize frames list
#         self.detected_faces = []  # Initialize detected_faces list
#         self.face_detection_thread = threading.Thread(target=self._process_frames)
#         self.face_detection_thread.daemon = True
#         self.face_detection_thread.start()

#         # Start the timer for periodic saving of the running buffer
#         self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#         self.save_timer.start()

#     def _initialize_model(self):
#         from tensorflow.keras.applications.resnet50 import ResNet50
#         from tensorflow.keras.models import Model
#         self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#         self.model = self._build_feature_extractor(self.base_model)

#     def __del__(self):
#         if self.video:
#             self.video.release()
#         self.save_timer.cancel()  # Cancel the timer when the object is deleted

#     def _build_feature_extractor(self, base_model):
#         from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(1024, activation='relu')(x)
#         predictions = Dense(128, activation='relu')(x)
#         return Model(inputs=base_model.input, outputs=predictions)

#     def _preprocess_image(self, img):
#         from tensorflow.keras.applications.resnet50 import preprocess_input
#         if img is None or img.size == 0:
#             return None
#         img = cv2.resize(img, (224, 224))
#         img_array = np.array(img, dtype='float32')
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         return img_array

#     def _extract_features(self, img_array):
#         if img_array is None:
#             return None
#         features = self.model.predict(img_array)
#         return features.flatten()

#     def _detect_faces(self, img, confidence_threshold=0.95):
#         small_img = cv2.resize(img, (160, 120))
#         faces = self.detector.detect_faces(small_img)
#         for face in faces:
#             face['box'] = [int(coordinate * 2) for coordinate in face['box']]
#         filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
#         return filtered_faces

#     def load_known_faces(self):
#         known_faces_dir = settings.KNOWN_FACES_DIR
#         for filename in os.listdir(known_faces_dir):
#             if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#                 img_path = os.path.join(known_faces_dir, filename)
#                 label = os.path.splitext(filename)[0]
#                 img = cv2.imread(img_path)
#                 face_features = self._preprocess_and_extract(img)
#                 if face_features is not None:
#                     self.known_faces_features.append(face_features)
#                     self.known_faces_labels.append(label)

#     def _preprocess_and_extract(self, img):
#         faces = self._detect_faces(img)
#         if faces:
#             x, y, width, height = faces[0]['box']
#             face = img[y:y+height, x:x+width]
#             face_array = self._preprocess_image(face)
#             if face_array is None:
#                 return None
#             features = self._extract_features(face_array)
#             return features
#         return None

#     def detect_movement(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)

#         if self.previous_frame is None:
#             self.previous_frame = gray
#             return False, None

#         frame_diff = cv2.absdiff(self.previous_frame, gray)
#         self.previous_frame = gray

#         thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)

#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) < 500:
#                 continue
#             (x, y, w, h) = cv2.boundingRect(contour)
#             return True, (x, y, w, h)

#         return False, None

#     def get_frame(self):
#         if not self.video:
#             return None
#         success, image = self.video.read()
#         if not success:
#             return None

#         self.frame_count += 1
#         if self.frame_count % self.frame_skip_interval != 0:
#             return None

#         # Optionally resize the frame to further optimize processing
#         image = cv2.resize(image, (320, 240))

#         # Detect movement
#         movement_detected, movement_box = self.detect_movement(image)
#         if movement_detected:
#             with self.lock:
#                 self.frames.append(image.copy())

#             x, y, width, height = movement_box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
#             cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#             self.frame_buffer.append(image.copy())  # Store the frame
#             self.running_buffer.append(image.copy())  # Store the frame in the running buffer
#             self.log_event("Movement detected")  # Log movement event
#             print(" Get frame call, Movement detected")  # Debug statement

#         with self.lock:
#             detected_faces = self.detected_faces[:]

#         for face in detected_faces:
#             x, y, width, height = face['box']
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
#             label = face.get('label', 'Unknown')
#             cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Add timestamp to the frame
#         try:
#             local_time = datetime.now(pytz.timezone('US/Pacific'))
#             timestamp_text = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
#             cv2.putText(image, timestamp_text, (10, image.shape[0] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         except Exception as e:
#             print(f"Error adding timestamp: {e}")

#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def _process_frames(self):
#         while True:
#             if not self.frames:
#                 time.sleep(0.01)
#                 continue

#             with self.lock:
#                 frame = self.frames.pop(0)

#             gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
#             faces = self._detect_faces(gray_image_3ch)

#             for face in faces:
#                 x, y, width, height = face['box']
#                 if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
#                     continue  # Skip invalid face boxes
#                 face_img = frame[y:y+height, x:x+width]
#                 face_array = self._preprocess_image(face_img)
#                 if face_array is None:
#                     continue
#                 features = self._extract_features(face_array)

#                 min_distance = float('inf')
#                 label = None
#                 for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
#                     distance = euclidean(features, known_features)
#                     if distance < min_distance:
#                         min_distance = distance
#                         label = known_label

#                 face['label'] = label if label else 'Unknown'

#                 # Save the face image
#                 self.save_face_image(face_img, label)

#             with self.lock:
#                 self.detected_faces = faces

#     def log_event(self, event):
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_entry = f"[{timestamp}] {event}"
#         self.alert_buffer.append(log_entry)
#         log_event(event)
#         print("VideoCamera logged event:", log_entry)  # Debug statement
#         # Check if it's time to send the email
#         if time.time() - self.last_alert_time >= self.alert_interval:
#             self.send_email_snapshot()
#             self.last_alert_time = time.time()

#     def send_email_snapshot(self):
#         if not self.alert_buffer:
#             return

#         try:
#             smtp_server = "smtp.gmail.com"
#             smtp_port = 587  # For starttls
#             smtp_user = "cm3070.smtptest@gmail.com"
#             smtp_password = "lxvv mgoj dzld gbae"  # Use App Password for better security

#             from_email = smtp_user
#             to_email = "cm3070.smtptest@gmail.com"
#             subject = "Motion Detection Alert Snapshot"
#             body = "\n".join(self.alert_buffer)

#             msg = MIMEMultipart()
#             msg['From'] = from_email
#             msg['To'] = to_email
#             msg['Subject'] = subject

#             # Modify the email body to include details about detected faces
#             if self.detected_faces:
#                 body += "\n\nDetected Faces:\n"
#                 for i, face in enumerate(self.detected_faces):
#                     label = face.get('label', 'Unknown')
#                     body += f"Person {i + 1}: {label}\n"

#             msg.attach(MIMEText(body, 'plain'))

#             # Select representative frames to attach
#             selected_frames = self.select_representative_frames(self.frame_buffer, 2)  # Select 2 frames

#             for i, frame in enumerate(selected_frames):
#                 _, img_encoded = cv2.imencode('.jpg', frame)
#                 image_data = img_encoded.tobytes()
#                 image = MIMEImage(image_data, name=f"event_{i + 1}.jpg")
#                 msg.attach(image)

#             server = smtplib.SMTP(smtp_server, smtp_port)
#             server.starttls()
#             server.login(smtp_user, smtp_password)
#             text = msg.as_string()
#             server.sendmail(from_email, to_email, text)
#             server.quit()

#             # Clear the buffer after sending
#             self.alert_buffer = []
#             self.frame_buffer = []
#             print("Email sent successfully")  # Debug statement
#         except Exception as e:
#             print(f"Failed to send snapshot email: {str(e)}")

#     def select_representative_frames(self, frames, num_frames):
#         if len(frames) <= num_frames:
#             return frames
#         interval = len(frames) // num_frames
#         selected_frames = [frames[i * interval] for i in range(num_frames)]
#         return selected_frames

#     def save_face_image(self, face_img, label):
#         # Create the faces_seen directory if it does not exist
#         faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')
#         if not os.path.exists(faces_seen_dir):
#             os.makedirs(faces_seen_dir)

#         # Generate a unique filename
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"{label}_{timestamp}.jpg"
#         filepath = os.path.join(faces_seen_dir, filename)

#         # Save the face image
#         cv2.imwrite(filepath, face_img)
#         print(f"Face image saved: {filepath}")  # Debug statement

#         # Save the face record in the database
#         Face.objects.create(name=label, image=f"faces_seen/{filename}")
#         print(f"Face record saved: {label}, {filename}")  # Debug statement


#     def save_running_buffer_clip(self):
#         if self.running_buffer:
#             # Create the event_clips directory if it does not exist
#             event_clips_dir = os.path.join(settings.MEDIA_ROOT, 'event_clips')
#             if not os.path.exists(event_clips_dir):
#                 os.makedirs(event_clips_dir)

#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f"event_{timestamp}.mp4"
#             file_path = os.path.join(event_clips_dir, filename)

#             # Define the codec and create VideoWriter object
#             fps = 20  # Frames per second, adjust this according to your camera settings
#             out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))

#             for frame in self.running_buffer:
#                 out.write(frame)
#             out.release()

#             # Save the event record in the database
#             event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{filename}')
#             event.save()

#             self.running_buffer = []  # Clear buffer after saving
#             print(f"Running buffer clip saved: {file_path}")  # Debug statement

#             # Restart the timer for the next interval
#             self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#             self.save_timer.start()


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
