import threading
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .movement_detection import MovementDetection
from .facial_recognition import FacialRecognition
from .send_email import SendEmail
from .save_audio import SaveAudio
import pytz
import os
from django.conf import settings
from .models import Event

class VideoCamera:
    def __init__(self, resolution=(320, 240), request=None):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Could not open video device.")
            self.video = None
        else:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.movement_detection = MovementDetection()
        self.facial_recognition = FacialRecognition()
        self.send_email = SendEmail(request)
        self.save_audio = SaveAudio()

        self.frame_skip_interval = 2
        self.frame_count = 0
        self.face_recognition_interval = 10
        self.face_recognition_counter = 0

        self.lock = threading.Lock()
        self.frames = []
        self.detected_faces = []

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self._process_frames)

        self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
        self.save_timer.start()

        self.frame_buffer = []
        self.running_buffer = []
        self.last_alert_time = time.time()
        self.alert_interval = 30  # 30 seconds

    def __del__(self):
        if self.video:
            self.video.release()
        self.save_timer.cancel()
        self.executor.shutdown(wait=False)
        if self.save_audio.audio_stream:
            self.save_audio.audio_stream.stop_stream()
            self.save_audio.audio_stream.close()

    def get_frame(self):
        if not self.video:
            return None
        success, image = self.video.read()
        if not success:
            return None

        self.frame_count += 1
        if self.frame_count % self.frame_skip_interval != 0:
            return None

        image = cv2.resize(image, (320, 240))

        movement_detected, movement_box = self.movement_detection.detect_movement(image)
        if movement_detected:
            with self.lock:
                self.frames.append(image.copy())
            x, y, width, height = movement_box
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
            cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            self.frame_buffer.append(image.copy())  # Ensure frame is added here
            self.running_buffer.append(image.copy())
            self.send_email.log_event("Movement detected")
            # Attempt to send email snapshot
            if time.time() - self.last_alert_time >= self.alert_interval:
                self.send_email.frame_buffer = self.frame_buffer  # Ensure SendEmail class has access to frame_buffer
                self.send_email.send_email_snapshot()
                self.last_alert_time = time.time()

        detected_faces = []
        with self.lock:
            for face in self.detected_faces:
                detected_faces.append(face)

        for face in detected_faces:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
            label = face.get('label', 'Unknown')
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)

        try:
            local_time = datetime.now(pytz.timezone('US/Pacific'))
            timestamp_text = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            cv2.putText(image, timestamp_text, (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error adding timestamp: {e}")

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def _process_frames(self):
        while True:
            if not self.frames:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.frames.pop(0)

            self.face_recognition_counter += 1
            if self.face_recognition_counter >= self.face_recognition_interval:
                recognized_faces = self.facial_recognition.recognize_faces(frame)
                with self.lock:
                    self.detected_faces = recognized_faces
                self.face_recognition_counter = 0

    def save_running_buffer_clip(self):
        if self.running_buffer:
            event_clips_dir = os.path.join(settings.MEDIA_ROOT, 'event_clips')
            if not os.path.exists(event_clips_dir):
                os.makedirs(event_clips_dir)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"event_{timestamp}.mp4"
            video_file_path = os.path.join(event_clips_dir, video_filename)

            fps = 20
            out = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))

            for frame in self.running_buffer:
                out.write(frame)
            out.release()

            audio_filename = f"audio_{timestamp}.wav"
            audio_file_path = os.path.join(event_clips_dir, audio_filename)
            self.save_audio.save_audio_clip(audio_file_path)

            final_filename = f"final_event_{timestamp}.mp4"
            final_file_path = os.path.join(event_clips_dir, final_filename)
            self.save_audio.combine_audio_video(video_file_path, audio_file_path, final_file_path)

            event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{final_filename}')
            event.save()

            self.running_buffer = []
            self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
            self.save_timer.start()

# import threading
# import logging
# from concurrent.futures import ThreadPoolExecutor
# from django.http import StreamingHttpResponse, JsonResponse
# from django.shortcuts import render, redirect
# from django.conf import settings
# from django.contrib.auth.decorators import login_required
# from django.contrib.admin.views.decorators import staff_member_required
# from django.contrib.auth import login
# from .models import Face, Event
# from .forms import TagFaceForm, CustomUserCreationForm, UploadFaceForm
# import cv2
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from scipy.spatial.distance import euclidean
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# import time
# from datetime import datetime, timezone, timedelta
# from .utils import reconcile_faces
# import pytz
# import pyaudio
# import wave
# import subprocess
# from .models import EmailSettings

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Check if the script is running a management command
# import sys
# is_management_command = len(sys.argv) > 1 and sys.argv[1] in ['makemigrations', 'migrate', 'createsuperuser', 'collectstatic']

# if not is_management_command:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
#     from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
#     from tensorflow.keras.models import Model

# # Global variable to hold the camera instance
# camera_instance = None

# # Global variables for threshold values
# CONFIDENCE_THRESHOLD = 0.90  # Adjust this value for face detection confidence
# RECOGNITION_THRESHOLD = 4.5   # Adjust this value for face recognition labeling

# class VideoCamera:
#     """
#     A class to represent a video camera for face detection and recognition.

#     Attributes:
#     video: cv2.VideoCapture
#         Video capture object for accessing the webcam.
#     detector: MTCNN
#         Face detection model.
#     base_model: ResNet50
#         Base model for feature extraction.
#     model: Model
#         Custom model for feature extraction.
#     known_faces_features: list
#         List of known faces' features.
#     known_faces_labels: list
#         List of known faces' labels.
#     frame_skip_interval: int
#         Interval for skipping frames.
#     frame_count: int
#         Counter for frames processed.
#     face_recognition_interval: int
#         Interval for face recognition.
#     face_recognition_counter: int
#         Counter for face recognition.
#     lock: threading.Lock
#         Lock for thread-safe operations.
#     frames: list
#         List of frames to be processed.
#     detected_faces: list
#         List of detected faces.
#     executor: ThreadPoolExecutor
#         Executor for processing frames in a separate thread.
#     save_timer: threading.Timer
#         Timer for periodic saving of the running buffer.
#     previous_frame: np.ndarray
#         Previous frame for motion detection.
#     detection_log: dict
#         Log for detection events.
#     detection_interval: int
#         Interval for logging detections.
#     alert_interval: int
#         Interval for sending alerts.
#     alert_buffer: list
#         Buffer for alert messages.
#     frame_buffer: list
#         Buffer for frames of detected events.
#     running_buffer: list
#         Running buffer to store frames continuously.
#     audio_stream: pyaudio.Stream
#         Audio stream for recording.
#     audio_frames: list
#         List of recorded audio frames.
#     last_alert_time: float
#         Timestamp for the last alert sent.
#     """

#     def __init__(self, resolution=(320, 240), request=None):
#         """
#         Initializes the video camera, face detection model, and other attributes.

#         Parameters:
#         resolution (tuple): Resolution for the video capture.
#         """
#         self.video = cv2.VideoCapture(0)
#         if not self.video.isOpened():
#             print("Error: Could not open video device.")
#             self.video = None
#         else:
#             self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#             self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

#         self.detector = MTCNN()
#         self._initialize_model()

#         self.known_faces_features = []
#         self.known_faces_labels = []
#         self.load_known_faces()

#         self.frame_skip_interval = 2
#         self.frame_count = 0
#         self.face_recognition_interval = 10
#         self.face_recognition_counter = 0

#         self.lock = threading.Lock()
#         self.frames = []
#         self.detected_faces = []

#         self.executor = ThreadPoolExecutor(max_workers=1)
#         self.executor.submit(self._process_frames)

#         # Start the timer for periodic saving of the running buffer
#         self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#         self.save_timer.start()

#         self.previous_frame = None
#         self.detection_log = {}
#         self.detection_interval = 5  # seconds
#         self.alert_interval = 30  # 30 seconds
#         self.alert_buffer = []
#         self.frame_buffer = []  # Buffer to store frames for each detected event
#         self.running_buffer = []  # Running buffer to store frames continuously
#         self.audio_frames = []
#         self.last_alert_time = time.time()

#         # Initialize audio stream
#         self.audio_stream = self._initialize_audio()
#         self.request = request  # Save the request object

#     def open_video_device(self):
#         """
#         Opens the video device by trying the first two indexes.
#         """
#         for index in range(2):  # Try first two indexes
#             self.video = cv2.VideoCapture(index)
#             if self.video.isOpened():
#                 print(f"Video device at index {index} opened successfully.")
#                 return
#             else:
#                 print(f"Failed to initialize camera at index {index}: Error: Could not open video device at index {index}.")
#         self.video = None

#     def _initialize_model(self):
#         """
#         Initializes the feature extraction model.
#         """
#         self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#         self.model = self._build_feature_extractor(self.base_model)

#     def _build_feature_extractor(self, base_model):
#         """
#         Builds a custom feature extraction model.

#         Parameters:
#         base_model (Model): Base model for feature extraction.

#         Returns:
#         Model: Custom feature extraction model.
#         """
#         from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(1024, activation='relu')(x)
#         predictions = Dense(128, activation='relu')(x)
#         return Model(inputs=base_model.input, outputs=predictions)

#     def __del__(self):
#         """
#         Releases resources when the object is deleted.
#         """
#         if self.video:
#             self.video.release()
#         self.save_timer.cancel()
#         self.executor.shutdown(wait=False)
#         if self.audio_stream:
#             self.audio_stream.stop_stream()
#             self.audio_stream.close()

#     def _preprocess_image(self, img):
#         """
#         Preprocesses the image for feature extraction.

#         Parameters:
#         img (np.ndarray): Image to preprocess.

#         Returns:
#         np.ndarray: Preprocessed image.
#         """
#         from tensorflow.keras.applications.resnet50 import preprocess_input
#         if img is None or img.size == 0:
#             return None
#         img = cv2.resize(img, (224, 224))
#         img_array = np.array(img, dtype='float32')
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         return img_array

#     def _extract_features(self, img_array):
#         """
#         Extracts features from the preprocessed image.

#         Parameters:
#         img_array (np.ndarray): Preprocessed image array.

#         Returns:
#         np.ndarray: Extracted features.
#         """
#         if img_array is None:
#             return None
#         features = self.model.predict(img_array)
#         return features.flatten()

#     def _detect_faces(self, img, confidence_threshold=0.95):
#         """
#         Detects faces in the image using MTCNN.

#         Parameters:
#         img (np.ndarray): Image to detect faces in.
#         confidence_threshold (float): Confidence threshold for face detection.

#         Returns:
#         list: List of detected faces.
#         """
#         small_img = cv2.resize(img, (160, 120))
#         faces = self.detector.detect_faces(small_img)
#         for face in faces:
#             face['box'] = [int(coordinate * 2) for coordinate in face['box']]
#         filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
#         return filtered_faces

#     def load_known_faces(self):
#         """
#         Loads known faces from the specified directory.
#         """
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
#                     logger.debug(f"Loaded known face: {label} with features: {face_features}")
#                 else:
#                     logger.debug(f"Failed to extract features for known face: {label}")


#     def _preprocess_and_extract(self, img):
#         """
#         Detects and extracts features from the face in the image.

#         Parameters:
#         img (np.ndarray): Image to process.

#         Returns:
#         np.ndarray: Extracted features of the face.
#         """
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
#         """
#         Detects movement in the frame.

#         Parameters:
#         frame (np.ndarray): Frame to detect movement in.

#         Returns:
#         tuple: Boolean indicating if movement is detected and the bounding box of the movement.
#         """
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
#         """
#         Captures a frame from the video device and processes it for movement and face detection.

#         Returns:
#         bytes: Encoded frame in JPEG format.
#         """
#         if not self.video:
#             return None
#         success, image = self.video.read()
#         if not success:
#             return None

#         self.frame_count += 1
#         if self.frame_count % self.frame_skip_interval != 0:
#             return None

#         image = cv2.resize(image, (320, 240))

#         movement_detected, movement_box = self.detect_movement(image)
#         if movement_detected:
#             with self.lock:
#                 self.frames.append(image.copy())

#             x, y, width, height = movement_box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
#             cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
#             self.frame_buffer.append(image.copy())
#             self.running_buffer.append(image.copy())
#             self.log_event("Movement detected")
#             print("Get frame call, Movement detected")

#         detected_faces = []
#         with self.lock:
#             for face in self.detected_faces:
#                 detected_faces.append(face)

#         for face in detected_faces:
#             x, y, width, height = face['box']
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
#             label = face.get('label', 'Unknown')
#             cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)

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
#         """
#         Processes frames for face recognition in a separate thread.
#         """
#         while True:
#             if not self.frames:
#                 time.sleep(0.01)
#                 continue

#             with self.lock:
#                 frame = self.frames.pop(0)

#             self.face_recognition_counter += 1
#             if self.face_recognition_counter >= self.face_recognition_interval:
#                 self._recognize_faces(frame)
#                 self.face_recognition_counter = 0

#     def _recognize_faces(self, frame):
#         """
#         Recognizes faces in the frame and logs the detected faces.
#         """
#         gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
#         faces = self._detect_faces(gray_image_3ch)
#         recognized_faces = []
#         for face in faces:
#             x, y, width, height = face['box']
#             if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
#                 continue
#             face_img = frame[y:y+height, x:x+width]
#             face_array = self._preprocess_image(face_img)
#             if face_array is None:
#                 continue
#             features = self._extract_features(face_array)
#             min_distance = float('inf')
#             label = 'Unknown'
#             logger.debug(f"Extracted features for detected face: {features}")
#             for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
#                 distance = euclidean(features, known_features)
#                 logger.debug(f"Distance to known face {known_label}: {distance}")
#                 if distance < min_distance:
#                     min_distance = distance
#                     label = known_label
#             logger.debug(f"Min distance: {min_distance}, Threshold: {RECOGNITION_THRESHOLD}")
#             if min_distance > RECOGNITION_THRESHOLD:
#                 label = 'Unknown'
#             face['label'] = label
#             recognized_faces.append(face)

#             # Save the face image
#             self.save_face_image(face_img, face['label'])

#         with self.lock:
#             self.detected_faces = recognized_faces
#         with self.lock:
#             self.detected_faces = recognized_faces
#     def log_event(self, event):
#         """
#         Logs an event and sends an email snapshot if necessary.

#         Parameters:
#         event (str): Event description.
#         """
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_entry = f"[{timestamp}] {event}"
#         self.alert_buffer.append(log_entry)
#         print("VideoCamera logged event:", log_entry)
#         if time.time() - self.last_alert_time >= self.alert_interval:
#             self.send_email_snapshot()
#             self.last_alert_time = time.time()



#     def send_email_snapshot(self):
#         """
#         Sends an email with a snapshot of the detected event.

#         If the alert buffer is empty, the method returns without sending an email.
#         The method attempts to retrieve email settings for the user and uses them to send an email
#         containing the alerts and any detected faces. It includes up to two representative frames
#         from the frame buffer as attachments.

#         If any error occurs during the process, it is caught and printed.

#         Attributes:
#             alert_buffer (list): Buffer containing alert messages to be sent in the email.
#             request (HttpRequest): Django HttpRequest object to retrieve the user for email settings.
#             detected_faces (list): List of detected faces with their labels.
#             frame_buffer (list): Buffer containing frames of detected events.
#         """
#         if not self.alert_buffer:
#             return
#         try:
#             if self.request:
#                 # Attempt to retrieve the email settings for the current user
#                 try:
#                     email_settings = EmailSettings.objects.get(user=self.request.user)
#                 except EmailSettings.DoesNotExist:
#                     # If no settings exist for the user, log the issue and exit
#                     print("Email settings not found for the user. Please configure email settings.")
#                     return
#             else:
#                 print("Request object is not available.")
#                 return

#             print(f"Email Settings: {email_settings.__dict__}")  # Debug statement

#             smtp_server = email_settings.smtp_server
#             smtp_port = email_settings.smtp_port
#             smtp_user = email_settings.smtp_user
#             smtp_password = email_settings.smtp_password
#             from_email = smtp_user
#             to_email = email_settings.email
#             subject = "Motion Detection Alert Snapshot"
#             body = "\n".join(self.alert_buffer)
#             msg = MIMEMultipart()
#             msg['From'] = from_email
#             msg['To'] = to_email
#             msg['Subject'] = subject

#             if self.detected_faces:
#                 body += "\n\nDetected Faces:\n"
#                 for i, face in enumerate(self.detected_faces):
#                     label = face.get('label', 'Unknown')
#                     body += f"Person {i + 1}: {label}\n"

#             msg.attach(MIMEText(body, 'plain'))

#             selected_frames = self.select_representative_frames(self.frame_buffer, 2)

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

#             self.alert_buffer = []
#             self.frame_buffer = []
#             print("Email sent successfully")
#         except Exception as e:
#             print(f"Failed to send snapshot email: {str(e)}")


            
#     def select_representative_frames(self, frames, num_frames):
#         """
#         Selects representative frames from the buffer.

#         Parameters:
#         frames (list): List of frames.
#         num_frames (int): Number of frames to select.

#         Returns:
#         list: List of selected frames.
#         """
#         if len(frames) <= num_frames:
#             return frames
#         interval = len(frames) // num_frames
#         selected_frames = [frames[i * interval] for i in range(num_frames)]
#         return selected_frames

#     def save_face_image(self, face_img, label):
#         """
#         Saves the face image to the specified directory.

#         Parameters:
#         face_img (np.ndarray): Face image to save.
#         label (str): Label of the face.
#         """
#         faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')
#         if not os.path.exists(faces_seen_dir):
#             os.makedirs(faces_seen_dir)

#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"{label}_{timestamp}.jpg"
#         filepath = os.path.join(faces_seen_dir, filename)

#         cv2.imwrite(filepath, face_img)
#         print(f"Face image saved: {filepath}")

#         Face.objects.create(name=label, image=f"faces_seen/{filename}")
#         print(f"Face record saved: {label}, {filename}")

#     def save_running_buffer_clip(self):
#         """
#         Saves a video clip from the running buffer.
#         """
#         if self.running_buffer:
#             event_clips_dir = os.path.join(settings.MEDIA_ROOT, 'event_clips')
#             if not os.path.exists(event_clips_dir):
#                 os.makedirs(event_clips_dir)

#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             video_filename = f"event_{timestamp}.mp4"
#             video_file_path = os.path.join(event_clips_dir, video_filename)

#             fps = 20
#             out = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))

#             for frame in self.running_buffer:
#                 out.write(frame)
#             out.release()

#             # Save audio clip
#             audio_filename = f"audio_{timestamp}.wav"
#             audio_file_path = os.path.join(event_clips_dir, audio_filename)
#             self.save_audio_clip(audio_file_path)

#             # Combine audio and video
#             final_filename = f"final_event_{timestamp}.mp4"
#             final_file_path = os.path.join(event_clips_dir, final_filename)
#             self.combine_audio_video(video_file_path, audio_file_path, final_file_path)

#             event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{final_filename}')
#             event.save()

#             self.running_buffer = []
#             print(f"Running buffer clip saved: {final_file_path}")

#             self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#             self.save_timer.start()

#     def _initialize_audio(self):
#         """
#         Initializes the audio stream for recording.

#         Returns:
#         pyaudio.Stream: Initialized audio stream.
#         """
#         try:
#             audio = pyaudio.PyAudio()
#             stream = audio.open(format=pyaudio.paInt16,
#                                 channels=1,
#                                 rate=44100,
#                                 input=True,
#                                 frames_per_buffer=1024)
#             print("Audio stream initialized")
#             return stream
#         except Exception as e:
#             print(f"Error initializing audio: {e}")
#             return None

#     def save_audio_clip(self, file_path):
#         """
#         Saves the recorded audio clip to the specified file path.

#         Parameters:
#         file_path (str): Path to save the audio clip.
#         """
#         if not self.audio_stream:
#             print("Audio stream is not initialized. Cannot save audio clip.")
#             return
#         try:
#             audio = pyaudio.PyAudio()
#             wf = wave.open(file_path, 'wb')
#             wf.setnchannels(1)
#             wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#             wf.setframerate(44100)
#             wf.writeframes(b''.join(self.audio_frames))
#             wf.close()
#             print(f"Audio clip saved: {file_path}")
#         except Exception as e:
#             print(f"Error saving audio clip: {e}")

#     def combine_audio_video(self, video_path, audio_path, output_path):
#         """
#         Combines the audio and video into a single file.

#         Parameters:
#         video_path (str): Path to the video file.
#         audio_path (str): Path to the audio file.
#         output_path (str): Path to save the combined file.
#         """
#         try:
#             command = f"ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path}"
#             subprocess.call(command, shell=True)
#             print(f"Audio and video combined: {output_path}")
#         except Exception as e:
#             print(f"Error combining audio and video: {e}")


