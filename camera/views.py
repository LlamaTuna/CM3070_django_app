from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.models import Model
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import euclidean
import os
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
from datetime import datetime

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.previous_frame = None
        self.detector = MTCNN()
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = self._build_feature_extractor(self.base_model)
        self.known_faces_features = []
        self.known_faces_labels = []
        self.load_known_faces("known_faces")
        self.detection_log = {}
        self.detection_interval = 5  # seconds
        self.alert_buffer = []
        self.frame_buffer = []
        self.alert_interval = 300  # 5 minutes (300 seconds)
        self.last_alert_time = time.time()

    def __del__(self):
        self.video.release()

    def _build_feature_extractor(self, base_model):
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(128, activation='relu')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def _preprocess_image(self, img):
        img = cv2.resize(img, (224, 224))
        img_array = np.array(img, dtype='float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _extract_features(self, img_array):
        features = self.model.predict(img_array)
        return features.flatten()

    def _detect_faces(self, img, confidence_threshold=0.95):
        faces = self.detector.detect_faces(img)
        filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
        return filtered_faces

    def load_known_faces(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                label = os.path.splitext(filename)[0]
                img = cv2.imread(img_path)
                face_features = self._preprocess_and_extract(img)
                if face_features is not None:
                    self.known_faces_features.append(face_features)
                    self.known_faces_labels.append(label)

    def _preprocess_and_extract(self, img):
        faces = self._detect_faces(img)
        if faces:
            x, y, width, height = faces[0]['box']
            face = img[y:y+height, x:x+width]
            face_array = self._preprocess_image(face)
            features = self._extract_features(face_array)
            return features
        return None

    def detect_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

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

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale image back to 3 channels to match the input shape of MTCNN
        gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Detect movement
        movement_detected, movement_box = self.detect_movement(gray_image_3ch)
        if movement_detected:
            x, y, width, height = movement_box
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            self.frame_buffer.append(image.copy())  # Store the frame
            self.log_event("Movement detected")  # Log movement event

        # Detect faces
        faces = self._detect_faces(gray_image_3ch)
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            face_img = image[y:y+height, x:x+width]
            face_array = self._preprocess_image(face_img)
            features = self._extract_features(face_array)

            min_distance = float('inf')
            label = None
            for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
                distance = euclidean(features, known_features)
                if distance < min_distance:
                    min_distance = distance
                    label = known_label

            # Log faces detected with confidence > 95%
            current_time = time.time()
            if label and (label not in self.detection_log or current_time - self.detection_log[label] > self.detection_interval):
                self.detection_log[label] = current_time  # Update the detection time log
                self.log_event(f"{label} detected")
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def log_event(self, event):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {event}"
        self.alert_buffer.append(log_entry)
        # Check if it's time to send the email
        if time.time() - self.last_alert_time >= self.alert_interval:
            self.send_email_snapshot()
            self.last_alert_time = time.time()

    def send_email_snapshot(self):
        if not self.alert_buffer:
            return

        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587  # For starttls
            smtp_user = "cm3070.smtptest@gmail.com"
            smtp_password = "lxvv mgoj dzld gbae"  # Use App Password for better security

            from_email = smtp_user
            to_email = "cm3070.smtptest@gmail.com"
            subject = "Motion Detection Alert Snapshot"
            body = "\n".join(self.alert_buffer)

            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            # Select representative frames to attach
            selected_frames = self.select_representative_frames(self.frame_buffer, 2)  # Select 2 frames

            for i, frame in enumerate(selected_frames):
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_data = img_encoded.tobytes()
                image = MIMEImage(image_data, name=f"event_{i + 1}.jpg")
                msg.attach(image)

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()

            # Clear the buffer after sending
            self.alert_buffer = []
            self.frame_buffer = []
        except Exception as e:
            print(f"Failed to send snapshot email: {str(e)}")

    def select_representative_frames(self, frames, num_frames):
        if len(frames) <= num_frames:
            return frames
        interval = len(frames) // num_frames
        selected_frames = [frames[i * interval] for i in range(num_frames)]
        return selected_frames

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'camera/index.html')

# from django.http import StreamingHttpResponse
# from django.shortcuts import render
# import cv2
# import numpy as np
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, ResNet50
# from keras.models import Model
# from mtcnn.mtcnn import MTCNN
# from scipy.spatial.distance import euclidean
# import os
# import tensorflow as tf
# import time

# class VideoCamera:
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         self.previous_frame = None
#         self.detector = MTCNN()
#         self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#         self.model = self._build_feature_extractor(self.base_model)
#         self.known_faces_features = []
#         self.known_faces_labels = []
#         self.load_known_faces("known_faces")
#         self.detection_log = {}
#         self.detection_interval = 5  # seconds

#     def __del__(self):
#         self.video.release()

#     def _build_feature_extractor(self, base_model):
#         x = base_model.output
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         x = tf.keras.layers.Dense(1024, activation='relu')(x)
#         predictions = tf.keras.layers.Dense(128, activation='relu')(x)
#         return Model(inputs=base_model.input, outputs=predictions)

#     def _preprocess_image(self, img):
#         img = cv2.resize(img, (224, 224))
#         img_array = np.array(img, dtype='float32')
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         return img_array

#     def _extract_features(self, img_array):
#         features = self.model.predict(img_array)
#         return features.flatten()

#     def _detect_faces(self, img, confidence_threshold=0.95):
#         faces = self.detector.detect_faces(img)
#         filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
#         return filtered_faces

#     def load_known_faces(self, directory):
#         for filename in os.listdir(directory):
#             if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#                 img_path = os.path.join(directory, filename)
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
#         success, image = self.video.read()

#         # Detect movement
#         movement_detected, movement_box = self.detect_movement(image)
#         if movement_detected:
#             x, y, width, height = movement_box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
#             cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#         # Detect faces
#         faces = self._detect_faces(image)
#         for face in faces:
#             x, y, width, height = face['box']
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
#             face_img = image[y:y+height, x:x+width]
#             face_array = self._preprocess_image(face_img)
#             features = self._extract_features(face_array)

#             min_distance = float('inf')
#             label = None
#             for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
#                 distance = euclidean(features, known_features)
#                 if distance < min_distance:
#                     min_distance = distance
#                     label = known_label

#             # Log faces detected with confidence > 95%
#             current_time = time.time()
#             if label and (label not in self.detection_log or current_time - self.detection_log[label] > self.detection_interval):
#                 self.detection_log[label] = current_time  # Update the detection time log
#                 cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# def video_feed(request):
#     return StreamingHttpResponse(gen(VideoCamera()),
#                                  content_type='multipart/x-mixed-replace; boundary=frame')

# def index(request):
#     return render(request, 'camera/index.html')
