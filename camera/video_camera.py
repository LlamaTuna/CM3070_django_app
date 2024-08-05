import threading
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .movement_detection import MovementDetection
from .facial_recognition import FacialRecognition
from .send_email import SendEmail
# from .save_audio import SaveAudio
import pytz
import os
from django.conf import settings
from .models import Event

class VideoCamera:
    def __init__(self, camera_index=0, resolution=(320, 240), request=None):
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print(f"Error: Could not open video device {camera_index}.")
            self.video = None
            return

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.movement_detection = MovementDetection()
        self.facial_recognition = FacialRecognition()
        self.send_email = SendEmail(request)
        # self.save_audio = SaveAudio()

        self.frame_skip_interval = 8
        self.frame_count = 0

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

        self.buffer_lock = threading.Lock()
        self.frame_buffer = []
        self.buffer_size = 10  # Adjust buffer size as needed

        self.face_recognition_executor = ThreadPoolExecutor(max_workers=1)
        self.face_recognition_executor.submit(self._background_face_recognition)

        self.email_executor = ThreadPoolExecutor(max_workers=1)  # Executor for email sending

        # New attributes for continuous frame capture
        self.frame = None
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Buffer for movement detection to reduce flickering
        self.movement_buffer = []
        self.movement_buffer_size = 5  # Adjust as needed

    def __del__(self):
        if self.video:
            self.video.release()
        if hasattr(self, 'save_timer'):
            self.save_timer.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'face_recognition_executor'):
            self.face_recognition_executor.shutdown(wait=False)
        if hasattr(self, 'email_executor'):
            self.email_executor.shutdown(wait=False)
        # if hasattr(self, 'save_audio') and self.save_audio.audio_stream:
        #     self.save_audio.audio_stream.stop_stream()
        #     self.save_audio.audio_stream.close()

    def _capture_loop(self):
        while self.video and self.video.isOpened():
            success, image = self.video.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(image, (320, 240))
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            image = self.frame.copy()

        self.frame_count += 1

        # Run movement detection on every frame
        movement_detected, movement_box = self.movement_detection.detect_movement(image)
        if movement_detected:
            with self.lock:
                self.frames.append(image.copy())
            self.movement_buffer.append(movement_box)
            if len(self.movement_buffer) > self.movement_buffer_size:
                self.movement_buffer.pop(0)
            x, y, width, height = self._average_movement_boxes(self.movement_buffer)
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
            cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            with self.buffer_lock:
                self.frame_buffer.append(image.copy())
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
            self.running_buffer.append(image.copy())
            self.send_email.log_event("Movement detected")
            if time.time() - self.last_alert_time >= self.alert_interval:
                self.send_email.frame_buffer = self.frame_buffer.copy()
                self.email_executor.submit(self.send_email.send_email_snapshot)  # Send email asynchronously
                print("Email sent from VC class")
                self.last_alert_time = time.time()
        else:
            if self.frame_count % self.frame_skip_interval != 0:
                return None

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

    def _average_movement_boxes(self, boxes):
        if not boxes:
            return 0, 0, 0, 0
        x = sum(box[0] for box in boxes) // len(boxes)
        y = sum(box[1] for box in boxes) // len(boxes)
        width = sum(box[2] for box in boxes) // len(boxes)
        height = sum(box[3] for box in boxes) // len(boxes)
        return x, y, width, height

    def _process_frames(self):
        while True:
            if not self.frames:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.frames.pop(0)

            with self.buffer_lock:
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)

    def _background_face_recognition(self):
        while True:
            with self.buffer_lock:
                if not self.frame_buffer:
                    time.sleep(0.01)
                    continue
                frame = self.frame_buffer.pop(0)

            recognized_faces = self.facial_recognition.recognize_faces(frame)
            with self.lock:
                self.detected_faces = recognized_faces
            self.send_email.set_detected_faces(recognized_faces)

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
            # self.save_audio.save_audio_clip(audio_file_path)

            final_filename = f"final_event_{timestamp}.mp4"
            final_file_path = os.path.join(event_clips_dir, final_filename)
            # self.save_audio.combine_audio_video(video_file_path, audio_file_path, final_file_path)

            event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{final_filename}')
            event.save()

            self.running_buffer = []
            self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
            self.save_timer.start()



'''gemini code'''
# import threading
# import cv2
# import time
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
# from .movement_detection import MovementDetection
# from .facial_recognition import FacialRecognition
# from .send_email import SendEmail
# from .save_audio import SaveAudio
# import pytz
# import os
# from django.conf import settings
# from .models import Event

# class VideoCamera:
#     def __init__(self, camera_index=0, resolution=(320, 240), request=None):
#         self.video = cv2.VideoCapture(camera_index)
#         if not self.video.isOpened():
#             print(f"Error: Could not open video device {camera_index}.")
#             self.video = None
#             return  # Exit the constructor if the camera fails to open
#         else:
#             self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#             self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

#         self.movement_detection = MovementDetection()
#         self.facial_recognition = FacialRecognition()
#         self.send_email = SendEmail(request)
#         self.save_audio = SaveAudio()

#         self.frame_skip_interval = 8
#         self.frame_count = 0

#         self.lock = threading.Lock()
#         self.frames = []
#         self.detected_faces = []

#         self.executor = ThreadPoolExecutor(max_workers=1)
#         self.executor.submit(self._process_frames)

#         self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#         self.save_timer.start()

#         self.frame_buffer = []
#         self.running_buffer = []
#         self.last_alert_time = time.time()
#         self.alert_interval = 30  # 30 seconds

#         self.buffer_lock = threading.Lock()
#         self.frame_buffer = []
#         self.buffer_size = 10  # Adjust buffer size as needed

#         self.face_recognition_executor = ThreadPoolExecutor(max_workers=1)
#         self.face_recognition_executor.submit(self._background_face_recognition)

#         # New attributes for continuous frame capture
#         self.frame = None
#         self.capture_thread = threading.Thread(target=self._capture_loop)
#         self.capture_thread.daemon = True
#         self.capture_thread.start()

#     def __del__(self):
#         if self.video:
#             self.video.release()
#         if hasattr(self, 'save_timer'):
#             self.save_timer.cancel()
#         self.executor.shutdown(wait=False)
#         self.face_recognition_executor.shutdown(wait=False)
#         if self.save_audio.audio_stream:
#             self.save_audio.audio_stream.stop_stream()
#             self.save_audio.audio_stream.close()

#     def _capture_loop(self):
#         while self.video and self.video.isOpened():
#             success, image = self.video.read()
#             if success:
#                 with self.lock:
#                     self.frame = cv2.resize(image, (320, 240))
#             else:
#                 time.sleep(0.1)  # Increased sleep interval

#     def get_frame(self):
#         with self.lock:
#             if self.frame is None:
#                 return None
#             image = self.frame.copy()

#         self.frame_count += 1
#         if self.frame_count % self.frame_skip_interval != 0:
#             return None

#         movement_detected, movement_box = self.movement_detection.detect_movement(image)
#         if movement_detected:
#             with self.lock:
#                 self.frames.append(image.copy())
#             x, y, width, height = movement_box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
#             cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
#             with self.buffer_lock:
#                 self.frame_buffer.append(image.copy())
#                 if len(self.frame_buffer) > self.buffer_size:
#                     self.frame_buffer.pop(0)
#             self.running_buffer.append(image.copy())
#             self.send_email.log_event("Movement detected")
#             if time.time() - self.last_alert_time >= self.alert_interval:
#                 self.send_email.frame_buffer = self.frame_buffer.copy()
#                 self.send_email.send_email_snapshot()
#                 print("Email sent from VC class")
#                 self.last_alert_time = time.time()

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
#         while True:
#             if not self.frames:
#                 time.sleep(0.01)
#                 continue

#             with self.lock:
#                 frame = self.frames.pop(0)

#             with self.buffer_lock:
#                 self.frame_buffer.append(frame)
#                 if len(self.frame_buffer) > self.buffer_size:
#                     self.frame_buffer.pop(0)
            
#             time.sleep(0.01)  # Added sleep interval

#     def _background_face_recognition(self):
#         while True:
#             with self.buffer_lock:
#                 if not self.frame_buffer:
#                     time.sleep(0.01)
#                     continue
#                 frame = self.frame_buffer.pop(0)

#             recognized_faces = self.facial_recognition.recognize_faces(frame)
#             with self.lock:
#                 self.detected_faces = recognized_faces
#             self.send_email.set_detected_faces(recognized_faces)
            
#             time.sleep(0.01)  # Added sleep interval

#     def save_running_buffer_clip(self):
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

#             audio_filename = f"audio_{timestamp}.wav"
#             audio_file_path = os.path.join(event_clips_dir, audio_filename)
#             self.save_audio.save_audio_clip(audio_file_path)

#             final_filename = f"final_event_{timestamp}.mp4"
#             final_file_path = os.path.join(event_clips_dir, final_filename)
#             self.save_audio.combine_audio_video(video_file_path, audio_file_path, final_file_path)

#             event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{final_filename}')
#             event.save()

#             self.running_buffer = []
#             self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#             self.save_timer.start()

'''original code'''
# import threading
# import cv2
# import time
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
# from .movement_detection import MovementDetection
# from .facial_recognition import FacialRecognition
# from .send_email import SendEmail
# from .save_audio import SaveAudio
# import pytz
# import os
# from django.conf import settings
# from .models import Event

# import threading
# import cv2
# import time
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
# from .movement_detection import MovementDetection
# from .facial_recognition import FacialRecognition
# from .send_email import SendEmail
# from .save_audio import SaveAudio
# import pytz
# import os
# from django.conf import settings
# from .models import Event

# class VideoCamera:
#     def __init__(self, camera_index=0, resolution=(320, 240), request=None):
#         self.video = cv2.VideoCapture(camera_index)
#         if not self.video.isOpened():
#             print(f"Error: Could not open video device {camera_index}.")
#             self.video = None
#         else:
#             self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#             self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

#         self.movement_detection = MovementDetection()
#         self.facial_recognition = FacialRecognition()
#         self.send_email = SendEmail(request)
#         self.save_audio = SaveAudio()

#         self.frame_skip_interval = 8
#         self.frame_count = 0

#         self.lock = threading.Lock()
#         self.frames = []
#         self.detected_faces = []

#         self.executor = ThreadPoolExecutor(max_workers=1)
#         self.executor.submit(self._process_frames)

#         self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#         self.save_timer.start()

#         self.frame_buffer = []
#         self.running_buffer = []
#         self.last_alert_time = time.time()
#         self.alert_interval = 30  # 30 seconds

#         self.buffer_lock = threading.Lock()
#         self.frame_buffer = []
#         self.buffer_size = 10  # Adjust buffer size as needed

#         self.face_recognition_executor = ThreadPoolExecutor(max_workers=1)
#         self.face_recognition_executor.submit(self._background_face_recognition)

#         # New attributes for continuous frame capture
#         self.frame = None
#         self.capture_thread = threading.Thread(target=self._capture_loop)
#         self.capture_thread.daemon = True
#         self.capture_thread.start()

#     def __del__(self):
#         if self.video:
#             self.video.release()
#         if hasattr(self, 'save_timer'):
#             self.save_timer.cancel()
#         self.executor.shutdown(wait=False)
#         self.face_recognition_executor.shutdown(wait=False)
#         if self.save_audio.audio_stream:
#             self.save_audio.audio_stream.stop_stream()
#             self.save_audio.audio_stream.close()

#     def _capture_loop(self):
#         while self.video and self.video.isOpened():
#             success, image = self.video.read()
#             if success:
#                 with self.lock:
#                     self.frame = cv2.resize(image, (320, 240))
#             else:
#                 time.sleep(0.01)

#     def get_frame(self):
#         with self.lock:
#             if self.frame is None:
#                 return None
#             image = self.frame.copy()

#         self.frame_count += 1
#         if self.frame_count % self.frame_skip_interval != 0:
#             return None

#         movement_detected, movement_box = self.movement_detection.detect_movement(image)
#         if movement_detected:
#             with self.lock:
#                 self.frames.append(image.copy())
#             x, y, width, height = movement_box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
#             cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
#             with self.buffer_lock:
#                 self.frame_buffer.append(image.copy())
#                 if len(self.frame_buffer) > self.buffer_size:
#                     self.frame_buffer.pop(0)
#             self.running_buffer.append(image.copy())
#             self.send_email.log_event("Movement detected")
#             if time.time() - self.last_alert_time >= self.alert_interval:
#                 self.send_email.frame_buffer = self.frame_buffer.copy()
#                 self.send_email.send_email_snapshot()
#                 print("Email sent from VC class")
#                 self.last_alert_time = time.time()

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
#         while True:
#             if not self.frames:
#                 time.sleep(0.01)
#                 continue

#             with self.lock:
#                 frame = self.frames.pop(0)

#             with self.buffer_lock:
#                 self.frame_buffer.append(frame)
#                 if len(self.frame_buffer) > self.buffer_size:
#                     self.frame_buffer.pop(0)

#     def _background_face_recognition(self):
#         while True:
#             with self.buffer_lock:
#                 if not self.frame_buffer:
#                     time.sleep(0.01)
#                     continue
#                 frame = self.frame_buffer.pop(0)

#             recognized_faces = self.facial_recognition.recognize_faces(frame)
#             with self.lock:
#                 self.detected_faces = recognized_faces
#             self.send_email.set_detected_faces(recognized_faces)

#     def save_running_buffer_clip(self):
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

#             audio_filename = f"audio_{timestamp}.wav"
#             audio_file_path = os.path.join(event_clips_dir, audio_filename)
#             self.save_audio.save_audio_clip(audio_file_path)

#             final_filename = f"final_event_{timestamp}.mp4"
#             final_file_path = os.path.join(event_clips_dir, final_filename)
#             self.save_audio.combine_audio_video(video_file_path, audio_file_path, final_file_path)

#             event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{final_filename}')
#             event.save()

#             self.running_buffer = []
#             self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
#             self.save_timer.start()