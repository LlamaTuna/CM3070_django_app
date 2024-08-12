#video_camera.py
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
import subprocess

class VideoCamera:
    def __init__(self, camera_index=0, resolution=(320, 240), request=None):
        self.camera_index = camera_index
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print(f"Error: Could not open video device at {camera_index}.")
            self.video = None
            self.initialized = False  # Camera failed to open
            return

        self.initialized = True  # Camera successfully opened
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.movement_detection = MovementDetection()
        self.facial_recognition = FacialRecognition()
        self.send_email = SendEmail(request)
        # self.save_audio = SaveAudio()  

        self.frame_skip_interval = 2
        self.frame_count = 0
        self.face_recognition_interval = 10
        self.face_recognition_counter = 0

        self.lock = threading.Lock()
        self.frames = []
        self.detected_faces = []
        

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self._process_frames)

        self.email_executor = ThreadPoolExecutor(max_workers=1)  # Executor for email sending

        self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
        self.save_timer.start()

        self.frame_buffer = []
        self.running_buffer = []
        self.last_alert_time = time.time()
        self.alert_interval = 30  # 30 seconds

    def __del__(self):
        if self.video:
            self.video.release()
        if hasattr(self, 'save_timer'):
            self.save_timer.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'email_executor'):  # Ensure the email executor is shut down properly
            self.email_executor.shutdown(wait=False)
        # if self.save_audio and hasattr(self.save_audio, 'audio_stream') and self.save_audio.audio_stream:
        #     self.save_audio.audio_stream.stop_stream()
        #     self.save_audio.audio_stream.close()

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

            # Attempt to send email snapshot
            if time.time() - self.last_alert_time >= self.alert_interval:
                self.send_email.log_event("Movement detected")
                self.send_email.frame_buffer = self.frame_buffer.copy()
                self.email_executor.submit(self.send_email.send_email_snapshot)  # Send email asynchronously
                print("Email sent from VC class")
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
                self.send_email.set_detected_faces(recognized_faces)  # Pass detected faces to SendEmail
                self.face_recognition_counter = 0

    def save_running_buffer_clip(self):
        if self.running_buffer:
            event_clips_dir = os.path.join(settings.MEDIA_ROOT, 'event_clips')
            if not os.path.exists(event_clips_dir):
                os.makedirs(event_clips_dir)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"event_{timestamp}.mp4"
            video_file_path = os.path.join(event_clips_dir, video_filename)

            # Adjust frame rate and duration (if needed)
            fps = 6
            duration_seconds = 30  # Length of the snippet in seconds
            expected_frame_count = fps * duration_seconds

            # Use FFmpeg to record both video and audio
            command = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-f', 'rawvideo',  # Format of the input video data
                '-pix_fmt', 'bgr24',  # Pixel format of the input data
                '-s', '320x240',  # Frame size: width x height
                '-r', str(fps),  # Frame rate
                '-i', '-',  # Input comes from a pipe
                '-f', 'pulse',  # Use PulseAudio
                '-i', 'default',  # Default PulseAudio source
                '-c:v', 'libx264',  # Video codec
                '-preset', 'fast',  # FFmpeg preset
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-pix_fmt', 'yuv420p',  # Pixel format for output video
                '-t', str(duration_seconds),  # Set the duration of the output file
                video_file_path
            ]

            # Start FFmpeg process
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            try:
                for i in range(min(expected_frame_count, len(self.running_buffer))):
                    frame = self.running_buffer[i]
                    process.stdin.write(frame.tobytes())
            except Exception as e:
                print(f"Error writing frame to FFmpeg process: {e}")

            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                error_output = process.stderr.read().decode()
                print(f"FFmpeg error: {error_output}")

            # Save event in the database
            event = Event(event_type='Periodic', description='Periodic buffer save', clip=f'event_clips/{video_filename}')
            event.save()

            # Clear the buffer after saving the clip
            self.running_buffer = []
            self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
            self.save_timer.start()
