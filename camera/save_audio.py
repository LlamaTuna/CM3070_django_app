import pyaudio
import wave
import subprocess

class SaveAudio:
    def __init__(self):
        self.audio_stream = self._initialize_audio()
        self.audio_frames = []

    def _initialize_audio(self):
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=44100,
                                input=True,
                                frames_per_buffer=1024)
            print("Audio stream initialized")
            return stream
        except Exception as e:
            print(f"Error initializing audio: {e}")
            return None

    def save_audio_clip(self, file_path):
        if not self.audio_stream:
            print("Audio stream is not initialized. Cannot save audio clip.")
            return
        try:
            audio = pyaudio.PyAudio()
            wf = wave.open(file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            print(f"Audio clip saved: {file_path}")
        except Exception as e:
            print(f"Error saving audio clip: {e}")

    def combine_audio_video(self, video_path, audio_path, output_path):
        try:
            command = f"ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path}"
            subprocess.call(command, shell=True)
            print(f"Audio and video combined: {output_path}")
        except Exception as e:
            print(f"Error combining audio and video: {e}")
