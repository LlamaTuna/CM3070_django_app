import alsaaudio
import threading
import numpy as np
import time 


class AudioSource:
    """Class for audio input using ALSA."""
    
    def __init__(self, sample_freq=44100, device=None, threshold=1000):
        """
        Initialize audio capture.
        - Attempts to use 'sysdefault:CARD=webcam' by default.
        - Falls back to the system's 'default' device if 'sysdefault:CARD=webcam' is not available.
        """
        self.sample_freq = sample_freq
        self.device = device or self.get_default_device()
        self.threshold = threshold  # Audio volume threshold for triggering events
        self.inp = None  # Audio input object
        self.started = False  # Capture state
        self.listeners = []  # List of functions to call when an event is triggered
        
        if self.device:
            # Attempt to initialize the audio device
            try:
                self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
                self.inp.setchannels(1)
                self.inp.setrate(self.sample_freq)
                self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
                self.inp.setperiodsize(512)
                print(f"Audio source initialized with device: {self.device}")
            except alsaaudio.ALSAAudioError as e:
                print(f"Error initializing audio device '{self.device}': {e}. Skipping audio.")
                self.inp = None  # Set input to None if initialization fails

    def get_default_device(self):
        """Return 'sysdefault:CARD=webcam' if available, otherwise return 'default'."""
        devices = self.list_usable_audio_devices()
        return 'sysdefault:CARD=webcam' if 'sysdefault:CARD=webcam' in devices else 'default' if 'default' in devices else None

    @staticmethod
    def list_usable_audio_devices():
        """List available and usable ALSA devices."""
        available_devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
        usable_devices = []
        for device in available_devices:
            try:
                # Test device by trying to open it
                pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=device)
                pcm.close()
                usable_devices.append(device)
            except alsaaudio.ALSAAudioError:
                continue
        return usable_devices

    def start(self):
        """Start capturing audio if a valid audio input is available."""
        if self.inp is None:
            print("[!] No audio input initialized, skipping audio capture.")
            return
        
        self.started = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        """Capture audio data with timestamps (runs in a separate thread)."""
        while self.started and self.inp:
            try:
                length, data = self.inp.read()  # Read the audio data from the device
                timestamp = time.time()  # Capture the exact time when the data is captured

                if length > 0:
                    # Convert raw audio data to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio_data).mean()  # Measure audio intensity

                    # Check if audio volume exceeds the threshold
                    if volume > self.threshold:
                        self.trigger_event(volume, timestamp)  # Pass both volume and timestamp

            except alsaaudio.ALSAAudioError as e:
                print(f"Error capturing audio: {e}")
            self.stop()

    def trigger_event(self, volume):
        """Call all listeners when an audio event is triggered."""
        print(f"Audio event triggered! Volume: {volume}")
        for listener in self.listeners:
            listener(volume)  # Call each registered listener with the volume level

    def stop(self):
        """Stop capturing audio."""
        self.started = False
        if self.inp and self.thread:
            if threading.current_thread() != self.thread:  # Ensure the thread is not trying to join itself
                self.thread.join()
            else:
                print("Cannot join the current thread.")

    def get_device_name(self):
        """Return the active ALSA device name, or None if no device is active."""
        return self.device if self.inp else None

    def add_listener(self, listener_func):
        """Add a listener function that will be called when an audio event is triggered."""
        self.listeners.append(listener_func)

