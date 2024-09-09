from camera.audio_source import AudioSource

def main():
    # List available audio devices
    print("Available ALSA audio devices:")
    available_devices = AudioSource.list_usable_audio_devices()
    for device in available_devices:
        print(f"- {device}")
    
    # Initialize the AudioSource class
    print("\nInitializing AudioSource...")
    audio_source = AudioSource()  # It will try to use 'sysdefault:CARD=webcam' or fallback to 'default'

    # Check which device was initialized
    if audio_source.get_device_name():
        print(f"Audio source initialized using device: {audio_source.get_device_name()}")
    else:
        print("No usable audio device found.")
    
    # Start capturing audio (this runs in a separate thread)
    if audio_source.inp:
        audio_source.start()
        print("Audio capture started. Press Ctrl+C to stop.")
        
        # Run for 5 seconds then stop the capture
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("Audio capture interrupted.")
        
        # Stop capturing
        audio_source.stop()
        print("Audio capture stopped.")
    else:
        print("Audio capture not started due to lack of a valid audio device.")

if __name__ == "__main__":
    main()
