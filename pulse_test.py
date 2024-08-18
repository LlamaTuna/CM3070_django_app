# Description: This script is used to test the PulseAudioManager class.
from camera.pulse_audio_manager import PulseAudioManager

if __name__ == "__main__":
    pulse_manager = PulseAudioManager()
    sources = pulse_manager.list_audio_sources()

    # Replace 'some_source_name' with an actual source name from the listed sources
    valid_source_name = 'alsa_input.usb-SunplusIT_Inc_Depstech_webcam_J20220909-02.iec958-stereo'
    
    pulse_manager.select_audio_source(valid_source_name)  # Use a valid source name from the list
    selected_source = pulse_manager.get_selected_source()
    print(f"Selected audio source: {selected_source}")
