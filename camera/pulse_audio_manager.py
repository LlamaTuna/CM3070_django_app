# Description: This module is responsible for managing PulseAudio sources.
import pulsectl

class PulseAudioManager:
    def __init__(self):
        print("Initializing PulseAudioManager...")
        self.pulse = pulsectl.Pulse('audio-capture')
        self.selected_source = None
        self.cached_sources = None  # Cache the list of sources

    def list_audio_sources(self):
        if not self.cached_sources:
            print("Listing audio sources...")
            self.cached_sources = self.pulse.source_list()
            for source in self.cached_sources:
                print(f"Found source: {source.name} - {source.description}")
        return self.cached_sources

    def select_audio_source(self, source_name):
        sources = self.list_audio_sources()
        for source in sources:
            if source.name == source_name:
                self.selected_source = source.name
                print(f"Selected audio source: {self.selected_source}")
                return
        raise ValueError(f"Audio source {source_name} not found")

    def get_selected_source(self):
        return self.selected_source or 'default'

    def close(self):
        print("Closing PulseAudioManager...")
        self.pulse.close()
