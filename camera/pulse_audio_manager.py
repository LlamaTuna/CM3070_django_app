#camera/pulse_audio_manager.py
# Description: This module is responsible for managing PulseAudio sources.
import pulsectl

class PulseAudioManager:
    """
    A class to manage PulseAudio sources, including listing available sources,
    selecting a source, and managing the PulseAudio connection.

    Attributes:
        pulse (pulsectl.Pulse): The PulseAudio client instance.
        selected_source (str): The name of the currently selected audio source.
        cached_sources (list): Cached list of available audio sources.
    """

    def __init__(self):
        """
        Initializes the PulseAudioManager, sets up the PulseAudio connection,
        and initializes attributes for selected source and cached sources.
        """
        print("Initializing PulseAudioManager...")
        self.pulse = pulsectl.Pulse('audio-capture')
        self.selected_source = None
        self.cached_sources = None  # Cache the list of sources

    def list_audio_sources(self):
        """
        Lists available audio sources from PulseAudio. Caches the list to avoid redundant calls.

        Returns:
            list: A list of PulseAudio sources.
        """
        if not self.cached_sources:
            print("Listing audio sources...")
            self.cached_sources = self.pulse.source_list()
            for source in self.cached_sources:
                print(f"Found source: {source.name} - {source.description}")
        return self.cached_sources

    def select_audio_source(self, source_name):
        """
        Selects an audio source by name from the list of available sources.

        Args:
            source_name (str): The name of the audio source to select.

        Raises:
            ValueError: If the specified audio source is not found.
        """
        sources = self.list_audio_sources()
        for source in sources:
            if source.name == source_name:
                self.selected_source = source.name
                print(f"Selected audio source: {self.selected_source}")
                return
        raise ValueError(f"Audio source {source_name} not found")

    def get_selected_source(self):
        """
        Gets the name of the currently selected audio source.

        Returns:
            str: The name of the selected audio source, or 'default' if no source is selected.
        """
        return self.selected_source or 'default'

    def close(self):
        """
        Closes the PulseAudio connection managed by this instance.
        """
        print("Closing PulseAudioManager...")
        self.pulse.close()
