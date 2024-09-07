# camera/pulse_audio_manager.py
# Description: This module is responsible for managing PulseAudio sources and includes a fallback mechanism.
import pulsectl
import logging

class PulseAudioManager:
    """
    A class to manage PulseAudio sources, including listing available sources,
    selecting a source, and managing the PulseAudio connection. Falls back to default if PulseAudio is unavailable.

    Attributes:
        pulse (pulsectl.Pulse): The PulseAudio client instance.
        selected_source (str): The name of the currently selected audio source.
        cached_sources (list): Cached list of available audio sources.
    """

    def __init__(self):
        """
        Initializes the PulseAudioManager, attempts to set up the PulseAudio connection.
        Falls back to default audio source if PulseAudio connection fails.
        """
        print("Initializing PulseAudioManager...")
        self.pulse = None
        self.selected_source = None
        self.cached_sources = None  # Cache the list of sources

        try:
            self.pulse = pulsectl.Pulse('audio-capture')
            print("PulseAudio connection established.")
        except pulsectl.PulseError:
            logging.error("Failed to connect to PulseAudio. Falling back to default audio source.")
            self.pulse = None  # Set to None if PulseAudio connection fails

    def list_audio_sources(self):
        """
        Lists available audio sources from PulseAudio. Caches the list to avoid redundant calls.
        If PulseAudio is unavailable, returns an empty list.

        Returns:
            list: A list of PulseAudio sources, or an empty list if PulseAudio is unavailable.
        """
        if not self.pulse:
            logging.warning("PulseAudio is not available. Returning empty source list.")
            return []

        if not self.cached_sources:
            print("Listing audio sources...")
            self.cached_sources = self.pulse.source_list()
            for source in self.cached_sources:
                print(f"Found source: {source.name} - {source.description}")
        return self.cached_sources

    def select_audio_source(self, source_name):
        """
        Selects an audio source by name from the list of available sources.
        If PulseAudio is unavailable, this function simply returns without selecting a source.

        Args:
            source_name (str): The name of the audio source to select.

        Raises:
            ValueError: If the specified audio source is not found.
        """
        if not self.pulse:
            logging.warning("PulseAudio is not available. Cannot select an audio source.")
            return

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
        If PulseAudio is unavailable, returns 'default'.

        Returns:
            str: The name of the selected audio source, or 'default' if no source is selected or PulseAudio is unavailable.
        """
        if not self.pulse:
            return 'default'  # Fallback to default if PulseAudio is not available
        return self.selected_source or 'default'

    def close(self):
        """
        Closes the PulseAudio connection managed by this instance, if available.
        """
        if self.pulse:
            print("Closing PulseAudioManager...")
            self.pulse.close()
        else:
            print("No PulseAudio connection to close.")
