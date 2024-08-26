from rest_framework import serializers

class LogSerializer(serializers.Serializer):
    """
    Serializer for handling log data. This serializer converts log entries
    into a format that can be easily rendered into JSON for API responses
    or parsed from JSON for API requests.

    Attributes:
        timestamp (DateTimeField): The timestamp of when the event occurred.
        event_type (CharField): A string representing the type of event.
        description (CharField): A brief description of the event.
        extra_data (JSONField): Optional additional data related to the event, stored in JSON format.
    """
    timestamp = serializers.DateTimeField()
    event_type = serializers.CharField(max_length=100)
    description = serializers.CharField(max_length=255)
    extra_data = serializers.JSONField(required=False)
