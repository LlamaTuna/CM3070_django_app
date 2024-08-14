from rest_framework import serializers

class LogSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    event_type = serializers.CharField(max_length=100)
    description = serializers.CharField(max_length=255)
    extra_data = serializers.JSONField(required=False)
