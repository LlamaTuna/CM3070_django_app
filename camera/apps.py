# camera/apps.py


from django.apps import AppConfig

class CameraConfig(AppConfig):
    """
    Configuration class for the 'camera' app.

    This class is used to configure the 'camera' application and provides
    metadata about the app. It is used by Django to include the app in the project.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'camera'
