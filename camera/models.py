# camera/models.py
from django.contrib.auth.models import AbstractUser, Group, Permission, User
from django.db import models
from django.conf import settings

class CustomUser(AbstractUser):
    """
    Custom user model extending Django's AbstractUser. Adds a role field and modifies
    the groups and user_permissions fields to use custom related names and query names.
    """
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('viewer', 'Viewer'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='viewer')

    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',
        blank=True,
        help_text=('The groups this user belongs to. A user will get all permissions '
                   'granted to each of their groups.'),
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser',
    )

class Event(models.Model):
    """
    Model representing an event, which includes a timestamp, event type, description,
    and an optional video clip associated with the event.
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=100)
    description = models.TextField()
    clip = models.FileField(upload_to='event_clips/', null=True, blank=True)

class Face(models.Model):
    """
    Model representing a face, which includes the name, timestamp of when it was recorded,
    the image file, and whether it has been tagged or not.
    """
    name = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='faces_seen/')
    tagged = models.BooleanField(default=False)


class EmailSettings(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    email = models.EmailField()
    smtp_server = models.CharField(max_length=100)
    smtp_port = models.IntegerField()
    smtp_user = models.CharField(max_length=100)
    smtp_password = models.CharField(max_length=100)