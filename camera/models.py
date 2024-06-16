from django.db import models

class Event(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=100)
    description = models.TextField()

class Face(models.Model):
    name = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='faces/')

