# camera/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('faces/', views.list_faces, name='list_faces'),
    path('tag_face/<int:face_id>/', views.tag_face, name='tag_face'),
]
