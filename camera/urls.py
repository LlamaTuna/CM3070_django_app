# urls.py
from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('admin/', views.admin_view, name='admin_view'),
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='camera/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('list_faces/', views.list_faces, name='list_faces'),
    path('tag_face/<int:face_id>/', views.tag_face, name='tag_face'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('get_logs/', views.get_logs, name='get_logs'),
    path('upload_face/', views.upload_face, name='upload_face'),
]
