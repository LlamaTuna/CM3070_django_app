import unittest
from unittest.mock import patch
from .video_camera import VideoCamera
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model  # Use this to get the User model
from django.utils.crypto import get_random_string
from django.test import TestCase
from django.urls import reverse

User = get_user_model()  # Assign the User model

class TestVideoCameraInitialization(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_camera_initialization_success(self, mock_video_capture):
        # Simulate successful camera initialization
        mock_video_capture.return_value.isOpened.return_value = True
        
        # Create an instance of VideoCamera
        camera = VideoCamera(camera_index=0)
        
        # Assert that the camera is initialized successfully
        self.assertTrue(camera.initialized)
        mock_video_capture.assert_called_with(0)
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization_failure(self, mock_video_capture):
        # Simulate camera initialization failure
        mock_video_capture.return_value.isOpened.return_value = False
        
        # Create an instance of VideoCamera
        camera = VideoCamera(camera_index=0)
        
        # Assert that the camera is not initialized successfully
        self.assertFalse(camera.initialized)
        mock_video_capture.assert_called_with(0)


class UserAuthTests(TestCase):

    def generate_password(self):
        """Generate a random password that meets the complexity requirements."""
        return get_random_string(length=12, allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()')

    def test_user_registration(self):
        password = self.generate_password()

        response = self.client.post(reverse('register'), {
            'username': 'testuser',
            'email': 'testuser@example.com',
            'password1': password,
            'password2': password,
            'role': 'user' 
        })

        # Debugging information
        if response.status_code == 200:
            form_errors = response.context['form'].errors
            print("Form errors during registration:", form_errors)
        else:
            print("Registration successful with status code:", response.status_code)

        # Expecting a redirect after successful registration
        self.assertEqual(response.status_code, 302)

        # Check if the user was created
        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_user_login(self):
        password = self.generate_password()

        User.objects.create_user(username='testuser', password=password)

        response = self.client.post(reverse('login'), {
            'username': 'testuser',
            'password': password
        })

        self.assertEqual(response.status_code, 302)


# class LogTests(TestCase):

#     def test_log_event(self):
#         log_event("Test Event")
#         self.assertIn("Test Event", logs[-1])

#     def test_get_logs(self):
#         log_event("Test Event 1")
#         log_event("Test Event 2")
#         response = self.client.get(reverse('get_logs'))
#         self.assertEqual(response.status_code, 200)
#         self.assertIn("Test Event 1", response.json()['logs'])



# class FaceManagementTests(TestCase):

#     def setUp(self):
#         self.user = User.objects.create_user(username='testuser', password='testpassword')
#         self.client.login(username='testuser', password='testpassword')

#     def test_list_faces(self):
#         response = self.client.get(reverse('list_faces'))
#         self.assertEqual(response.status_code, 200)
#         self.assertTemplateUsed(response, 'camera/list_faces.html')

#     def test_tag_face(self):
#         face = Face.objects.create(name="Test Face", tagged=False)
#         with open('path_to_dummy_image.jpg', 'rb') as img:
#             face.image.save('dummy_image.jpg', File(img), save=True)
#         response = self.client.post(reverse('tag_face', args=[face.id]), {'name': 'Tagged Face'})
#         self.assertEqual(response.status_code, 302)
