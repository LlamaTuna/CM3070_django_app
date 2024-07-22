from django import forms
from .models import Face
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, User
from .models import CustomUser, EmailSettings

class CustomUserCreationForm(UserCreationForm):
    """
    A form that creates a user, with no privileges, from the given username, email, and password.
    """
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'role']

class CustomAuthenticationForm(AuthenticationForm):
    """
    A form used for user authentication.
    """
    pass

class TagFaceForm(forms.ModelForm):
    """
    A form for tagging faces with a name and an image.
    """
    class Meta:
        model = Face
        fields = ['name', 'image']

class RegisterForm(forms.ModelForm):
    """
    A form for registering a new user, including password confirmation.
    """
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm password')

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_password2(self):
        """
        Verifies that the entered passwords match.
        """
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('Passwords don\'t match.')
        return cd['password2']
    
class UploadFaceForm(forms.ModelForm):
    """
    A form for uploading a face image with a name.
    """
    class Meta:
        model = Face
        fields = ['name', 'image']

class EmailSettingsForm(forms.ModelForm):
    class Meta:
        model = EmailSettings
        fields = ['smtp_server', 'smtp_port', 'smtp_user', 'smtp_password', 'email']