from django import forms
from .models import Face
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, User
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'role']

class CustomAuthenticationForm(AuthenticationForm):
    pass

class TagFaceForm(forms.ModelForm):
    class Meta:
        model = Face
        fields = ['name', 'image']

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm password')

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('Passwords don\'t match.')
        return cd['password2']
    
class UploadFaceForm(forms.ModelForm):
    class Meta:
        model = Face
        fields = ['name', 'image']