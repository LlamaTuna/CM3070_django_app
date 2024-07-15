from django import forms
from .models import Face

class TagFaceForm(forms.ModelForm):
    class Meta:
        model = Face
        fields = ['name', 'image']
