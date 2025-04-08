from django import forms
from django.contrib.auth.models import User
from . import models
from quiz.models import Course

class TeacherUserForm(forms.ModelForm):
    class Meta:
        model=User
        fields=['first_name','last_name','username','password']
        widgets = {
        'password': forms.PasswordInput()
        }

class TeacherForm(forms.ModelForm):
    class Meta:
        model=models.Teacher
        fields=['address','mobile','profile_pic']

class AIExamGenerationForm(forms.Form):
    course = forms.ModelChoiceField(
        queryset=Course.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Course"
    )
    
    title = forms.CharField(
        max_length=200, 
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label="Exam Title"
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        label="Exam Description"
    )
    
    difficulty = forms.ChoiceField(
        choices=[
            ('easy', 'Easy'),
            ('medium', 'Medium'),
            ('hard', 'Hard')
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Difficulty Level"
    )
    
    num_questions = forms.IntegerField(
        min_value=5,
        max_value=50,
        initial=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label="Number of Questions"
    )
    
    time_limit = forms.IntegerField(
        min_value=15,
        max_value=180,
        initial=60,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label="Time Limit (minutes)"
    )
    
    reference_material = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control-file'}),
        label="Reference Material (PDF/Word)",
        help_text="Upload course materials to base questions on (optional)"
    )

