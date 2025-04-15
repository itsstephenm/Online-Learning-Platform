from django import forms
from django.contrib.auth.models import User
from . import models

class ContactusForm(forms.Form):
    Name = forms.CharField(max_length=30)
    Email = forms.EmailField()
    Message = forms.CharField(max_length=500,widget=forms.Textarea(attrs={'rows': 3, 'cols': 30}))

class TeacherSalaryForm(forms.Form):
    salary=forms.IntegerField()

class CourseForm(forms.ModelForm):
    class Meta:
        model=models.Course
        fields=['course_name','question_number','total_marks',
                'is_timed','total_time_minutes','sequential_questions',
                'allow_backtracking','security_level']
        widgets = {
            'is_timed': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'is_timed'}),
            'sequential_questions': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'sequential_questions'}),
            'allow_backtracking': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'allow_backtracking'}),
            'total_time_minutes': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'id': 'total_time_minutes'}),
            'security_level': forms.Select(attrs={'class': 'form-control', 'id': 'security_level'})
        }
        labels = {
            'is_timed': 'Enable Time Limit',
            'total_time_minutes': 'Total Time (minutes)',
            'sequential_questions': 'Display Questions One at a Time',
            'allow_backtracking': 'Allow Going Back to Previous Questions',
            'security_level': 'Exam Security Level'
        }

class QuestionForm(forms.ModelForm):
    
    #this will show dropdown __str__ method course model is shown on html so override it
    #to_field_name this will fetch corresponding value  user_id present in course model and return it
    courseID=forms.ModelChoiceField(queryset=models.Course.objects.all(),empty_label="Course Name", to_field_name="id")
    class Meta:
        model=models.Question
        fields=['marks','question','option1','option2','option3','option4','answer']
        widgets = {
            'question': forms.Textarea(attrs={'rows': 3, 'cols': 50})
        }
