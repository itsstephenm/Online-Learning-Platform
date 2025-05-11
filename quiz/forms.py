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
        fields=['course_name', 'title', 'question_number','total_marks',
                'is_timed','total_time_minutes','sequential_questions',
                'allow_backtracking','security_level']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'id': 'title'}),
            'is_timed': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'is_timed'}),
            'sequential_questions': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'sequential_questions'}),
            'allow_backtracking': forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'allow_backtracking'}),
            'total_time_minutes': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'id': 'total_time_minutes'}),
            'security_level': forms.Select(attrs={'class': 'form-control', 'id': 'security_level'})
        }
        labels = {
            'title': 'Course Title',
            'is_timed': 'Enable Time Limit',
            'total_time_minutes': 'Total Time (minutes)',
            'sequential_questions': 'Display Questions One at a Time',
            'allow_backtracking': 'Allow Going Back to Previous Questions',
            'security_level': 'Exam Security Level'
        }

class QuestionForm(forms.ModelForm):
    #this will show dropdown __str__ method course model is shown on html so override it
    #to_field_name this will fetch corresponding value user_id present in course model and return it
    courseID = forms.ModelChoiceField(queryset=models.Course.objects.all(), empty_label="Course Name", to_field_name="id")
    
    class Meta:
        model = models.Question
        fields = ['marks', 'question', 'question_type', 'option1', 'option2', 'option3', 'option4', 
                  'answer', 'multiple_answers', 'short_answer_pattern']
        widgets = {
            'question': forms.Textarea(attrs={'rows': 3, 'cols': 50}),
            'short_answer_pattern': forms.Textarea(attrs={'rows': 2, 'cols': 50, 'placeholder': 'Enter keywords or patterns to match correct answers, separated by commas'}),
            'multiple_answers': forms.CheckboxSelectMultiple(choices=[
                ('Option1', 'Option 1'),
                ('Option2', 'Option 2'),
                ('Option3', 'Option 3'),
                ('Option4', 'Option 4'),
            ]),
            'question_type': forms.Select(attrs={'class': 'form-select', 'id': 'question-type-select'})
        }

    def clean(self):
        cleaned_data = super().clean()
        question_type = cleaned_data.get('question_type')
        
        if question_type == 'multiple_choice':
            # Validate required fields for multiple choice
            if not cleaned_data.get('answer'):
                self.add_error('answer', 'Answer is required for multiple choice questions')
                
        elif question_type == 'checkbox':
            # Validate required fields for checkbox questions
            multiple_answers = cleaned_data.get('multiple_answers')
            if not multiple_answers:
                self.add_error('multiple_answers', 'At least one correct answer is required for checkbox questions')
                
        elif question_type == 'short_answer':
            # Validate required fields for short answer questions
            if not cleaned_data.get('short_answer_pattern'):
                self.add_error('short_answer_pattern', 'Answer pattern is required for short answer questions')
                
        return cleaned_data

class AIExamGenerationForm(forms.Form):
    course = forms.ModelChoiceField(
        queryset=models.Course.objects.all(),
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
