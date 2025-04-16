from django.db import models
from django.db.models import JSONField
from student.models import Student
from django.contrib.auth.models import User

class Course(models.Model):
   course_name = models.CharField(max_length=50)
   question_number = models.PositiveIntegerField()
   total_marks = models.PositiveIntegerField()
   # New fields for timed quiz functionality
   is_timed = models.BooleanField(default=False, help_text="Whether this course has a time limit")
   total_time_minutes = models.PositiveIntegerField(default=0, help_text="Total time allowed for the quiz in minutes (0 = unlimited)")
   sequential_questions = models.BooleanField(default=False, help_text="Display questions one at a time sequentially")
   allow_backtracking = models.BooleanField(default=True, help_text="Whether students can go back to previous questions")
   security_level = models.CharField(max_length=50, choices=[
       ('low', 'Low - Basic protection'),
       ('medium', 'Medium - Prevent copy-paste and right-click'),
       ('high', 'High - Block dev tools and keyboard shortcuts')
   ], default='medium', help_text="Level of security measures to prevent cheating")
   
   def __str__(self):
        return self.course_name

class Question(models.Model):
    course=models.ForeignKey(Course,on_delete=models.CASCADE)
    marks=models.PositiveIntegerField()
    question=models.CharField(max_length=600)
    option1=models.CharField(max_length=200)
    option2=models.CharField(max_length=200)
    option3=models.CharField(max_length=200)
    option4=models.CharField(max_length=200)
    cat=(('Option1','Option1'),('Option2','Option2'),('Option3','Option3'),('Option4','Option4'))
    answer=models.CharField(max_length=200,choices=cat)

class Result(models.Model):
    student = models.ForeignKey(Student,on_delete=models.CASCADE)
    exam = models.ForeignKey(Course,on_delete=models.CASCADE)
    marks = models.PositiveIntegerField()
    date = models.DateTimeField(auto_now=True)

class AIGeneratedExam(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    difficulty = models.CharField(max_length=50, choices=[
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard')
    ], default='medium')
    time_limit = models.PositiveIntegerField(default=60)  # in minutes
    created_at = models.DateTimeField(auto_now_add=True)
    approved = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title

class QuestionAttempt(models.Model):
    """Track individual question attempts by students"""
    student = models.ForeignKey('student.Student', on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer_selected = models.CharField(max_length=200)
    is_correct = models.BooleanField()
    time_taken = models.PositiveIntegerField(help_text="Time taken in seconds")
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.student} - {self.question} - {'Correct' if self.is_correct else 'Incorrect'}"

class ResultAnalytics(models.Model):
    """Extended analytics data for quiz results"""
    result = models.OneToOneField(Result, on_delete=models.CASCADE, related_name='analytics')
    total_time = models.PositiveIntegerField(help_text="Total time taken in seconds")
    average_time_per_question = models.FloatField()
    correct_answers = models.PositiveIntegerField()
    incorrect_answers = models.PositiveIntegerField()
    
    def __str__(self):
        return f"Analytics for {self.result}"

class QuestionExplanation(models.Model):
    """AI-generated explanations for quiz questions"""
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='explanations')
    explanation_text = models.TextField(help_text="Detailed explanation of the question and correct answer")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Explanation for {self.question}"

class StudentFeedback(models.Model):
    """Track feedback given to students after question attempts"""
    question_attempt = models.OneToOneField(QuestionAttempt, on_delete=models.CASCADE, related_name='feedback')
    feedback_text = models.TextField(help_text="AI-generated feedback based on the student's answer")
    is_correct = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback for {self.question_attempt}"

class AdaptiveQuizSettings(models.Model):
    """Settings for adaptive quizzing behavior by course"""
    course = models.OneToOneField(Course, on_delete=models.CASCADE, related_name='adaptive_settings')
    is_adaptive = models.BooleanField(default=False, help_text="Whether this course uses adaptive quizzing")
    min_difficulty = models.PositiveIntegerField(default=1, help_text="Minimum difficulty level (1-10)")
    max_difficulty = models.PositiveIntegerField(default=10, help_text="Maximum difficulty level (1-10)")
    difficulty_step = models.FloatField(default=0.5, help_text="How much to adjust difficulty after each answer")
    
    def __str__(self):
        return f"Adaptive settings for {self.course}"

class StudentSkillLevel(models.Model):
    """Track student skill levels by course for adaptive quizzing"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='skill_levels')
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    current_level = models.FloatField(default=5.0, help_text="Current difficulty level (1-10)")
    confidence = models.FloatField(default=0.5, help_text="System confidence in the current level (0-1)")
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['student', 'course']
    
    def __str__(self):
        return f"{self.student}'s level in {self.course}: {self.current_level}"

class ContentRecommendation(models.Model):
    """Content recommendations for students"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='recommendations')
    course = models.ForeignKey(Course, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    resource_type = models.CharField(max_length=50, choices=[
        ('course', 'Course'),
        ('topic', 'Topic'),
        ('article', 'Article'),
        ('video', 'Video'),
        ('exercise', 'Exercise'),
        ('book', 'Book'),
        ('other', 'Other')
    ])
    url = models.URLField(null=True, blank=True, help_text="Optional URL to external resource")
    relevance_score = models.FloatField(default=0.0, help_text="Relevance score (0-1)")
    is_viewed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Recommendation for {self.student}: {self.title}"

class ReferenceDocument(models.Model):
    """Reference documents uploaded for AI exam generation"""
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='reference_documents')
    title = models.CharField(max_length=255)
    document_file = models.FileField(upload_to='reference_documents/')
    extracted_text = models.TextField(blank=True, null=True)
    file_type = models.CharField(max_length=10, choices=[
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('txt', 'Text File'),
    ])
    uploaded_by = models.ForeignKey('teacher.Teacher', on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.title} ({self.course})"

# AI Adoption Prediction Models
class AIAdoptionData(models.Model):
    """Model to store AI adoption data from students"""
    STUDY_LEVEL_CHOICES = [
        ('undergraduate', 'Undergraduate'),
        ('postgraduate', 'Postgraduate'),
        ('doctorate', 'Doctorate'),
        ('other', 'Other')
    ]
    
    FACULTY_CHOICES = [
        ('engineering', 'Engineering'),
        ('science', 'Science'),
        ('arts', 'Arts'),
        ('business', 'Business'),
        ('law', 'Law'),
        ('medicine', 'Medicine'),
        ('other', 'Other')
    ]
    
    FREQUENCY_CHOICES = [
        ('never', 'Never'),
        ('rarely', 'Rarely'),
        ('sometimes', 'Sometimes'),
        ('often', 'Often'),
        ('daily', 'Daily')
    ]
    
    ADOPTION_LEVEL_CHOICES = [
        ('non_adopter', 'Non-Adopter'),
        ('potential_adopter', 'Potential Adopter'),
        ('early_adopter', 'Early Adopter'),
        ('regular_adopter', 'Regular Adopter'),
        ('advanced_adopter', 'Advanced Adopter')
    ]
    
    YES_NO_CHOICES = [
        ('yes', 'Yes'),
        ('no', 'No'),
        ('maybe', 'Maybe')
    ]
    
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_adoption_data', null=True, blank=True)
    
    # Basic information
    level_of_study = models.CharField(max_length=20, choices=STUDY_LEVEL_CHOICES)
    faculty = models.CharField(max_length=20, choices=FACULTY_CHOICES)
    
    # AI familiarity and usage
    ai_familiarity = models.IntegerField(default=0, help_text="Scale of 1-5")
    uses_ai_tools = models.CharField(max_length=10, choices=YES_NO_CHOICES, default='no')
    tools_used = models.TextField(blank=True, null=True, help_text="Comma-separated list of tools")
    usage_frequency = models.CharField(max_length=20, choices=FREQUENCY_CHOICES, default='never')
    
    # Feedback
    challenges = models.TextField(blank=True, null=True)
    suggestions = models.TextField(blank=True, null=True)
    improves_learning = models.CharField(max_length=10, choices=YES_NO_CHOICES, default='no')
    
    # Derived fields
    tools_count = models.IntegerField(default=0)
    challenges_count = models.IntegerField(default=0)
    
    # Target variable - adoption level
    adoption_level = models.CharField(max_length=20, choices=ADOPTION_LEVEL_CHOICES)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    source = models.CharField(max_length=50, default='manual', help_text="Source of the data: csv, form, etc.")
    
    def __str__(self):
        student_name = self.student.username if self.student else "Anonymous"
        return f"{student_name} - {self.get_adoption_level_display()}"
    
    class Meta:
        verbose_name = "AI Adoption Data"
        verbose_name_plural = "AI Adoption Data"
        ordering = ['-created_at']

class AIPrediction(models.Model):
    """Model to store predictions made by the AI model"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_predictions')
    
    # Input data (only storing what's needed for later reference)
    level_of_study = models.CharField(max_length=20)
    faculty = models.CharField(max_length=20)
    ai_familiarity = models.IntegerField()
    uses_ai_tools = models.CharField(max_length=10)
    tools_count = models.IntegerField()
    challenges_count = models.IntegerField()
    
    # Prediction results
    predicted_adoption_level = models.CharField(max_length=20)
    confidence = models.FloatField()
    features_used = models.JSONField(default=dict)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.user.username} - {self.predicted_adoption_level} ({self.confidence:.2f})"
    
    class Meta:
        verbose_name = "AI Prediction"
        verbose_name_plural = "AI Predictions"
        ordering = ['-created_at']

class InsightTopic(models.Model):
    """Model to store insight topics"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Insight Topic"
        verbose_name_plural = "Insight Topics"

class AIInsight(models.Model):
    """Model to store AI-generated insights from the data"""
    topic = models.ForeignKey(InsightTopic, on_delete=models.CASCADE, related_name='insights')
    title = models.CharField(max_length=200)
    content = models.TextField()
    source_data = models.JSONField(default=dict, help_text="Source data for the insight")
    chart_data = models.JSONField(null=True, blank=True, help_text="Data for chart rendering")
    chart_type = models.CharField(max_length=50, null=True, blank=True, help_text="Type of chart: bar, pie, etc.")
    relevance_score = models.FloatField(default=0.0, help_text="How relevant this insight is")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        verbose_name = "AI Insight"
        verbose_name_plural = "AI Insights"
        ordering = ['-relevance_score', '-created_at']

class NLQuery(models.Model):
    """Model to store natural language queries and their responses"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='nl_queries')
    query = models.TextField()
    processed_query = models.TextField(help_text="Query after processing")
    response = models.TextField()
    response_type = models.CharField(max_length=50, default='text', help_text="Type of response: text, chart, etc.")
    chart_data = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}: {self.query[:50]}"
    
    class Meta:
        verbose_name = "NL Query"
        verbose_name_plural = "NL Queries"
        ordering = ['-created_at']

