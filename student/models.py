from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Student(models.Model):
    user=models.OneToOneField(User,on_delete=models.CASCADE)
    profile_pic= models.ImageField(upload_to='profile_pic/Student/',null=True,blank=True)
    address = models.CharField(max_length=40)
    mobile = models.CharField(max_length=20,null=False)
    ai_usage_count = models.IntegerField(default=0)
    last_ai_interaction = models.DateTimeField(null=True, blank=True)
   
    @property
    def get_name(self):
        return self.user.first_name+" "+self.user.last_name
    @property
    def get_instance(self):
        return self
    def __str__(self):
        return self.user.first_name

class AIChatHistory(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    course = models.ForeignKey('quiz.Course', on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.student.get_name} - {self.timestamp}"

class AIUsageAnalytics(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    total_queries = models.IntegerField(default=0)
    average_response_time = models.FloatField(default=0)
    topics_covered = models.JSONField(default=list)
    
    class Meta:
        unique_together = ['student', 'date']
    
    def __str__(self):
        return f"{self.student.get_name} - {self.date}"