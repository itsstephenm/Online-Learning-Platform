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

    def get_avg_score(self):
        """Get average score across all exams and quizzes"""
        from quiz.models import Result
        results = Result.objects.filter(student=self)
        if not results.exists():
            return 0
        total_score = sum(result.marks for result in results)
        return total_score / results.count()

    def get_attendance_rate(self):
        """Get student attendance rate (placeholder)"""
        # This would be implemented based on your attendance tracking system
        # For now, return a default value
        return 85.0  # 85% attendance rate

    def get_quiz_completion_rate(self):
        """Get the rate of quiz completion"""
        from quiz.models import Result, Course
        total_quizzes = Course.objects.count()
        if total_quizzes == 0:
            return 100.0
        completed_quizzes = Result.objects.filter(student=self).count()
        return (completed_quizzes / total_quizzes) * 100

    def get_avg_study_time(self):
        """Get average study time (placeholder)"""
        # This would be implemented based on your study time tracking system
        # For now, return a default value
        return 3.5  # 3.5 hours per day

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