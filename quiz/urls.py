from django.urls import path
from . import views

app_name = 'quiz'

urlpatterns = [
    # Add basic URL patterns for quiz app
    path('', views.home_view, name='quiz-home'),
] 