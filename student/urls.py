from django.urls import path
from . import views
from django.contrib.auth.views import LoginView

urlpatterns = [
path('studentclick', views.studentclick_view, name='studentclick'),
path('studentlogin', LoginView.as_view(template_name='student/studentlogin.html'), name='studentlogin'),
path('studentsignup', views.student_signup_view, name='studentsignup'),
path('student-dashboard', views.student_dashboard_view, name='student-dashboard'),
path('student-exam', views.student_exam_view, name='student-exam'),
path('take-exam/<int:pk>', views.take_exam_view, name='take-exam'),
path('start-exam/<int:pk>', views.start_exam_view, name='start-exam'),
path('start-sequential-exam/<int:pk>', views.start_sequential_exam_view, name='start-sequential-exam'),
path('calculate-marks', views.calculate_marks_view, name='calculate-marks'),
path('view-result', views.view_result_view, name='view-result'),
path('check-marks/<int:pk>', views.check_marks_view, name='check-marks'),
path('student-marks', views.student_marks_view, name='student-marks'),
path('question-feedback/<int:attempt_id>', views.question_feedback_view, name='question-feedback'),
path('logout', views.logout_view, name='logout'),
# AI Chat URLs
path('ai-chat', views.ai_chat_view, name='ai_chat'),
path('get-ai-response', views.get_ai_response_view, name='get_ai_response'),
path('check-ai-connection', views.check_ai_connection_view, name='check_ai_connection'),
# Analytics URL
path('analytics', views.student_analytics_view, name='student_analytics'),
]