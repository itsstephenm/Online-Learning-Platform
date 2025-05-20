from django.urls import path, include
from django.contrib import admin
from quiz import views as quiz_views
from django.contrib.auth.views import LogoutView, LoginView
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

# Simple health check function
def health_check(request):
    return HttpResponse("OK", content_type="text/plain")

urlpatterns = [
    # Health check endpoint for Render
    path('health/', health_check, name='health_check'),
    
    path('explorer/', include('explorer.urls')),
    path('admin/', admin.site.urls),
    path('teacher/', include('teacher.urls')),
    path('student/', include('student.urls')),
    
    path('', quiz_views.home_view, name=''),
    path('logout', LogoutView.as_view(template_name='quiz/logout.html'), name='logout'),
    path('aboutus', quiz_views.aboutus_view),
    path('contactus', quiz_views.contactus_view),
    path('afterlogin', quiz_views.afterlogin_view, name='afterlogin'),

    path('adminclick', quiz_views.adminclick_view),
    path('adminlogin', quiz_views.admin_login_view, name='adminlogin'),
    path('admin-dashboard', quiz_views.admin_dashboard_view, name='admin-dashboard'),
    path('admin-teacher', quiz_views.admin_teacher_view, name='admin-teacher'),
    path('admin-view-teacher', quiz_views.admin_view_teacher_view, name='admin-view-teacher'),
    path('update-teacher/<int:pk>', quiz_views.update_teacher_view, name='update-teacher'),
    path('delete-teacher/<int:pk>', quiz_views.delete_teacher_view, name='delete-teacher'),
    path('admin-view-pending-teacher', quiz_views.admin_view_pending_teacher_view, name='admin-view-pending-teacher'),
    path('admin-view-teacher-salary', quiz_views.admin_view_teacher_salary_view, name='admin-view-teacher-salary'),
    path('approve-teacher/<int:pk>', quiz_views.approve_teacher_view, name='approve-teacher'),
    path('reject-teacher/<int:pk>', quiz_views.reject_teacher_view, name='reject-teacher'),

    path('admin-student', quiz_views.admin_student_view, name='admin-student'),
    path('admin-view-student', quiz_views.admin_view_student_view, name='admin-view-student'),
    path('admin-view-student-marks', quiz_views.admin_view_student_marks_view, name='admin-view-student-marks'),
    path('admin-view-marks/<int:pk>', quiz_views.admin_view_marks_view, name='admin-view-marks'),
    path('admin-check-marks/<int:pk>', quiz_views.admin_check_marks_view, name='admin-check-marks'),
    path('update-student/<int:pk>', quiz_views.update_student_view, name='update-student'),
    path('delete-student/<int:pk>', quiz_views.delete_student_view, name='delete-student'),

    path('admin-course', quiz_views.admin_course_view, name='admin-course'),
    path('admin-add-course', quiz_views.admin_add_course_view, name='admin-add-course'),
    path('admin-view-course', quiz_views.admin_view_course_view, name='admin-view-course'),
    path('delete-course/<int:pk>', quiz_views.delete_course_view, name='delete-course'),

    path('admin-question', quiz_views.admin_question_view, name='admin-question'),
    path('admin-add-question', quiz_views.admin_add_question_view, name='admin-add-question'),
    path('admin-view-question', quiz_views.admin_view_question_view, name='admin-view-question'),
    path('view-question/<int:pk>', quiz_views.view_question_view, name='view-question'),
    path('delete-question/<int:pk>', quiz_views.delete_question_view, name='delete-question'),
    path('update-question/<int:pk>', quiz_views.update_question_view, name='update-question'),
    path('admin-generate-questions', quiz_views.admin_generate_questions_view, name='admin-generate-questions'),
    path('admin-ai-exam', quiz_views.admin_ai_exam_view, name='admin-ai-exam'),
    path('admin-review-ai-exam', quiz_views.admin_review_ai_exam_view, name='admin-review-ai-exam'),
    
    # Include quiz URLs with namespace
    path('quiz/', include('quiz.urls')),
]

# Serve media files in development
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
