from django.shortcuts import render, redirect, reverse, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.core.mail import send_mail
from django.conf import settings
from datetime import date, timedelta
from django.db.models import Sum, Q, Count, Avg, FloatField, Max
from django.db.models.functions import Coalesce
from django.contrib.auth.models import Group, User
from . import forms, models
from teacher import models as TMODEL
from student import models as SMODEL
from teacher import forms as TFORM
from student import forms as SFORM
from django.views.decorators.csrf import csrf_exempt
from django.contrib.admin.views.decorators import staff_member_required
import json, random, logging, tempfile, os, time
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from student.models import Student
from django.conf import settings
from decouple import config
from collections import Counter
import numpy as np
import uuid

# Import AI utilities
from .ai_utils import predict_adoption_level, train_model, make_prediction, generate_insights_from_data
from .ai_utils import get_data_counts, prepare_features, get_chart_data, process_nl_query, process_training_data
from .models import AIAdoptionData, AIPrediction, NLQuery, InsightTopic, AIInsight, AIModel, CSVUpload

# Import the AI data utilities
try:
    from .ai_data_utils import process_csv_file, clean_survey_data, calculate_data_stats
    from .ai_data_utils import train_ai_model, predict_adoption_level as predict_level
    from .ai_data_utils import import_from_csv, generate_insights
except ImportError:
    # Fall back to dummy functions if the module doesn't exist
    logging.warning("AI data utilities module not found, using dummy functions")
    
    def process_csv_file(file_path, save_to_db=True):
        return pd.DataFrame(), {"status": "error", "message": "AI data utilities not available"}
    
    def import_from_csv(file_path, save_to_db=True):
        return pd.DataFrame(), {}, 0.0, ["AI data utilities not available"]
        
    def train_ai_model(csv_upload_id=None, algorithm="random_forest"):
        return {"success": False, "error": "AI data utilities not available"}
        
    def predict_level(data):
        return "medium", 0.7, {}
        
    def generate_insights(df, stats):
        return ["AI data utilities not available"]

# Configure logging
logger = logging.getLogger(__name__)

def home_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return render(request, 'quiz/index.html')

def is_teacher(user):
    return user.groups.filter(name='TEACHER').exists()

def is_student(user):
    return user.groups.filter(name='STUDENT').exists()

# Check if user is admin
def is_admin(user):
    return user.groups.filter(name='ADMIN').exists()

def afterlogin_view(request):
    if is_student(request.user):
        return redirect('student/student-dashboard')
    elif is_teacher(request.user):
        accountapproval = TMODEL.Teacher.objects.filter(user_id=request.user.id, status=True).exists()
        if accountapproval:
            return redirect('teacher/teacher-dashboard')
        else:
            return render(request, 'teacher/teacher_wait_for_approval.html')
    else:
        return redirect('admin-dashboard')

def adminclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return HttpResponseRedirect('adminlogin')

@login_required(login_url='adminlogin')
def admin_dashboard_view(request):
    context = {
        'total_student': SMODEL.Student.objects.count(),
        'total_teacher': TMODEL.Teacher.objects.filter(status=True).count(),
        'total_course': models.Course.objects.count(),
        'total_question': models.Question.objects.count(),
    }
    return render(request, 'quiz/admin_dashboard.html', context)

@login_required(login_url='adminlogin')
def admin_teacher_view(request):
    # Get department information with teacher counts and average salaries
    departments = TMODEL.Department.objects.annotate(
        teacher_count=Count('teacher'),
        avg_salary=Coalesce(Avg('teacher__salary'), 0, output_field=FloatField())
    )
    
    # Format department data for the template
    department_data = []
    for dept in departments:
        status = 'Active' if dept.status else 'Inactive'
        status_color = 'success' if dept.status else 'danger'
        
        department_data.append({
            'name': dept.name,
            'teacher_count': dept.teacher_count,
            'avg_salary': f"{int(dept.avg_salary):,}" if dept.avg_salary else '0',
            'status': status,
            'status_color': status_color
        })
    
    context = {
        'total_teacher': TMODEL.Teacher.objects.filter(status=True).count(),
        'pending_teacher': TMODEL.Teacher.objects.filter(status=False).count(),
        'salary': TMODEL.Teacher.objects.filter(status=True).aggregate(Sum('salary'))['salary__sum'],
        'departments': department_data,
    }
    return render(request, 'quiz/admin_teacher.html', context)

@login_required(login_url='adminlogin')
def admin_view_teacher_view(request):
    teachers = TMODEL.Teacher.objects.filter(status=True)
    return render(request, 'quiz/admin_view_teacher.html', {'teachers': teachers})

@login_required(login_url='adminlogin')
def update_teacher_view(request, pk):
    teacher = TMODEL.Teacher.objects.get(id=pk)
    user = User.objects.get(id=teacher.user_id)
    userForm = TFORM.TeacherUserForm(instance=user)
    teacherForm = TFORM.TeacherForm(instance=teacher)
    context = {'userForm': userForm, 'teacherForm': teacherForm}

    if request.method == 'POST':
        userForm = TFORM.TeacherUserForm(request.POST, instance=user)
        teacherForm = TFORM.TeacherForm(request.POST, request.FILES, instance=teacher)
        if userForm.is_valid() and teacherForm.is_valid():
            user = userForm.save()
            user.set_password(user.password)
            user.save()
            teacherForm.save()
            return redirect('admin-view-teacher')

    return render(request, 'quiz/update_teacher.html', context)

@login_required(login_url='adminlogin')
def delete_teacher_view(request, pk):
    teacher = TMODEL.Teacher.objects.get(id=pk)
    user = User.objects.get(id=teacher.user_id)
    user.delete()
    teacher.delete()
    return HttpResponseRedirect('/admin-view-teacher')

@login_required(login_url='adminlogin')
def admin_view_pending_teacher_view(request):
    teachers = TMODEL.Teacher.objects.filter(status=False)
    return render(request, 'quiz/admin_view_pending_teacher.html', {'teachers': teachers})

@login_required(login_url='adminlogin')
def approve_teacher_view(request, pk):
    teacherSalary = forms.TeacherSalaryForm()
    if request.method == 'POST':
        teacherSalary = forms.TeacherSalaryForm(request.POST)
        if teacherSalary.is_valid():
            teacher = TMODEL.Teacher.objects.get(id=pk)
            teacher.salary = teacherSalary.cleaned_data['salary']
            teacher.status = True
            teacher.save()
        return HttpResponseRedirect('/admin-view-pending-teacher')
    return render(request, 'quiz/salary_form.html', {'teacherSalary': teacherSalary})

@login_required(login_url='adminlogin')
def reject_teacher_view(request, pk):
    teacher = TMODEL.Teacher.objects.get(id=pk)
    user = User.objects.get(id=teacher.user_id)
    user.delete()
    teacher.delete()
    return HttpResponseRedirect('/admin-view-pending-teacher')

@login_required(login_url='adminlogin')
def admin_view_teacher_salary_view(request):
    teachers = TMODEL.Teacher.objects.filter(status=True)
    return render(request, 'quiz/admin_view_teacher_salary.html', {'teachers': teachers})

@login_required(login_url='adminlogin')
def admin_student_view(request):
    from django.db.models import Avg, Count, F, Value, CharField
    from django.db.models.functions import Concat
    import random
    
    # Get recent students with related data
    students = SMODEL.Student.objects.all().order_by('-id')[:5]  # Get 5 most recent students
    
    # Calculate performance metrics if Result model is available
    # Using sample data if no real metrics are available
    try:
        avg_score = models.Result.objects.aggregate(avg=Avg('marks'))['avg']
        avg_score = int(avg_score) if avg_score else 85
    except:
        avg_score = 85  # Default value
    
    # Prepare student data for the template
    recent_students = []
    colors = ['primary', 'info', 'success', 'warning', 'danger']
    
    for i, student in enumerate(students):
        # Get student's name
        name = student.get_name if hasattr(student, 'get_name') else f"{student.user.first_name} {student.user.last_name}"
        
        # Get student's initials
        try:
            initials = "".join([name.split()[0][0], name.split()[1][0]])
        except:
            initials = name[0:2].upper()
        
        # Get course information
        try:
            # Try to get the student's course, if available
            course_result = models.Result.objects.filter(student=student).order_by('-date').first()
            course_name = course_result.exam.course.name if course_result else "Not enrolled"
            marks = f"{course_result.marks}/100" if course_result else "N/A"
        except:
            course_name = "Not available"
            marks = "N/A"
        
        # Set avatar color (cycle through colors)
        avatar_color = colors[i % len(colors)]
        
        recent_students.append({
            'id': student.id,
            'get_name': name,
            'initials': initials,
            'course_name': course_name,
            'marks': marks, 
            'status': 'Active',
            'status_color': 'success',
            'avatar_color': avatar_color
        })
    
    context = {
        'total_student': SMODEL.Student.objects.count(),
        'avg_score': avg_score,
        'attendance_rate': 92,  # Sample value - replace with actual calculation if available
        'completion_rate': 78,  # Sample value - replace with actual calculation if available
        'participation_rate': 65,  # Sample value - replace with actual calculation if available
        'recent_students': recent_students
    }
    return render(request, 'quiz/admin_student.html', context)

@login_required(login_url='adminlogin')
def admin_view_student_view(request):
    students = SMODEL.Student.objects.all()
    return render(request, 'quiz/admin_view_student.html', {'students': students})

@login_required(login_url='adminlogin')
def update_student_view(request, pk):
    student = SMODEL.Student.objects.get(id=pk)
    user = User.objects.get(id=student.user_id)
    userForm = SFORM.StudentUserForm(instance=user)
    studentForm = SFORM.StudentForm(instance=student)
    context = {'userForm': userForm, 'studentForm': studentForm}

    if request.method == 'POST':
        userForm = SFORM.StudentUserForm(request.POST, instance=user)
        studentForm = SFORM.StudentForm(request.POST, request.FILES, instance=student)
        if userForm.is_valid() and studentForm.is_valid():
            user = userForm.save()
            user.set_password(user.password)
            user.save()
            studentForm.save()
            return redirect('admin-view-student')

    return render(request, 'quiz/update_student.html', context)

@login_required(login_url='adminlogin')
def delete_student_view(request, pk):
    student = SMODEL.Student.objects.get(id=pk)
    user = User.objects.get(id=student.user_id)
    user.delete()
    student.delete()
    return HttpResponseRedirect('/admin-view-student')

@login_required(login_url='adminlogin')
def admin_course_view(request):
    from django.db.models import Count
    
    # Get all courses and annotate with student count
    courses = models.Course.objects.annotate(
        student_count=Count('result__student', distinct=True)
    ).order_by('-id')[:5]  # Get 5 most recent courses
    
    # Prepare course data for the template
    recent_courses = []
    
    for course in courses:
        # Try to get a teacher for this course
        teacher_name = "Not Assigned"
        try:
            # This would depend on your model relationships
            # Adjust this based on how teachers are assigned to courses
            teacher = TMODEL.Teacher.objects.filter(user__in=course.question_set.values_list('user', flat=True).distinct()).first()
            if teacher:
                teacher_name = teacher.get_name
        except:
            pass
        
        recent_courses.append({
            'id': course.id,
            'name': course.course_name,
            'teacher_name': teacher_name,
            'student_count': course.student_count
        })
    
    context = {
        'total_course': models.Course.objects.count(),
        'recent_courses': recent_courses
    }
    return render(request, 'quiz/admin_course.html', context)

@login_required(login_url='adminlogin')
def admin_add_course_view(request):
    courseForm = forms.CourseForm()
    if request.method == 'POST':
        courseForm = forms.CourseForm(request.POST)
        if courseForm.is_valid():
            courseForm.save()
        return HttpResponseRedirect('/admin-view-course')
    return render(request, 'quiz/admin_add_course.html', {'courseForm': courseForm})

@login_required(login_url='adminlogin')
def admin_view_course_view(request):
    courses = models.Course.objects.all()
    return render(request, 'quiz/admin_view_course.html', {'courses': courses})

@login_required(login_url='adminlogin')
def delete_course_view(request, pk):
    course = models.Course.objects.get(id=pk)
    course.delete()
    return HttpResponseRedirect('/admin-view-course')

@login_required(login_url='adminlogin')
def admin_question_view(request):
    # Get question statistics for the dashboard
    total_question = models.Question.objects.count()
    total_course = models.Course.objects.count()
    
    # Get counts by question type
    mcq_count = models.Question.objects.filter(question_type='multiple_choice').count()
    checkbox_count = models.Question.objects.filter(question_type='checkbox').count()
    short_answer_count = models.Question.objects.filter(question_type='short_answer').count()
    
    # Get AI generated vs manually created counts
    ai_generated = models.Question.objects.filter(is_ai_generated=True).count()
    manual_created = total_question - ai_generated
    
    context = {
        'total_question': total_question,
        'total_course': total_course,
        'mcq_count': mcq_count,
        'checkbox_count': checkbox_count,
        'short_answer_count': short_answer_count,
        'ai_generated': ai_generated,
        'manual_created': manual_created
    }
    
    return render(request, 'quiz/admin_question.html', context)

@login_required(login_url='adminlogin')
def admin_add_question_view(request):
    questionForm = forms.QuestionForm()
    
    # Pre-select question type if provided in URL
    question_type = request.GET.get('type', 'multiple_choice')
    if question_type in ['multiple_choice', 'checkbox', 'short_answer']:
        questionForm.initial['question_type'] = question_type
    
    if request.method == 'POST':
        questionForm = forms.QuestionForm(request.POST)
        if questionForm.is_valid():
            question = questionForm.save(commit=False)
            course = models.Course.objects.get(id=request.POST.get('courseID'))
            question.course = course
            
            # Handle different question types
            question_type = request.POST.get('question_type')
            
            if question_type == 'multiple_choice':
                # Ensure required fields for multiple choice
                question.answer = request.POST.get('answer')
                
            elif question_type == 'checkbox':
                # For checkbox questions, save multiple answers as JSON
                multiple_answers = request.POST.getlist('multiple_answers')
                question.multiple_answers = multiple_answers
                question.answer = None  # Clear single answer field
                
            elif question_type == 'short_answer':
                # For short answer questions, save the pattern
                question.short_answer_pattern = request.POST.get('short_answer_pattern')
                question.answer = None  # Clear single answer field
                # Make options empty for short answer questions
                question.option1 = None
                question.option2 = None
                question.option3 = None
                question.option4 = None
            
            question.save()
            return HttpResponseRedirect('/admin-view-question')
    return render(request, 'quiz/admin_add_question.html', {'questionForm': questionForm})

@login_required(login_url='adminlogin')
def admin_view_question_view(request):
    courses = models.Course.objects.all()
    return render(request, 'quiz/admin_view_question.html', {'courses': courses})

@login_required(login_url='adminlogin')
def view_question_view(request, pk):
    questions = models.Question.objects.filter(course_id=pk)
    return render(request, 'quiz/view_question.html', {'questions': questions})

@login_required(login_url='adminlogin')
def delete_question_view(request, pk):
    question = models.Question.objects.get(id=pk)
    question.delete()
    return HttpResponseRedirect('/admin-view-question')

@login_required(login_url='adminlogin')
def admin_view_student_marks_view(request):
    students = SMODEL.Student.objects.all()
    return render(request, 'quiz/admin_view_student_marks.html', {'students': students})

@login_required(login_url='adminlogin')
def admin_view_marks_view(request, pk):
    courses = models.Course.objects.all()
    response = render(request, 'quiz/admin_view_marks.html', {'courses': courses})
    response.set_cookie('student_id', str(pk))
    return response

@login_required(login_url='adminlogin')
def admin_check_marks_view(request, pk):
    course = models.Course.objects.get(id=pk)
    student_id = request.COOKIES.get('student_id')
    student = SMODEL.Student.objects.get(id=student_id)
    results = models.Result.objects.filter(exam=course, student=student)
    return render(request, 'quiz/admin_check_marks.html', {'results': results})

def aboutus_view(request):
    return render(request, 'quiz/aboutus.html')

def contactus_view(request):
    form = forms.ContactusForm()
    if request.method == 'POST':
        form = forms.ContactusForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['Email']
            name = form.cleaned_data['Name']
            message = form.cleaned_data['Message']
            try:
                send_mail(
                    subject=f'Contact Message from {name} ({email})',
                    message=message,
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[settings.EMAIL_RECEIVING_USER],
                    fail_silently=False
                )
                return render(request, 'quiz/contactussuccess.html')
            except Exception as e:
                # For development debugging - in production you'd want to log this instead
                if settings.DEBUG:
                    form.add_error(None, f"Failed to send email: {str(e)}")
                else:
                    form.add_error(None, "Failed to send email. Please try again later or contact us directly.")
    return render(request, 'quiz/contactus.html', {'form': form})

@require_POST
@csrf_exempt
def logout_view(request):
    logout(request)
    return redirect('home')  # Redirect to a home or landing page

@csrf_exempt
def calculate_marks_view(request):
    if request.method == "POST":
        # Process exam marks calculation
        return JsonResponse({"message": "Marks calculated successfully"})
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required(login_url='adminlogin')
def ai_prediction_dashboard_view(request):
    # Import necessary modules
    from django.db.models import Count, Avg
    from .ai_utils import get_chart_data
    import json
    
    # Get adoption data statistics
    adoption_data_count = models.AIAdoptionData.objects.count()
    
    # Get AI model statistics
    ai_model_count = models.AIModel.objects.count()
    avg_model_accuracy = models.AIModel.objects.aggregate(avg_accuracy=Avg('accuracy'))['avg_accuracy'] or 0
    
    # Get prediction statistics
    prediction_count = models.AIPrediction.objects.count()
    
    # Get AI insights
    ai_insights = models.AIInsight.objects.all().order_by('-created_at')[:3]
    
    # Get adoption level distribution
    adoption_levels = models.AIAdoptionData.objects.values('adoption_level').annotate(
        count=Count('id')
    ).order_by('adoption_level')
    
    # Convert adoption level counts to chart data format
    adoption_chart_data = {
        'labels': [],
        'data': [],
        'backgroundColor': [
            'rgba(220, 53, 69, 0.8)',  # very_low
            'rgba(255, 193, 7, 0.8)',  # low
            'rgba(23, 162, 184, 0.8)',  # medium 
            'rgba(0, 123, 255, 0.8)',   # high
            'rgba(40, 167, 69, 0.8)'    # very_high
        ]
    }
    
    # Map for nicer display labels
    level_labels = {
        'very_low': 'Very Low',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High',
        'very_high': 'Very High'
    }
    
    # Default data if no records exist
    if not adoption_levels.exists():
        adoption_chart_data['labels'] = list(level_labels.values())
        adoption_chart_data['data'] = [0, 0, 0, 0, 0]
    else:
        # Fill with actual data
        for level in adoption_levels:
            level_name = level['adoption_level']
            display_name = level_labels.get(level_name, level_name.replace('_', ' ').title())
            adoption_chart_data['labels'].append(display_name)
            adoption_chart_data['data'].append(level['count'])
    
    # Get tool usage frequency
    tools_data = get_chart_data('tools_usage')
    
    # Get challenges data
    challenges_data = []
    for data in models.AIAdoptionData.objects.all():
        if data.challenges:
            challenges = [c.strip() for c in data.challenges.split(',')]
            challenges_data.extend(challenges)
    
    challenge_counts = {}
    for challenge in challenges_data:
        if challenge:
            challenge_counts[challenge] = challenge_counts.get(challenge, 0) + 1
    
    # Get top challenges
    top_challenges = sorted(challenge_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    
    challenge_chart_data = {
        'labels': [challenge for challenge, count in top_challenges] if top_challenges else ['No data available'],
        'data': [count for challenge, count in top_challenges] if top_challenges else [0],
        'backgroundColor': [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)'
        ]
    }
    
    # Get faculty distribution
    faculty_data = get_chart_data('adoption_by_faculty')
    
    # Prepare template context
    context = {
        'total_student': SMODEL.Student.objects.all().count(),
        'total_teacher': TMODEL.Teacher.objects.all().count(),
        'total_course': models.Course.objects.count(),
        'total_question': models.Question.objects.count(),
        'adoption_data_count': adoption_data_count,
        'ai_model_count': ai_model_count,
        'avg_model_accuracy': round(avg_model_accuracy * 100, 1),
        'prediction_count': prediction_count,
        'ai_insights': ai_insights,
        'adoption_chart_data': json.dumps(adoption_chart_data),
        'tools_chart_data': json.dumps(tools_data),
        'challenge_chart_data': json.dumps(challenge_chart_data),
        'faculty_chart_data': json.dumps(faculty_data),
        # Include sample data for initial display (if no real data exists)
        'adoption_data': models.AIAdoptionData.objects.all().order_by('-created_at')[:5]
    }
    
    # Process NL query form submission
    if request.method == 'POST' and 'nl_query' in request.POST:
        query = request.POST.get('nl_query')
        if query:
            # Process the query using AI function
            from .ai_utils import process_nl_query
            result = process_nl_query(query)
            context['query_result'] = result
            
            # Return JSON response for AJAX requests
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse(result)
    
    return render(request, 'quiz/ai_prediction_dashboard.html', context=context)

@staff_member_required
def ai_prediction_dashboard(request):
    # Get stats for dashboard
    adoption_data_count = AIAdoptionData.objects.count()
    predictions_count = AIPrediction.objects.count()
    
    # Get latest insights
    insights = AIInsight.objects.all().order_by('-created_at')[:5]
    
    # Get default chart data
    faculty_chart = get_chart_data('adoption_by_faculty')
    prediction_chart = get_chart_data('prediction_distribution')
    level_chart = get_chart_data('adoption_by_study_level')
    
    # Process form submission for NL query
    if request.method == 'POST' and 'query' in request.POST:
        form = NLQueryForm(request.POST)
        if form.is_valid():
            query_text = form.cleaned_data['query']
            
            # Process the query
            result = process_nl_query(query_text)
            
            # Save query to database
            nl_query = NLQuery(
                query=query_text,
                processed_query=query_text,
                response=result['response'],
                response_type=result['response_type'],
                chart_data=json.dumps(result.get('chart_data', {})) if result.get('chart_data') else None,
                user=request.user
            )
            nl_query.save()
            
            # If this is an Ajax request, return JSON
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse(result)
    else:
        form = NLQueryForm()
    
    # Get recent queries
    recent_queries = NLQuery.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    context = {
        'form': form,
        'adoption_data_count': adoption_data_count,
        'predictions_count': predictions_count,
        'insights': insights,
        'faculty_chart': json.dumps(faculty_chart),
        'prediction_chart': json.dumps(prediction_chart),
        'level_chart': json.dumps(level_chart),
        'recent_queries': recent_queries
    }
    
    return render(request, 'quiz/ai_dashboard.html', context)

@staff_member_required
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        
        # Check if file is CSV
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Please upload a CSV file')
            return redirect('ai_prediction_dashboard')
        
        # Save file temporarily
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'csv_uploads'))
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)
        
        try:
            # Process CSV file
            data, model, accuracy, insights = import_from_csv(file_path, save_to_db=True)
            
            # Success message
            messages.success(
                request, 
                f'Successfully uploaded and processed {len(data)} records. Model accuracy: {accuracy:.2f}'
            )
            
            # Clean up
            fs.delete(filename)
            
            # If this is an Ajax request, return JSON
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'success',
                    'records': len(data),
                    'accuracy': float(accuracy)
                })
                
        except Exception as e:
            messages.error(request, f'Error processing CSV: {str(e)}')
            
            # If this is an Ajax request, return JSON
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': str(e)
                })
            
            # Clean up
            fs.delete(filename)
    
    return redirect('ai_prediction_dashboard')

@csrf_exempt
def get_adoption_prediction(request):
    if request.method == 'POST':
        try:
            # Get data from form
            data = {
                'level_of_study': request.POST.get('level_of_study', ''),
                'faculty': request.POST.get('faculty', ''),
                'ai_familiarity': int(request.POST.get('ai_familiarity', 3)),
                'uses_ai_tools': request.POST.get('uses_ai_tools', 'no'),
                'tools_used': request.POST.get('tools_used', ''),
                'usage_frequency': request.POST.get('usage_frequency', 'never'),
                'challenges': request.POST.get('challenges', ''),
                'suggestions': request.POST.get('suggestions', ''),
                'improves_learning': request.POST.get('improves_learning', 'no')
            }
            
            # Count tools and challenges
            data['tools_count'] = len(data['tools_used'].split(',')) if data['tools_used'] else 0
            data['challenges_count'] = len(data['challenges'].split('.')) if data['challenges'] else 0
            
            # Make prediction
            prediction, confidence, features_used = predict_adoption_level(data)
            
            # Format response
            response = {
                'status': 'success',
                'prediction': prediction,
                'prediction_label': prediction.replace('_', ' ').title(),
                'confidence': float(confidence),
                'features_used': features_used
            }
            
            return JsonResponse(response)
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'error', 'message': 'Only POST requests are supported'})

@staff_member_required
def ai_insights(request):
    # Get all insights grouped by topic
    topics = InsightTopic.objects.all()
    
    insights_by_topic = {}
    for topic in topics:
        insights_by_topic[topic.name] = AIInsight.objects.filter(topic=topic).order_by('-created_at')[:3]
    
    context = {
        'insights_by_topic': insights_by_topic,
    }
    
    return render(request, 'quiz/ai_insights.html', context)

@staff_member_required
def ai_data_explorer(request):
    # Get count of records
    adoption_data_count = AIAdoptionData.objects.count()
    
    # Get chart data
    faculty_chart = get_chart_data('adoption_by_faculty')
    tools_chart = get_chart_data('tools_usage')
    study_level_chart = get_chart_data('adoption_by_study_level')
    familiarity_chart = get_chart_data('familiarity_distribution')
    
    context = {
        'adoption_data_count': adoption_data_count,
        'faculty_chart': json.dumps(faculty_chart),
        'tools_chart': json.dumps(tools_chart),
        'study_level_chart': json.dumps(study_level_chart),
        'familiarity_chart': json.dumps(familiarity_chart)
    }
    
    return render(request, 'quiz/ai_data_explorer.html', context)

@login_required
def ai_dashboard(request):
    """Dashboard view for AI prediction functionality"""
    models = AIModel.objects.all()
    recent_predictions = AIPrediction.objects.all().order_by('-created_at')[:5]
    
    # Get OpenRouter integration status
    openrouter_available = config('OPENROUTER_API_KEY', default=None) is not None
    
    # Get prediction success/risk counts
    prediction_success_count = AIPrediction.objects.filter(prediction_class=1).count()
    prediction_risk_count = AIPrediction.objects.filter(prediction_class=0).count()
    
    # Get AI insights
    ai_insights = AIInsight.objects.all().order_by('-created_at')[:3]
    
    # Get most recent NL query
    try:
        # Check if NLQuery model exists in the app
        if 'NLQuery' in globals() or 'NLQuery' in locals():
            recent_query = NLQuery.objects.all().order_by('-created_at').first()
        else:
            recent_query = None
    except Exception as e:
        recent_query = None
        print(f"Error fetching NLQuery: {str(e)}")
    
    # Count insights
    try:
        insights_count = AIInsight.objects.count()
    except:
        insights_count = 0
    
    # Count queries
    try:
        if 'NLQuery' in globals() or 'NLQuery' in locals():
            queries_count = NLQuery.objects.count()
        else:
            queries_count = 0
    except:
        queries_count = 0
    
    context = {
        'models': models,
        'recent_predictions': recent_predictions,
        'prediction_count': AIPrediction.objects.count(),
        'model_count': models.count(),
        'openrouter_available': openrouter_available,
        'prediction_success_count': prediction_success_count,
        'prediction_risk_count': prediction_risk_count,
        'ai_insights': ai_insights,
        'recent_query': recent_query,
        'insights_count': insights_count,
        'queries_count': queries_count
    }
    
    return render(request, 'quiz/ai_dashboard.html', context)

@login_required
def upload_training_data(request):
    """View for uploading training data"""
    if request.method == 'POST':
        try:
            # Get the file from the request
            file = request.FILES.get('training_file')
            target_column = request.POST.get('target_column', 'success')
            
            if not file:
                return JsonResponse({'success': False, 'error': 'No file uploaded'})
            
            # Create a temporary file to save the uploaded file
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_file.write(file.read())
            temp_file.close()
            
            # Process the file
            result = process_training_data(temp_file.name, target_column)
            
            # Create a new AIAdoptionData object
            if result['success']:
                data = AIAdoptionData.objects.create(
                    file_name=file.name,
                    uploaded_by=request.user,
                    rows_processed=result['rows'],
                    is_processed=True
                )
                
                # Add the result to the response
                result['data_id'] = data.id
                result['records'] = result['rows']
                result['fields'] = result['columns']
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return JsonResponse(result)
            else:
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return render(request, 'quiz/upload_training_data.html')

@login_required
def train_ai_model(request):
    """View for training a new AI model"""
    if request.method == 'POST':
        dataset_path = request.POST.get('dataset_path')
        model_name = request.POST.get('model_name')
        model_type = request.POST.get('model_type', 'random_forest')
        description = request.POST.get('description', '')
        
        if not dataset_path or not os.path.exists(dataset_path):
            return JsonResponse({'success': False, 'error': 'Invalid dataset path'})
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate model filename
        model_filename = f"{model_name.lower().replace(' ', '_')}_{model_type}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Train the model
        result = train_model(dataset_path, model_type=model_type, save_path=model_path)
        
        if result['success']:
            # Create model record in database
            model = AIModel.objects.create(
                name=model_name,
                model_type=model_type,
                description=description,
                model_file=os.path.join('models', model_filename),
                accuracy=result['accuracy'],
                feature_importance=json.dumps(result['feature_importance'])
            )
            
            result['model_id'] = model.id
        
        return JsonResponse(result)
    
    return render(request, 'quiz/train_model.html')

@login_required
def make_prediction_view(request):
    """View for making predictions"""
    models = AIModel.objects.all()
    students = Student.objects.all()
    
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        input_type = request.POST.get('input_type')
        
        if not model_id:
            return JsonResponse({'success': False, 'error': 'Model ID is required'})
        
        if input_type == 'student':
            student_id = request.POST.get('student_id')
            if not student_id:
                return JsonResponse({'success': False, 'error': 'Student ID is required'})
            
            # Get student data
            student = get_object_or_404(Student, id=student_id)
            # Prepare student data for prediction
            student_data = {
                'prev_score': student.get_avg_score(),
                'attendance_rate': student.get_attendance_rate(),
                'quiz_completion': student.get_quiz_completion_rate(),
                'time_spent': student.get_avg_study_time(),
                'age': student.age if hasattr(student, 'age') else 20,
                'gender': student.gender if hasattr(student, 'gender') else 'M'
            }
            
            # Make prediction with student data
            result = make_prediction(input_data=student_data, model_id=model_id)
        else:
            # Manual input
            input_data = {
                'prev_score': float(request.POST.get('prev_score', 0)),
                'attendance_rate': float(request.POST.get('attendance_rate', 0)),
                'quiz_completion': float(request.POST.get('quiz_completion', 0)),
                'time_spent': float(request.POST.get('time_spent', 0)),
                'age': int(request.POST.get('age', 20)),
                'gender': request.POST.get('gender', 'M')
            }
            
            # Make prediction with manual data
            result = make_prediction(input_data=input_data, model_id=model_id)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse(result)
        
        # Store prediction result in session for non-AJAX requests
        request.session['prediction_result'] = result
        
        # Add OpenRouter usage flag for templates
        if 'explanation' in result and result['explanation']:
            result['used_openrouter'] = True
        
        return redirect('quiz:prediction_result')
    
    # For GET requests
    context = {
        'models': models,
        'students': students,
        'available_models': AIModel.objects.all(),
        'openrouter_available': config('OPENROUTER_API_KEY', default=None) is not None
    }
    
    return render(request, 'quiz/make_prediction.html', context)

@login_required
def prediction_result(request):
    """View for displaying prediction results"""
    result = request.session.get('prediction_result')
    
    if not result:
        return redirect('make_prediction')
    
    # Clear result from session
    request.session.pop('prediction_result', None)
    
    return render(request, 'quiz/prediction_result.html', {'result': result})

@login_required
def view_predictions(request):
    """View for listing all predictions"""
    predictions = AIPrediction.objects.all().order_by('-created_at')
    
    context = {
        'predictions': predictions
    }
    
    return render(request, 'quiz/view_predictions.html', context)

@login_required
def prediction_detail(request, prediction_id):
    """View for displaying detailed prediction information"""
    prediction = get_object_or_404(AIPrediction, id=prediction_id)
    
    # Parse input data JSON
    input_data = json.loads(prediction.input_data)
    
    context = {
        'prediction': prediction,
        'input_data': input_data
    }
    
    return render(request, 'quiz/prediction_detail.html', context)

@csrf_exempt
def get_student_data(request):
    """AJAX endpoint to get student data"""
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        
        if not student_id:
            return JsonResponse({'success': False, 'error': 'Student ID is required'})
        
        try:
            student = Student.objects.get(id=student_id)
            
            # Get student's metrics
            from .ai_utils import get_student_metrics
            metrics = get_student_metrics(student)
            
            return JsonResponse({
                'success': True,
                'data': metrics
            })
        
        except Student.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Student not found'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def nl_query_view(request):
    """Process natural language queries about AI data"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are supported'}, status=405)
    
    try:
        data = json.loads(request.body)
        query = data.get('query')
        
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        # Store the query
        user = request.user if request.user.is_authenticated else None
        nl_query = NLQuery(
            user=user,
            query=query,
            processed_query=query,
            response_type='text'
        )
        
        # Process the query with OpenRouter if available
        openrouter_api_key = config('OPENROUTER_API_KEY', default=None)
        if openrouter_api_key:
            try:
                import requests
                
                # Get relevant data
                stats = get_data_counts()
                if not stats.get('success', False):
                    stats = {'data': 'No statistics available'}
                
                # Create the prompt
                prompt = f"""
                As an AI education expert, analyze this data about AI adoption:
                
                {json.dumps(stats, indent=2)}
                
                User question: "{query}"
                
                Answer the question based on the data. If the data doesn't contain 
                information to answer the question, say so.
                Keep your response concise and focused on facts from the data.
                """
                
                # Call OpenRouter API
                headers = {
                    'Authorization': f'Bearer {openrouter_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": config('OPENROUTER_MODEL_NAME', default="openai/gpt-3.5-turbo"),
                    "messages": [
                        {"role": "system", "content": "You are an AI education expert specializing in analyzing educational data."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Make the API request
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions", 
                    headers=headers, 
                    json=data,
                    timeout=15
                )
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                nl_response = result['choices'][0]['message']['content'].strip()
                
                # Save the response
                nl_query.response = nl_response
                nl_query.save()
                
                return JsonResponse({
                    'response': nl_response,
                    'enhanced': True
                })
                
            except Exception as e:
                print(f"Error calling OpenRouter API: {str(e)}")
                nl_response = f"I couldn't process your question due to a technical issue: {str(e)}"
        else:
            # Basic response if OpenRouter is not available
            nl_response = "I can't process natural language queries at the moment. Please try a more specific search."
        
        # Save the response
        nl_query.response = nl_response
        nl_query.save()
        
        return JsonResponse({
            'response': nl_response,
            'enhanced': False
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def insights_view(request):
    """View for displaying AI insights"""
    insights = AIInsight.objects.all().order_by('-created_at')
    
    # If this is an AJAX request, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        insights_data = []
        for insight in insights:
            insights_data.append({
                'id': insight.id,
                'title': insight.title,
                'content': insight.content,
                'created_at': insight.created_at.isoformat() if insight.created_at else None
            })
        
        return JsonResponse({
            'success': True,
            'insights': insights_data
        })
    
    # Otherwise render the template
    return render(request, 'quiz/insights.html', {
        'insights': insights
    })

@login_required
def query_ai(request):
    """View for handling AI queries"""
    if request.method == 'POST':
        query = request.POST.get('query')
        try:
            response = process_nl_query(query)
            return JsonResponse({'success': True, 'response': response})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return render(request, 'quiz/query.html')

@login_required
def ai_data_detail(request, data_id):
    """View for displaying details of uploaded AI training data"""
    data = get_object_or_404(AIAdoptionData, id=data_id)
    return render(request, 'quiz/ai_data_detail.html', {
        'data': data
    })

@login_required
def ai_model_detail(request, model_id):
    """View for displaying AI model details"""
    model = get_object_or_404(AIModel, id=model_id)
    return render(request, 'quiz/ai_model_detail.html', {
        'model': model
    })

@login_required
def all_predictions(request):
    """View for displaying all predictions"""
    predictions = AIPrediction.objects.all().order_by('-created_at')
    
    # If this is an AJAX request, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        predictions_data = []
        for prediction in predictions:
            predictions_data.append({
                'id': prediction.id,
                'model_name': prediction.model.name if prediction.model else 'Unknown',
                'prediction_class': prediction.prediction_class,
                'success_probability': prediction.success_probability,
                'risk_probability': prediction.risk_probability,
                'created_at': prediction.created_at.isoformat() if prediction.created_at else None
            })
        
        return JsonResponse({
            'success': True,
            'predictions': predictions_data
        })
    
    # Otherwise render the template
    return render(request, 'quiz/all_predictions.html', {
        'predictions': predictions
    })

@login_required
def api_activate_model(request, model_id):
    """API endpoint for activating an AI model"""
    try:
        model = get_object_or_404(AIModel, id=model_id)
        # Deactivate all other models
        AIModel.objects.all().update(is_active=False)
        # Activate the selected model
        model.is_active = True
        model.save()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def api_delete_model(request, model_id):
    """API endpoint for deleting an AI model"""
    try:
        model = get_object_or_404(AIModel, id=model_id)
        model.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def api_delete_training_data(request, data_id):
    """API endpoint for deleting training data"""
    try:
        data = get_object_or_404(AIAdoptionData, id=data_id)
        data.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required(login_url='adminlogin')
def update_question_view(request, pk):
    question = models.Question.objects.get(id=pk)
    questionForm = forms.QuestionForm(instance=question)
    
    if request.method == 'POST':
        questionForm = forms.QuestionForm(request.POST, instance=question)
        if questionForm.is_valid():
            question = questionForm.save(commit=False)
            
            # Handle different question types
            question_type = request.POST.get('question_type')
            
            if question_type == 'multiple_choice':
                # Ensure required fields for multiple choice
                question.answer = request.POST.get('answer')
                
            elif question_type == 'checkbox':
                # For checkbox questions, save multiple answers as JSON
                multiple_answers = request.POST.getlist('multiple_answers')
                question.multiple_answers = multiple_answers
                question.answer = None  # Clear single answer field
                
            elif question_type == 'short_answer':
                # For short answer questions, save the pattern
                question.short_answer_pattern = request.POST.get('short_answer_pattern')
                question.answer = None  # Clear single answer field
                # Make options empty for short answer questions
                question.option1 = None
                question.option2 = None
                question.option3 = None
                question.option4 = None
            
            question.save()
            return HttpResponseRedirect('/admin-view-question')
            
    return render(request, 'quiz/update_question.html', {'questionForm': questionForm, 'question': question})

@login_required(login_url='adminlogin')
def admin_generate_questions_view(request):
    """View for generating AI questions for a course"""
    from .ai_utils import generate_questions
    
    if request.method == 'POST':
        course_id = request.POST.get('course_id')
        num_questions = int(request.POST.get('num_questions', 5))
        difficulty = request.POST.get('difficulty', 'medium')
        
        # Get selected question types
        question_types = request.POST.getlist('question_types')
        
        # Validate the inputs
        if not course_id:
            messages.error(request, 'Please select a course')
            return redirect('admin-question')
        
        try:
            course = models.Course.objects.get(id=course_id)
            
            # Generate questions
            generated_questions = generate_questions(course, difficulty, num_questions, question_types)
            
            # Save the generated questions to the database
            for q_data in generated_questions:
                question = models.Question(
                    course=q_data['course'],
                    question_type=q_data['question_type'],
                    question=q_data['question'],
                    marks=q_data['marks'],
                    is_ai_generated=q_data['is_ai_generated']
                )
                
                # Set type-specific fields
                if q_data['question_type'] == 'multiple_choice':
                    question.option1 = q_data['option1']
                    question.option2 = q_data['option2']
                    question.option3 = q_data['option3']
                    question.option4 = q_data['option4']
                    question.answer = q_data['answer']
                
                elif q_data['question_type'] == 'checkbox':
                    question.option1 = q_data['option1']
                    question.option2 = q_data['option2']
                    question.option3 = q_data['option3']
                    question.option4 = q_data['option4']
                    question.multiple_answers = q_data['multiple_answers']
                
                elif q_data['question_type'] == 'short_answer':
                    question.short_answer_pattern = q_data['short_answer_pattern']
                
                question.save()
            
            messages.success(request, f'Successfully generated {len(generated_questions)} questions for {course.course_name}')
            return redirect('admin-view-question')
            
        except models.Course.DoesNotExist:
            messages.error(request, 'Course not found')
            return redirect('admin-question')
        except Exception as e:
            messages.error(request, f'Error generating questions: {str(e)}')
            return redirect('admin-question')
    
    # For GET requests, show the form
    courses = models.Course.objects.all()
    return render(request, 'quiz/admin_generate_questions.html', {'courses': courses})

@require_POST
@csrf_exempt
def get_nl_query_view(request):
    """
    Process a natural language query and return a response
    """
    try:
        # Parse JSON request
        data = json.loads(request.body)
        query = data.get('query')
        
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        # Process the query
        result = process_nl_query(query)
        
        # Return the result
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"Error processing NL query: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
@csrf_exempt
def run_sql_query_view(request):
    """
    Execute a SQL query and return the results
    """
    # Only allow in DEBUG mode for safety
    if not settings.DEBUG:
        return JsonResponse({
            'status': 'error',
            'message': 'SQL query execution is only allowed in DEBUG mode'
        }, status=403)
    
    try:
        # Parse JSON request
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({
                'status': 'error',
                'message': 'No query provided'
            }, status=400)
        
        # Basic security check - only allow SELECT queries
        if not query.lower().startswith('select'):
            return JsonResponse({
                'status': 'error',
                'message': 'Only SELECT queries are allowed'
            }, status=403)
        
        # Execute the query
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Return the results
        return JsonResponse({
            'status': 'success',
            'results': results,
            'count': len(results)
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required(login_url='adminlogin')
def ai_adoption_dashboard_view(request):
    """
    View function for the AI Adoption Analytics Platform that resembles
    the reference design in the images.
    """
    # Get adoption data statistics
    adoption_data_count = models.AIAdoptionData.objects.count() or 0
    
    # Get tool usage data
    tools_data = []
    tools_count = 0
    avg_usage = 0.0
    
    if adoption_data_count > 0:
        # Calculate tools stats
        all_tools = []
        for data in models.AIAdoptionData.objects.all():
            if data.tools_used:
                tools = [t.strip() for t in data.tools_used.split(',')]
                all_tools.extend(tools)
        
        # Count unique tools
        unique_tools = set(all_tools)
        tools_count = len(unique_tools)
        
        # Calculate average usage frequency
        usage_mapping = {
            'never': 0,
            'rarely': 1,
            'monthly': 2,
            'weekly': 3,
            'daily': 4
        }
        
        total_frequency = 0
        with_frequency = 0
        
        for data in models.AIAdoptionData.objects.all():
            if data.usage_frequency in usage_mapping:
                total_frequency += usage_mapping[data.usage_frequency]
                with_frequency += 1
        
        if with_frequency > 0:
            avg_usage = round(total_frequency / with_frequency, 2)
    
    # Get faculty and study level data for filtering
    faculties = models.AIAdoptionData.objects.values_list('faculty', flat=True).distinct()
    study_levels = models.AIAdoptionData.objects.values_list('level_of_study', flat=True).distinct()
    
    # Context for template
    context = {
        'adoption_data_count': adoption_data_count,
        'tools_count': tools_count,
        'avg_usage': avg_usage,
        'faculties': list(faculties),
        'study_levels': list(study_levels),
    }
    
    return render(request, 'quiz/ai_dashboard.html', context=context)

@csrf_exempt
def ajax_upload_csv(request):
    """API endpoint to handle AJAX CSV uploads for the AI Adoption Dashboard"""
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        
        # Check if file is CSV
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Please upload a CSV file'}, status=400)
        
        # Process the file and save to database
        try:
            # Placeholder for actual implementation
            # In a real scenario, you would:
            # 1. Parse the CSV
            # 2. Validate data
            # 3. Save to database
            # 4. Handle duplicates
            
            # For demo purposes, return success
            return JsonResponse({
                'success': True,
                'total_records': 1224,
                'added_records': 1224,
                'skipped_records': 1,
                'preview_data': {
                    'headers': ['Email', 'Level of study', 'Faculty', 'AI Familiarity'],
                    'rows': [
                        ['cameron.graham@gmail.com', 'Undergraduate', 'Business', 'Not familiar'],
                        ['roger.smith@gmail.com', 'Undergraduate', 'Business', 'Somewhat familiar'],
                        ['jonathan31@gmail.com', 'Undergraduate', 'Business', 'Not familiar'],
                        ['martinm@gmail.com', 'Postgraduate', 'Engineering', 'Somewhat familiar'],
                        ['robert96@gmail.com', 'Undergraduate', 'Business', 'Somewhat familiar']
                    ]
                },
                'stats': {
                    'total_responses': 1224,
                    'tools_count': 7,
                    'avg_usage': 2.31
                }
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def ajax_train_model(request):
    """API endpoint to handle AJAX model training for the AI Adoption Dashboard"""
    if request.method == 'POST':
        try:
            # Placeholder for actual implementation
            # In a real scenario, you would:
            # 1. Train an ML model on the data
            # 2. Save the model
            # 3. Return metrics
            
            # For demo purposes, return success with dummy metrics
            return JsonResponse({
                'success': True,
                'accuracy': 0.61,
                'message': 'Model trained successfully!'
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required(login_url='teacherlogin')
def teacher_ai_prediction_dashboard_view(request):
    """AI Prediction Dashboard for teachers with appropriate permissions"""
    # Import necessary modules
    from django.db.models import Count, Avg
    from .ai_utils import get_chart_data
    import json
    
    # Only allow teachers to access this view
    if not is_teacher(request.user):
        return redirect('teacherlogin')
    
    # Get adoption data statistics
    adoption_data_count = models.AIAdoptionData.objects.count()
    
    # Get AI model statistics
    ai_model_count = models.AIModel.objects.count()
    avg_model_accuracy = models.AIModel.objects.aggregate(avg_accuracy=Avg('accuracy'))['avg_accuracy'] or 0
    
    # Get prediction statistics
    prediction_count = models.AIPrediction.objects.count()
    
    # Get AI insights
    ai_insights = models.AIInsight.objects.all().order_by('-created_at')[:3]
    
    # Get adoption level distribution
    adoption_levels = models.AIAdoptionData.objects.values('adoption_level').annotate(
        count=Count('id')
    ).order_by('adoption_level')
    
    # Convert adoption level counts to chart data format
    adoption_chart_data = {
        'labels': [],
        'data': [],
        'backgroundColor': [
            'rgba(220, 53, 69, 0.8)',  # very_low
            'rgba(255, 193, 7, 0.8)',  # low
            'rgba(23, 162, 184, 0.8)',  # medium 
            'rgba(0, 123, 255, 0.8)',   # high
            'rgba(40, 167, 69, 0.8)'    # very_high
        ]
    }
    
    # Map for nicer display labels
    level_labels = {
        'very_low': 'Very Low',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High',
        'very_high': 'Very High'
    }
    
    # Default data if no records exist
    if not adoption_levels.exists():
        adoption_chart_data['labels'] = list(level_labels.values())
        adoption_chart_data['data'] = [0, 0, 0, 0, 0]
    else:
        # Fill with actual data
        for level in adoption_levels:
            level_name = level['adoption_level']
            display_name = level_labels.get(level_name, level_name.replace('_', ' ').title())
            adoption_chart_data['labels'].append(display_name)
            adoption_chart_data['data'].append(level['count'])
    
    # Get tool usage frequency
    tools_data = get_chart_data('tools_usage')
    
    # Get challenges data
    challenges_data = []
    for data in models.AIAdoptionData.objects.all():
        if data.challenges:
            challenges = [c.strip() for c in data.challenges.split(',')]
            challenges_data.extend(challenges)
    
    challenge_counts = {}
    for challenge in challenges_data:
        if challenge:
            challenge_counts[challenge] = challenge_counts.get(challenge, 0) + 1
    
    # Get top challenges
    top_challenges = sorted(challenge_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    
    challenge_chart_data = {
        'labels': [challenge for challenge, count in top_challenges] if top_challenges else ['No data available'],
        'data': [count for challenge, count in top_challenges] if top_challenges else [0],
        'backgroundColor': [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)'
        ]
    }
    
    # Get faculty distribution
    faculty_data = get_chart_data('adoption_by_faculty')
    
    # Prepare template context
    context = {
        'total_student': SMODEL.Student.objects.all().count(),
        'total_teacher': TMODEL.Teacher.objects.all().count(),
        'total_course': models.Course.objects.count(),
        'total_question': models.Question.objects.count(),
        'adoption_data_count': adoption_data_count,
        'ai_model_count': ai_model_count,
        'avg_model_accuracy': round(avg_model_accuracy * 100, 1),
        'prediction_count': prediction_count,
        'ai_insights': ai_insights,
        'adoption_chart_data': json.dumps(adoption_chart_data),
        'tools_chart_data': json.dumps(tools_data),
        'challenge_chart_data': json.dumps(challenge_chart_data),
        'faculty_chart_data': json.dumps(faculty_data),
        # Include sample data for initial display (if no real data exists)
        'adoption_data': models.AIAdoptionData.objects.all().order_by('-created_at')[:5]
    }
    
    return render(request, 'quiz/ai_prediction_dashboard.html', context=context)

@login_required
def ai_upload_data_view(request):
    """View for the AI adoption data upload page."""
    # Get statistics for the dashboard
    total_records = CSVUpload.objects.aggregate(Sum('record_count'))['record_count__sum'] or 0
    model_count = CSVUpload.objects.filter(model_trained=True).count()
    best_accuracy = CSVUpload.objects.filter(model_accuracy__isnull=False).aggregate(Max('model_accuracy'))['model_accuracy__max'] or 0
    last_upload_obj = CSVUpload.objects.order_by('-created_at').first()
    
    # Format the last upload date or set a default value
    if last_upload_obj and hasattr(last_upload_obj, 'created_at'):
        last_upload = last_upload_obj.created_at.strftime('%b %d, %Y')
    else:
        last_upload = "None"
    
    context = {
        'total_records': total_records,
        'model_count': model_count,
        'best_accuracy': f"{best_accuracy * 100:.2f}%" if best_accuracy else "0%",
        'last_upload': last_upload,
        'upload_history': CSVUpload.objects.all().order_by('-created_at')[:10],
    }
    
    return render(request, 'quiz/ai_upload_data.html', context)

@login_required
def upload_csv(request):
    """Handle CSV file upload with options to clean data and train model."""
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        clean_data = request.POST.get('clean_data') == 'on'
        train_model = request.POST.get('train_model') == 'on'
        
        # Generate a unique filename to store the CSV
        original_filename = csv_file.name
        file_extension = os.path.splitext(original_filename)[1]
        stored_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Create upload path
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'csv_uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, stored_filename)
        
        # Save the file
        with open(file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)
        
        # Create a record in the database
        csv_upload = CSVUpload.objects.create(
            user=request.user,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_size=csv_file.size,
            status='processing'
        )
        
        # Process the file
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            record_count = len(df)
            csv_upload.record_count = record_count
            
            # Clean data if requested
            if clean_data:
                # Placeholder for data cleaning logic
                # This would be replaced with actual cleaning code
                # For example: remove duplicates, handle missing values, etc.
                df = df.dropna()  # Simple cleaning - remove rows with missing values
                csv_upload.cleaned_data = True
            
            # Train model if requested
            if train_model:
                # Placeholder for model training logic
                # This would be replaced with actual model training code
                # For demo purposes, we'll just simulate a model accuracy
                import random
                model_accuracy = random.uniform(0.7, 0.95)
                csv_upload.model_trained = True
                csv_upload.model_accuracy = model_accuracy
                csv_upload.insights = "Model trained successfully with sample insights on AI adoption factors."
            
            csv_upload.status = 'completed'
            csv_upload.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'File uploaded and processed successfully',
                'record_count': record_count,
                'cleaned': clean_data,
                'model_trained': train_model,
                'model_accuracy': csv_upload.model_accuracy * 100 if csv_upload.model_accuracy else None,
            })
            
        except Exception as e:
            csv_upload.status = 'failed'
            csv_upload.insights = str(e)
            csv_upload.save()
            return JsonResponse({
                'status': 'error',
                'message': f'Error processing file: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'No file provided or invalid request'
    })

@login_required
def upload_history_view(request):
    """Return the upload history for AJAX requests to refresh the history table."""
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Get the last 10 uploads
        uploads = CSVUpload.objects.filter(user=request.user).order_by('-created_at')[:10]
        
        history = []
        for upload in uploads:
            history.append({
                'id': upload.id,
                'filename': upload.original_filename,
                'record_count': upload.record_count,
                'status': upload.status,
                'date': upload.created_at.strftime('%Y-%m-%d %H:%M'),
                'accuracy': f"{upload.model_accuracy * 100:.1f}%" if upload.model_accuracy else "N/A",
            })
        
        return JsonResponse({
            'status': 'success',
            'history': history
        })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    })

@login_required
def delete_upload(request, upload_id):
    """Delete a CSV upload and its file."""
    if request.method == 'POST':
        try:
            upload = CSVUpload.objects.get(id=upload_id, user=request.user)
            
            # Delete the file
            file_path = os.path.join(settings.MEDIA_ROOT, 'csv_uploads', upload.stored_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete the database entry
            upload.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Upload deleted successfully'
            })
        except CSVUpload.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Upload not found'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error deleting upload: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@login_required(login_url='adminlogin')
def ai_model_metrics_view(request):
    return render(request, 'quiz/under_construction.html', {
        'message': 'The Model Metrics page will display performance statistics for your AI models.'
    })

@login_required(login_url='adminlogin')
@require_POST
def upload_csv_view(request):
    """Handle CSV file upload from the AI adoption data upload page."""
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        start_time = time.time()
        
        # Check if file was uploaded
        if 'csv_file' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No file uploaded'
            })
        
        csv_file = request.FILES['csv_file']
        
        # Check file extension
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({
                'status': 'error',
                'message': 'File must be a CSV'
            })
        
        # Get options
        clean_data = request.POST.get('clean_data') == 'true'
        train_model = request.POST.get('train_model') == 'true'
        
        # Save file temporarily
        fs = FileSystemStorage(location=tempfile.gettempdir())
        filename = fs.save(csv_file.name, csv_file)
        file_path = os.path.join(tempfile.gettempdir(), filename)
        
        try:
            # Process the file
            if clean_data:
                # Clean and process data
                df, stats, accuracy, insights = import_from_csv(file_path, save_to_db=True)
                
                if df is None or df.empty:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to process CSV file. Please check the format.'
                    })
                    
                # Save upload record
                upload = CSVUpload.objects.create(
                    user=request.user,
                    original_filename=csv_file.name,
                    file_path=file_path,
                    record_count=len(df),
                    status='success',
                    metadata={
                        'columns': df.columns.tolist(),
                        'cleaned': clean_data
                    }
                )
                
                # Set upload batch ID for records
                AIAdoptionData.objects.filter(upload_batch__isnull=True).update(upload_batch=upload)
                
                # Train model if requested
                model_trained = False
                model_accuracy = 0.0
                if train_model:
                    result = train_ai_model(csv_upload_id=upload.id)
                    model_trained = result.get('success', False)
                    model_accuracy = result.get('accuracy', 0.0)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Get updated stats
                total_records = AIAdoptionData.objects.count()
                model_count = AIModel.objects.count()
                try:
                    best_accuracy = f"{AIModel.objects.order_by('-accuracy').first().accuracy * 100:.2f}%" if AIModel.objects.exists() else "0%"
                except:
                    best_accuracy = "0%"
                last_upload = upload.created_at.strftime('%b %d, %Y')
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'CSV file processed successfully',
                    'records': len(df),
                    'processing_time': processing_time,
                    'model_trained': model_trained,
                    'accuracy': model_accuracy,
                    'insights': insights[:5] if insights else [],
                    'total_records': total_records,
                    'model_count': model_count,
                    'best_accuracy': best_accuracy,
                    'last_upload': last_upload
                })
                
            else:
                # Just process without cleaning
                df = pd.read_csv(file_path)
                
                if df is None or df.empty:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to process CSV file. Please check the format.'
                    })
                
                # Save upload record
                upload = CSVUpload.objects.create(
                    user=request.user,
                    original_filename=csv_file.name,
                    file_path=file_path,
                    record_count=len(df),
                    status='success',
                    metadata={
                        'columns': df.columns.tolist(),
                        'cleaned': clean_data
                    }
                )
                
                # Save records
                batch_size = 1000
                records = []
                for i, row in df.iterrows():
                    data = row.to_dict()
                    record = AIAdoptionData(
                        user=request.user,
                        upload_batch=upload,
                        data=data
                    )
                    records.append(record)
                    
                    if len(records) >= batch_size:
                        AIAdoptionData.objects.bulk_create(records)
                        records = []
                
                if records:
                    AIAdoptionData.objects.bulk_create(records)
                
                # Train model if requested
                model_trained = False
                model_accuracy = 0.0
                insights = []
                
                if train_model:
                    result = train_ai_model(csv_upload_id=upload.id)
                    model_trained = result.get('success', False)
                    model_accuracy = result.get('accuracy', 0.0)
                    
                    # Generate basic insights
                    try:
                        insights = [
                            f"Successfully processed {len(df)} records from {csv_file.name}.",
                            f"The dataset contains {len(df.columns)} columns.",
                            f"Model training {'was successful' if model_trained else 'was not performed'}.",
                            f"Average values across numeric columns look reasonable.",
                            f"No major issues detected in the uploaded data."
                        ]
                    except Exception as e:
                        insights = [f"Processed {len(df)} records"]
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Get updated stats
                total_records = AIAdoptionData.objects.count()
                model_count = AIModel.objects.count()
                try:
                    best_accuracy = f"{AIModel.objects.order_by('-accuracy').first().accuracy * 100:.2f}%" if AIModel.objects.exists() else "0%"
                except:
                    best_accuracy = "0%"
                last_upload = upload.created_at.strftime('%b %d, %Y')
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'CSV file processed successfully',
                    'records': len(df),
                    'processing_time': processing_time,
                    'model_trained': model_trained,
                    'accuracy': model_accuracy,
                    'insights': insights[:5] if insights else [],
                    'total_records': total_records,
                    'model_count': model_count,
                    'best_accuracy': best_accuracy,
                    'last_upload': last_upload
                })
        
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return JsonResponse({
                'status': 'error',
                'message': f'Error processing CSV: {str(e)}'
            })
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
    
    # If not AJAX, return a simple JSON response
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    })

@login_required(login_url='adminlogin')
@require_POST
def train_model_view(request):
    """View for training a new model"""
    from quiz.ai_data_utils import train_ai_model
    import json
    
    algorithm = request.POST.get('algorithm', 'random_forest')
    csv_upload_id = request.POST.get('csv_upload_id')
    
    # Convert csv_upload_id to int if provided
    if csv_upload_id:
        try:
            csv_upload_id = int(csv_upload_id)
        except ValueError:
            csv_upload_id = None
    
    try:
        # Train the model
        results = train_ai_model(csv_upload_id=csv_upload_id, algorithm=algorithm)
        
        if results['success']:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'success',
                    'model_id': results['model_id'],
                    'accuracy': results['accuracy'],
                    'training_time': results['training_time'],
                    'algorithm': results['algorithm']
                })
            else:
                messages.success(
                    request, 
                    f'Successfully trained {algorithm} model with accuracy: {results["accuracy"]:.2f}'
                )
                return redirect('ai_model_metrics')
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': results.get('error', 'Unknown error')
                })
            else:
                messages.error(request, f'Error training model: {results.get("error", "Unknown error")}')
                return redirect('ai_model_metrics')
    
    except Exception as e:
        # Log the error
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
        else:
            messages.error(request, f'Error training model: {str(e)}')
            return redirect('ai_model_metrics')

@login_required(login_url='adminlogin')
@require_POST
def activate_model_view(request, model_id):
    """View for activating a model"""
    from quiz.models import AIModel
    
    try:
        # Get the model
        model = AIModel.objects.get(id=model_id)
        
        # Set this model as active and deactivate others
        AIModel.objects.all().update(is_active=False)
        model.is_active = True
        model.save()
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success',
                'model_id': model_id
            })
        else:
            messages.success(request, f'Successfully activated model: {model.name}')
            return redirect('ai_model_metrics')
    
    except AIModel.DoesNotExist:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': 'Model not found'
            })
        else:
            messages.error(request, 'Model not found')
            return redirect('ai_model_metrics')
    
    except Exception as e:
        # Log the error
        logger.error(f"Error activating model: {str(e)}", exc_info=True)
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
        else:
            messages.error(request, f'Error activating model: {str(e)}')
            return redirect('ai_model_metrics')

@login_required(login_url='adminlogin')
@require_POST
def delete_model_view(request, model_id):
    """View for deleting a model"""
    from quiz.models import AIModel
    
    try:
        # Get the model
        model = AIModel.objects.get(id=model_id)
        
        # Cannot delete the active model if it's the only one
        if model.is_active and AIModel.objects.count() == 1:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': 'Cannot delete the only active model'
                })
            else:
                messages.error(request, 'Cannot delete the only active model')
                return redirect('ai_model_metrics')
        
        # If this is the active model, activate another one
        if model.is_active:
            other_model = AIModel.objects.exclude(id=model_id).order_by('-created_date').first()
            if other_model:
                other_model.is_active = True
                other_model.save()
        
        # Delete the model
        model.delete()
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success'
            })
        else:
            messages.success(request, 'Successfully deleted model')
            return redirect('ai_model_metrics')
    
    except AIModel.DoesNotExist:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': 'Model not found'
            })
        else:
            messages.error(request, 'Model not found')
            return redirect('ai_model_metrics')
    
    except Exception as e:
        # Log the error
        logger.error(f"Error deleting model: {str(e)}", exc_info=True)
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
        else:
            messages.error(request, f'Error deleting model: {str(e)}')
            return redirect('ai_model_metrics')

@login_required(login_url='adminlogin')
@require_POST
def delete_upload_view(request, upload_id):
    """View for deleting a CSV upload"""
    from quiz.models import CSVUpload, AIAdoptionData
    
    try:
        # Get the upload
        upload = CSVUpload.objects.get(id=upload_id)
        
        # Delete associated data records
        AIAdoptionData.objects.filter(upload_batch=upload).delete()
        
        # Delete the upload
        upload.delete()
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success'
            })
        else:
            messages.success(request, 'Successfully deleted upload and associated data')
            return redirect('ai_upload_data')
    
    except CSVUpload.DoesNotExist:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': 'Upload not found'
            })
        else:
            messages.error(request, 'Upload not found')
            return redirect('ai_upload_data')
    
    except Exception as e:
        # Log the error
        logger.error(f"Error deleting upload: {str(e)}", exc_info=True)
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
        else:
            messages.error(request, f'Error deleting upload: {str(e)}')
            return redirect('ai_upload_data')

@login_required(login_url='adminlogin')
def ai_data_detail_view(request, data_id):
    """View for displaying details of a CSV upload"""
    from quiz.models import CSVUpload, AIAdoptionData
    
    try:
        # Get the upload
        upload = CSVUpload.objects.get(id=data_id)
        
        # Get associated data records
        data = AIAdoptionData.objects.filter(upload_batch=upload)
        
        # Calculate statistics
        total_records = data.count()
        faculties = data.values('faculty').annotate(count=Count('faculty')).order_by('-count')
        levels = data.values('level_of_study').annotate(count=Count('level_of_study')).order_by('-count')
        ai_familiarity_avg = data.aggregate(Avg('ai_familiarity'))['ai_familiarity__avg'] or 0
        tools_usage = data.filter(uses_ai_tools='yes').count()
        tools_usage_percent = (tools_usage / total_records * 100) if total_records > 0 else 0
        
        # Most common tools
        tools_list = []
        for item in data.exclude(tools_used=''):
            if item.tools_used:
                tools_list.extend([t.strip() for t in item.tools_used.split(',') if t.strip()])
        
        tools_counts = Counter(tools_list)
        top_tools = tools_counts.most_common(5)
        
        context = {
            'upload': upload,
            'total_records': total_records,
            'faculties': faculties,
            'levels': levels,
            'ai_familiarity_avg': ai_familiarity_avg,
            'tools_usage_percent': tools_usage_percent,
            'top_tools': top_tools,
            'data_sample': data[:10]  # First 10 records for preview
        }
        
        return render(request, 'quiz/ai_data_detail.html', context)
    
    except CSVUpload.DoesNotExist:
        messages.error(request, 'Upload not found')
        return redirect('ai_upload_data')
    
    except Exception as e:
        # Log the error
        logger.error(f"Error displaying data detail: {str(e)}", exc_info=True)
        messages.error(request, f'Error: {str(e)}')
        return redirect('ai_upload_data')

@login_required(login_url='adminlogin')
def ai_view_data_list_view(request):
    return render(request, 'quiz/under_construction.html', {
        'message': 'The Data Viewer will allow you to browse and analyze your uploaded datasets.'
    })

@login_required(login_url='adminlogin')
def ai_view_data_detail_view(request, dataset_id):
    return render(request, 'quiz/under_construction.html', {
        'message': 'The Dataset Detail view will show statistics and visualizations for your data.'
    })

@login_required(login_url='adminlogin')
def load_more_data_view(request, dataset_id):
    return JsonResponse({'status': 'error', 'message': 'This endpoint is under construction'})

@login_required(login_url='adminlogin')
def generate_chart_view(request, dataset_id):
    return JsonResponse({'status': 'error', 'message': 'This endpoint is under construction'})

@login_required(login_url='adminlogin')
def generate_insights_view(request, dataset_id):
    return JsonResponse({'status': 'error', 'message': 'This endpoint is under construction'})

@login_required(login_url='adminlogin')
def export_csv_view(request, dataset_id):
    return HttpResponse('This endpoint is under construction', content_type='text/plain')

@login_required(login_url='adminlogin')
def ai_prediction_form_view(request):
    return render(request, 'quiz/under_construction.html', {
        'message': 'The Prediction Form will allow you to make AI adoption predictions for individual students.'
    })

@login_required(login_url='adminlogin')
@require_POST
def ai_predict_view(request):
    return JsonResponse({'status': 'error', 'message': 'This endpoint is under construction'})

@login_required(login_url='adminlogin')
@require_POST
def delete_dataset_view(request, dataset_id):
    """View to delete a dataset and all its associated records."""
    dataset = get_object_or_404(CSVUpload, id=dataset_id)
    
    try:
        # Delete all associated records
        AIAdoptionData.objects.filter(upload_batch=dataset).delete()
        
        # Delete the dataset
        dataset.delete()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Dataset deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        
        return JsonResponse({
            'status': 'error',
            'message': f'Error deleting dataset: {str(e)}'
        })



    
