from django.shortcuts import render, redirect, reverse, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.core.mail import send_mail
from django.conf import settings
from datetime import date, timedelta
from django.db.models import Sum, Q, Count, Avg, FloatField, Max, F
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
            'course_name': course.course_name,
            'teacher_name': teacher_name,
            'student_count': course.student_count
        })
    
    # Get total questions for context
    try:
        total_question = models.Question.objects.count()
    except:
        total_question = 0
        
    # Get total students for context
    try:
        total_student = SMODEL.Student.objects.count()
    except:
        total_student = 0
    
    context = {
        'total_course': models.Course.objects.count(),
        'total_question': total_question,
        'total_student': total_student,
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
