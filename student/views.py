from django.shortcuts import render,redirect,reverse, get_object_or_404
from . import forms,models
from django.db.models import Sum
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.conf import settings
from datetime import date, timedelta, timezone
from quiz import models as QMODEL
from teacher import models as TMODEL
from django.contrib.auth import logout
from .ai_utils import get_ai_response, get_student_ai_insights
import json
import logging

logger = logging.getLogger(__name__)

#for showing signup/login button for student
def studentclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return render(request,'student/studentclick.html')

def student_signup_view(request):
    userForm=forms.StudentUserForm()
    studentForm=forms.StudentForm()
    mydict={'userForm':userForm,'studentForm':studentForm}
    if request.method=='POST':
        userForm=forms.StudentUserForm(request.POST)
        studentForm=forms.StudentForm(request.POST,request.FILES)
        if userForm.is_valid() and studentForm.is_valid():
            user=userForm.save()
            user.set_password(user.password)
            user.save()
            student=studentForm.save(commit=False)
            student.user=user
            student.save()
            my_student_group = Group.objects.get_or_create(name='STUDENT')
            my_student_group[0].user_set.add(user)
        return HttpResponseRedirect('studentlogin')
    return render(request,'student/studentsignup.html',context=mydict)

def is_student(user):
    return user.groups.filter(name='STUDENT').exists()

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_dashboard_view(request):
    student = models.Student.objects.get(user_id=request.user.id)
    ai_insights = get_student_ai_insights(student)
    
    dict={
        'total_course':QMODEL.Course.objects.all().count(),
        'total_question':QMODEL.Question.objects.all().count(),
        'ai_insights': ai_insights,
        'courses': QMODEL.Course.objects.all(),
    }
    return render(request,'student/student_dashboard.html',context=dict)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_exam_view(request):
    courses=QMODEL.Course.objects.all()
    return render(request,'student/student_exam.html',{'courses':courses})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def take_exam_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    total_questions=QMODEL.Question.objects.all().filter(course=course).count()
    questions=QMODEL.Question.objects.all().filter(course=course)
    total_marks=0
    for q in questions:
        total_marks=total_marks + q.marks
    
    return render(request,'student/take_exam.html',{'course':course,'total_questions':total_questions,'total_marks':total_marks})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def start_exam_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    questions=QMODEL.Question.objects.all().filter(course=course)
    if request.method=='POST':
        pass
    response= render(request,'student/start_exam.html',{'course':course,'questions':questions})
    response.set_cookie('course_id',course.id)
    return response


@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def calculate_marks_view(request):
    if request.COOKIES.get('course_id') is not None:
        course_id = request.COOKIES.get('course_id')
        course=QMODEL.Course.objects.get(id=course_id)
        
        total_marks=0
        questions=QMODEL.Question.objects.all().filter(course=course)
        for i in range(len(questions)):
            
            selected_ans = request.COOKIES.get(str(i+1))
            actual_answer = questions[i].answer
            if selected_ans == actual_answer:
                total_marks = total_marks + questions[i].marks
        student = models.Student.objects.get(user_id=request.user.id)
        result = QMODEL.Result()
        result.marks=total_marks
        result.exam=course
        result.student=student
        result.save()

        return HttpResponseRedirect('view-result')



@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def view_result_view(request):
    courses=QMODEL.Course.objects.all()
    return render(request,'student/view_result.html',{'courses':courses})
    

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def check_marks_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    student = models.Student.objects.get(user_id=request.user.id)
    results= QMODEL.Result.objects.all().filter(exam=course).filter(student=student)
    return render(request,'student/check_marks.html',{'results':results})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_marks_view(request):
    courses=QMODEL.Course.objects.all()
    return render(request,'student/student_marks.html',{'courses':courses})

@login_required
def logout_view(request):
    logout(request)
    return redirect('studentlogin')

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def ai_chat_view(request):
    student = get_object_or_404(models.Student, user=request.user)
    selected_course = request.GET.get('course')
    
    # Get course list for the dropdown
    courses = QMODEL.Course.objects.all()
    
    # Get chat history
    chat_history = models.AIChatHistory.objects.filter(student=student)
    if selected_course:
        chat_history = chat_history.filter(course_id=selected_course)
    chat_history = chat_history.order_by('timestamp')
    
    # Get AI insights
    insights = get_student_ai_insights(student)
    
    if request.method == 'POST':
        question = request.POST.get('question')
        course = None
        if selected_course:
            course = get_object_or_404(QMODEL.Course, id=selected_course)
        
        try:
            response = get_ai_response(question, student, course)
            return JsonResponse({
                'status': 'success',
                'response': response,
                'timestamp': timezone.now().strftime('%g:%i %A')
            })
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to get AI response. Please try again.'
            }, status=500)
    
    context = {
        'student': student,
        'courses': courses,
        'selected_course': selected_course,
        'chat_history': chat_history,
        'insights': insights
    }
    return render(request, 'student/ai_chat.html', context)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def get_ai_response_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question')
            course_id = data.get('course_id')
            
            if not question:
                return JsonResponse({'error': 'Question is required'}, status=400)
            
            student = models.Student.objects.get(user_id=request.user.id)
            course = None
            
            if course_id:
                try:
                    course = QMODEL.Course.objects.get(id=course_id)
                except QMODEL.Course.DoesNotExist:
                    logger.warning(f"Course with ID {course_id} not found")
            
            response = get_ai_response(question, student, course)
            return JsonResponse({'response': response})
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON data received")
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except models.Student.DoesNotExist:
            logger.error(f"Student not found for user ID {request.user.id}")
            return JsonResponse({'error': 'Student not found'}, status=404)
        except Exception as e:
            logger.error(f"Error in get_ai_response_view: {str(e)}")
            return JsonResponse({'error': 'An unexpected error occurred'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
