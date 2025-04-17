from django.shortcuts import render,redirect,reverse, get_object_or_404
from . import forms,models
from django.db.models import Sum, Count
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
from datetime import date, timedelta, timezone
from quiz import models as QMODEL
from teacher import models as TMODEL
from django.contrib.auth import logout
from .ai_utils import get_ai_response, get_student_ai_insights, extract_topics
import json
import logging
import requests
from decouple import config

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
    
    # Calculate time per question if this is a timed exam
    time_per_question = 0
    if course.is_timed and total_questions > 0:
        total_seconds = course.total_time_minutes * 60
        time_per_question = round(total_seconds / total_questions)
    
    return render(request,'student/take_exam.html',{
        'course': course,
        'total_questions': total_questions,
        'total_marks': total_marks,
        'time_per_question': time_per_question
    })

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def start_exam_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    student = models.Student.objects.get(user_id=request.user.id)
    
    # Check if this course uses timed sequential questions
    if course.is_timed and course.sequential_questions:
        # Redirect to sequential exam view
        return HttpResponseRedirect(reverse('start-sequential-exam', args=[pk]))
    
    # Check if adaptive quizzing is enabled for this course
    try:
        adaptive_settings = QMODEL.AdaptiveQuizSettings.objects.get(course=course)
        is_adaptive = adaptive_settings.is_adaptive
    except QMODEL.AdaptiveQuizSettings.DoesNotExist:
        is_adaptive = False
    
    # Get questions based on adaptive settings or standard approach
    if is_adaptive:
        # Import our new adaptive quiz utility
        from quiz.adaptive_quiz_utils import get_adaptive_questions
        questions = get_adaptive_questions(student, course, course.question_number)
    else:
        # Fallback to existing behavior
        questions = QMODEL.Question.objects.all().filter(course=course)
    
    if request.method=='POST':
        pass
    response= render(request,'student/start_exam.html',{'course':course,'questions':questions})
    response.set_cookie('course_id',course.id)
    return response

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def start_sequential_exam_view(request, pk):
    """
    Start a sequential timed exam where questions are shown one at a time
    with specific time limits per question
    """
    course = QMODEL.Course.objects.get(id=pk)
    student = models.Student.objects.get(user_id=request.user.id)
    
    # Check if adaptive quizzing is enabled for this course
    try:
        adaptive_settings = QMODEL.AdaptiveQuizSettings.objects.get(course=course)
        is_adaptive = adaptive_settings.is_adaptive
    except QMODEL.AdaptiveQuizSettings.DoesNotExist:
        is_adaptive = False
    
    # Get questions based on adaptive settings or standard approach
    if is_adaptive:
        # Import our new adaptive quiz utility
        from quiz.adaptive_quiz_utils import get_adaptive_questions
        questions = get_adaptive_questions(student, course, course.question_number)
    else:
        # Fallback to existing behavior
        questions = QMODEL.Question.objects.all().filter(course=course)
    
    # Render the sequential exam template
    response = render(request, 'student/sequential_exam.html', {
        'course': course,
        'questions': questions,
        'hide_sidebar': True,
    })
    
    # Set cookie for exam identification
    response.set_cookie('course_id', course.id)
    
    return response

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
@csrf_protect
def calculate_marks_view(request):
    if request.COOKIES.get('exam_total_time'):
        total_time = int(request.COOKIES.get('exam_total_time'))
    else:
        total_time = 0
    
    course = QMODEL.Course.objects.get(id=request.COOKIES.get('course_id'))
    total_marks = 0
    questions = QMODEL.Question.objects.all().filter(course=course)
    adaptive_settings = None
    try:
        adaptive_settings = QMODEL.AdaptiveQuizSettings.objects.get(course=course)
    except QMODEL.AdaptiveQuizSettings.DoesNotExist:
        pass

    results = []
    total_correct_answers = 0
    total_attempted = 0
    
    for i in range(len(questions)):
        attempted = False
        question = questions[i]
        time_spent = 0
        
        # Get time spent on this question if available
        question_time_cookie = request.COOKIES.get(f'q_{question.id}_final_time')
        if question_time_cookie:
            time_spent = int(question_time_cookie)
        
        # Process based on question type
        if question.question_type == 'multiple_choice':
            # Handle regular multiple choice (radio buttons)
            selected_ans = request.COOKIES.get(str(i+1))
            if selected_ans:
                attempted = True
                if selected_ans == question.answer:
                    total_marks += question.marks
                    total_correct_answers += 1
        
        elif question.question_type == 'checkbox':
            # Handle checkbox/multiple answer questions
            checkbox_selection = request.COOKIES.get(f'checkbox_{question.id}')
            if checkbox_selection:
                attempted = True
                try:
                    selected_options = json.loads(checkbox_selection)
                    # Compare with the expected multiple answers
                    expected_answers = question.multiple_answers.split(',')
                    
                    # Convert option values (Option1, Option2) to expected format if needed
                    selected_options_normalized = [opt.replace('Option', '') for opt in selected_options]
                    
                    # Check if selections match expected answers
                    if sorted(selected_options_normalized) == sorted(expected_answers):
                        total_marks += question.marks
                        total_correct_answers += 1
                except json.JSONDecodeError:
                    # Handle invalid JSON
                    pass
        
        elif question.question_type == 'short_answer':
            # Handle short answer questions
            short_answer = request.COOKIES.get(f'{i+1}_short')
            if short_answer:
                attempted = True
                # Use regex pattern matching for short answers
                if question.short_answer_pattern:
                    import re
                    pattern = re.compile(question.short_answer_pattern, re.IGNORECASE)
                    if pattern.match(short_answer):
                        total_marks += question.marks
                        total_correct_answers += 1
                # Fallback to exact match if no pattern specified
                elif short_answer.lower().strip() == question.answer.lower().strip():
                    total_marks += question.marks
                    total_correct_answers += 1
        
        # Track question attempt data
        if attempted:
            total_attempted += 1
            # Store the attempt data for this question
            results.append({
                'question': question,
                'selected_answer': selected_ans if question.question_type == 'multiple_choice' else 
                                 checkbox_selection if question.question_type == 'checkbox' else 
                                 short_answer,
                'time_spent': time_spent,
                'is_correct': selected_ans == question.answer if question.question_type == 'multiple_choice' else False
            })
    
    # Compute success metrics
    success_rate = (total_correct_answers / len(questions)) * 100 if len(questions) > 0 else 0
    attempt_rate = (total_attempted / len(questions)) * 100 if len(questions) > 0 else 0
    
    # Create and save the Result object
    student = models.Student.objects.get(user_id=request.user.id)
    result = QMODEL.Result()
    result.marks = total_marks
    result.exam = course
    result.student = student
    result.save()
    
    # Save individual question attempts if the model exists
    for res in results:
        question_attempt = QMODEL.QuestionAttempt(
            student=student,
            question=res['question'],
            result=result,
            selected_answer=res['selected_answer'],
            time_spent=res['time_spent']
        )
        question_attempt.save()
    
    # Clear cookies
    response = render(request, 'student/check_marks.html', {'course': course, 'total_marks': total_marks, 
                                                          'total_time': total_time, 'success_rate': success_rate, 
                                                          'adaptive_settings': adaptive_settings})
    for i in range(len(questions)):
        response.delete_cookie(str(i+1))
    
    # Clear other cookies related to the exam
    response.delete_cookie('course_id')
    response.delete_cookie('exam_total_time')
    
    # Render the results page
    return response

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def view_result_view(request):
    courses = QMODEL.Course.objects.all()
    
    # Check if we have a recent quiz result to show
    last_result_id = request.session.get('last_result_id')
    last_attempt_ids = request.session.get('last_attempt_ids', [])
    
    recent_result = None
    recent_attempts = []
    
    if last_result_id:
        try:
            recent_result = QMODEL.Result.objects.get(id=last_result_id)
            # Clear from session after retrieving
            del request.session['last_result_id']
        except QMODEL.Result.DoesNotExist:
            pass
    
    if last_attempt_ids:
        recent_attempts = QMODEL.QuestionAttempt.objects.filter(id__in=last_attempt_ids)
        # Clear from session after retrieving
        if 'last_attempt_ids' in request.session:
            del request.session['last_attempt_ids']
    
    # Get recommendations for the student
    student = models.Student.objects.get(user_id=request.user.id)
    recommendations = QMODEL.ContentRecommendation.objects.filter(
        student=student,
        is_viewed=False
    ).order_by('-relevance_score', '-created_at')[:5]
    
    context = {
        'courses': courses,
        'recent_result': recent_result,
        'recent_attempts': recent_attempts,
        'recommendations': recommendations
    }
    
    return render(request, 'student/view_result.html', context)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def question_feedback_view(request, attempt_id):
    """View detailed feedback and explanation for a quiz question attempt"""
    attempt = get_object_or_404(QMODEL.QuestionAttempt, id=attempt_id, student__user=request.user)
    question = attempt.question
    
    # Get feedback for this attempt
    try:
        feedback = QMODEL.StudentFeedback.objects.get(question_attempt=attempt)
    except QMODEL.StudentFeedback.DoesNotExist:
        # If feedback doesn't exist yet, generate it
        from .ai_utils import get_ai_feedback
        feedback_text = get_ai_feedback(attempt)
        feedback = QMODEL.StudentFeedback.objects.create(
            question_attempt=attempt,
            feedback_text=feedback_text,
            is_correct=attempt.is_correct
        )
    
    # Get explanation for this question
    try:
        explanation = QMODEL.QuestionExplanation.objects.get(question=question)
    except QMODEL.QuestionExplanation.DoesNotExist:
        # If explanation doesn't exist yet, generate it
        from .ai_utils import get_ai_explanation
        explanation_text = get_ai_explanation(question)
        explanation = QMODEL.QuestionExplanation.objects.create(
            question=question,
            explanation_text=explanation_text
        )
    
    # Get recommendations related to this topic
    recommendations = []
    topics = extract_topics(question.question)
    if topics:
        recommendations = QMODEL.ContentRecommendation.objects.filter(
            student__user=request.user,
            description__icontains=topics[0]  # Simple matching for demo
        ).order_by('-relevance_score')[:3]
    
    context = {
        'attempt': attempt,
        'question': question,
        'feedback': feedback,
        'explanation': explanation,
        'recommendations': recommendations
    }
    
    return render(request, 'student/question_feedback.html', context)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def check_marks_view(request, pk):
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

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_analytics_view(request):
    student = models.Student.objects.get(user_id=request.user.id)
    course_id = request.GET.get('course')
    
    # Get all courses
    courses = QMODEL.Course.objects.all()
    
    # If course is specified, get detailed analytics for that course
    detailed_analytics = None
    selected_course = None
    
    if course_id:
        try:
            selected_course = QMODEL.Course.objects.get(id=course_id)
            # Use our analytics generator
            from quiz.adaptive_quiz_utils import generate_quiz_analytics
            detailed_analytics = generate_quiz_analytics(student, selected_course)
        except QMODEL.Course.DoesNotExist:
            pass
    
    # Get high-level analytics across all courses
    course_performance = []
    for course in courses:
        # Get total results and average score
        results = QMODEL.Result.objects.filter(student=student, exam=course)
        
        if results.exists():
            avg_score = results.aggregate(avg_marks=Sum('marks') / Count('id'))['avg_marks'] or 0
            attempts = results.count()
            
            # Get skill level if it exists
            try:
                skill_level = QMODEL.StudentSkillLevel.objects.get(student=student, course=course)
                level = round(skill_level.current_level, 1)
                confidence = round(skill_level.confidence * 100, 1)
            except QMODEL.StudentSkillLevel.DoesNotExist:
                level = "N/A"
                confidence = "N/A"
            
            course_performance.append({
                'course': course,
                'attempts': attempts,
                'avg_score': round(avg_score, 1),
                'skill_level': level,
                'confidence': confidence
            })
    
    context = {
        'courses': courses,
        'course_performance': course_performance,
        'selected_course': selected_course,
        'detailed_analytics': detailed_analytics
    }
    
    return render(request, 'student/student_analytics.html', context)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def check_ai_connection_view(request):
    """Check if the AI service is available"""
    try:
        # Get the OpenRouter API key from settings
        openrouter_api_key = config('OPENROUTER_API_KEY', default=None)
        
        # If no API key is configured, we are in mock mode
        if not openrouter_api_key:
            # In development/testing mode with mock responses
            return JsonResponse({'connected': True, 'mode': 'mock'})
        
        # Check the OpenRouter API connection
        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {
            'Authorization': f'Bearer {openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=3)
        
        if response.status_code == 200:
            return JsonResponse({'connected': True, 'mode': 'api'})
        else:
            logger.error(f"API connection check failed: {response.status_code}")
            return JsonResponse({'connected': False, 'error': 'API authentication failed'})
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection check failed: {str(e)}")
        return JsonResponse({'connected': False, 'error': 'Connection timeout'})
    except Exception as e:
        logger.error(f"Unexpected error checking AI connection: {str(e)}")
        return JsonResponse({'connected': False, 'error': str(e)})
