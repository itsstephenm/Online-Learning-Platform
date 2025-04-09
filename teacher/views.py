from django.shortcuts import render,redirect,reverse
from . import forms,models
from django.db.models import Sum, Avg, Count
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.conf import settings
from datetime import date, timedelta
from quiz import models as QMODEL
from student import models as SMODEL
from quiz import forms as QFORM
import json
import docx2txt
import PyPDF2
from io import BytesIO
from .ai_exam_utils import get_ai_exam_questions, save_ai_generated_exam


#for showing signup/login button for teacher
def teacherclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return render(request,'teacher/teacherclick.html')

def teacher_signup_view(request):
    userForm=forms.TeacherUserForm()
    teacherForm=forms.TeacherForm()
    mydict={'userForm':userForm,'teacherForm':teacherForm}
    if request.method=='POST':
        userForm=forms.TeacherUserForm(request.POST)
        teacherForm=forms.TeacherForm(request.POST,request.FILES)
        if userForm.is_valid() and teacherForm.is_valid():
            user=userForm.save()
            user.set_password(user.password)
            user.save()
            teacher=teacherForm.save(commit=False)
            teacher.user=user
            teacher.save()
            my_teacher_group = Group.objects.get_or_create(name='TEACHER')
            my_teacher_group[0].user_set.add(user)
        return HttpResponseRedirect('teacherlogin')
    return render(request,'teacher/teachersignup.html',context=mydict)



def is_teacher(user):
    return user.groups.filter(name='TEACHER').exists()

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_dashboard_view(request):
    dict={
    
    'total_course':QMODEL.Course.objects.all().count(),
    'total_question':QMODEL.Question.objects.all().count(),
    'total_student':SMODEL.Student.objects.all().count()
    }
    return render(request,'teacher/teacher_dashboard.html',context=dict)

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_exam_view(request):
    return render(request,'teacher/teacher_exam.html')


@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_add_exam_view(request):
    courseForm=QFORM.CourseForm()
    if request.method=='POST':
        courseForm=QFORM.CourseForm(request.POST)
        if courseForm.is_valid():        
            courseForm.save()
        else:
            print("form is invalid")
        return HttpResponseRedirect('/teacher/teacher-view-exam')
    return render(request,'teacher/teacher_add_exam.html',{'courseForm':courseForm})

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_view_exam_view(request):
    courses = QMODEL.Course.objects.all()
    return render(request,'teacher/teacher_view_exam.html',{'courses':courses})

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def delete_exam_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    course.delete()
    return HttpResponseRedirect('/teacher/teacher-view-exam')

@login_required(login_url='adminlogin')
def teacher_question_view(request):
    return render(request,'teacher/teacher_question.html')

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_add_question_view(request):
    questionForm=QFORM.QuestionForm()
    if request.method=='POST':
        questionForm=QFORM.QuestionForm(request.POST)
        if questionForm.is_valid():
            question=questionForm.save(commit=False)
            course=QMODEL.Course.objects.get(id=request.POST.get('courseID'))
            question.course=course
            question.save()       
        else:
            print("form is invalid")
        return HttpResponseRedirect('/teacher/teacher-view-question')
    return render(request,'teacher/teacher_add_question.html',{'questionForm':questionForm})

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_view_question_view(request):
    courses= QMODEL.Course.objects.all()
    return render(request,'teacher/teacher_view_question.html',{'courses':courses})

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def see_question_view(request,pk):
    questions=QMODEL.Question.objects.all().filter(course_id=pk)
    return render(request,'teacher/see_question.html',{'questions':questions})

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def remove_question_view(request,pk):
    question=QMODEL.Question.objects.get(id=pk)
    question.delete()
    return HttpResponseRedirect('/teacher/teacher-view-question')

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def analytics_dashboard_view(request):
    # Get filter parameters
    course_id = request.GET.get('course', '')
    date_range = int(request.GET.get('date_range', 30))
    
    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=date_range)
    
    # Get all courses for the filter dropdown
    courses = QMODEL.Course.objects.all()
    
    # Apply course filter if selected
    course_filter = {}
    if course_id:
        course_filter['exam__id'] = course_id
    
    # Get basic statistics
    total_students = SMODEL.Student.objects.count()
    active_students = SMODEL.Student.objects.filter(user__last_login__gte=start_date).count()
    total_exams = QMODEL.Course.objects.count()
    ai_exams = QMODEL.AIGeneratedExam.objects.count()
    
    # Get results in date range
    results = QMODEL.Result.objects.filter(date__gte=start_date, **course_filter)
    exams_taken = results.count()
    
    # Calculate average score
    avg_score = 0
    if exams_taken > 0:
        avg_score = round(results.aggregate(Avg('marks'))['marks__avg'] or 0)
    
    # AI usage statistics
    total_ai_queries = SMODEL.AIChatHistory.objects.filter(timestamp__gte=start_date).count()
    students_using_ai = SMODEL.AIChatHistory.objects.filter(timestamp__gte=start_date).values('student').distinct().count()
    
    # Generate date labels for charts (last n days)
    date_labels = []
    student_progress_data = []
    ai_query_volume_data = []
    
    for i in range(date_range, 0, -1):
        current_date = end_date - timedelta(days=i)
        date_labels.append(current_date.strftime('%Y-%m-%d'))
        
        # Get average score for that day
        daily_results = results.filter(date__date=current_date)
        daily_avg = daily_results.aggregate(Avg('marks'))['marks__avg'] or 0
        student_progress_data.append(round(daily_avg))
        
        # Get AI query count for that day
        daily_queries = SMODEL.AIChatHistory.objects.filter(timestamp__date=current_date).count()
        ai_query_volume_data.append(daily_queries)
    
    # Performance distribution
    performance_ranges = [0, 0, 0, 0]  # 0-25%, 26-50%, 51-75%, 76-100%
    for result in results:
        percentage = (result.marks / result.exam.total_marks) * 100
        if percentage <= 25:
            performance_ranges[0] += 1
        elif percentage <= 50:
            performance_ranges[1] += 1
        elif percentage <= 75:
            performance_ranges[2] += 1
        else:
            performance_ranges[3] += 1
    
    # Topic mastery and struggle areas
    # We'll use course names as topics for this example
    topic_labels = [course.course_name for course in courses]
    topic_mastery_data = []
    struggle_areas_data = []
    struggle_area_labels = []
    
    for course in courses:
        course_results = results.filter(exam=course)
        if course_results.exists():
            avg_score_percentage = (course_results.aggregate(Avg('marks'))['marks__avg'] or 0) / course.total_marks * 100
            topic_mastery_data.append(round(avg_score_percentage))
            
            # If below 60%, consider it a struggle area
            if avg_score_percentage < 60:
                struggle_areas_data.append(round(100 - avg_score_percentage))
                struggle_area_labels.append(course.course_name)
    
    # Question insights
    # For difficulty analysis, we need (time, success rate) pairs
    question_difficulty_data = []
    
    # Get real question attempt data from the database
    question_attempts = QMODEL.QuestionAttempt.objects.filter(timestamp__gte=start_date)
    if course_id:
        question_attempts = question_attempts.filter(question__course_id=course_id)
    
    # Group by question to calculate averages
    questions_with_attempts = question_attempts.values('question').distinct()
    for question_data in questions_with_attempts:
        question_id = question_data['question']
        q_attempts = question_attempts.filter(question_id=question_id)
        
        # Calculate success rate
        total_attempts = q_attempts.count()
        if total_attempts > 0:
            correct_attempts = q_attempts.filter(is_correct=True).count()
            success_rate = (correct_attempts / total_attempts) * 100
            
            # Calculate average time
            avg_time = q_attempts.aggregate(Avg('time_taken'))['time_taken__avg'] or 0
            
            question_difficulty_data.append({
                'x': round(avg_time),  # Average time in seconds
                'y': round(success_rate)  # Success rate percentage
            })
    
    # If no real data, provide sample data for demonstration
    if not question_difficulty_data:
        for i, course in enumerate(courses):
            question_difficulty_data.append({
                'x': 30 + (i * 10),  # Average time in seconds
                'y': 50 + (i * 5)     # Success rate percentage
            })
    
    # Time per question data
    # Get the top 10 most attempted questions
    top_questions = question_attempts.values('question').annotate(
        attempt_count=Count('id')
    ).order_by('-attempt_count')[:10]
    
    time_per_question_labels = []
    time_per_question_data = []
    
    if top_questions:
        for q_data in top_questions:
            question_id = q_data['question']
            question = QMODEL.Question.objects.get(id=question_id)
            # Truncate question text if too long
            question_text = question.question[:20] + "..." if len(question.question) > 20 else question.question
            time_per_question_labels.append(question_text)
            
            # Get average time for this question
            avg_time = question_attempts.filter(question_id=question_id).aggregate(
                Avg('time_taken')
            )['time_taken__avg'] or 0
            time_per_question_data.append(round(avg_time))
    else:
        # Sample data if no real data exists
        time_per_question_labels = [f"Q{i+1}" for i in range(min(10, QMODEL.Question.objects.count()))]
        time_per_question_data = [30, 45, 20, 60, 25, 40, 35, 50, 45, 30][:len(time_per_question_labels)]
    
    # Success rates by course
    course_names = [course.course_name for course in courses]
    success_rates_data = []
    
    for course in courses:
        course_results = results.filter(exam=course)
        if course_results.exists():
            success_rate = (course_results.aggregate(Avg('marks'))['marks__avg'] or 0) / course.total_marks * 100
        else:
            success_rate = 0
        success_rates_data.append(round(success_rate))
    
    # AI vs Manual questions comparison
    # Try to get real comparison data from AI generated exams
    ai_questions = QMODEL.Question.objects.filter(course__aigeneratedexam__isnull=False)
    manual_questions = QMODEL.Question.objects.filter(course__aigeneratedexam__isnull=True)
    
    # Get success rates and timing data for both types
    ai_success_rate = 0
    ai_avg_time = 0
    manual_success_rate = 0
    manual_avg_time = 0
    
    ai_attempts = question_attempts.filter(question__in=ai_questions)
    if ai_attempts.exists():
        ai_correct = ai_attempts.filter(is_correct=True).count()
        ai_success_rate = (ai_correct / ai_attempts.count()) * 100
        ai_avg_time = ai_attempts.aggregate(Avg('time_taken'))['time_taken__avg'] or 0
    
    manual_attempts = question_attempts.filter(question__in=manual_questions)
    if manual_attempts.exists():
        manual_correct = manual_attempts.filter(is_correct=True).count()
        manual_success_rate = (manual_correct / manual_attempts.count()) * 100
        manual_avg_time = manual_attempts.aggregate(Avg('time_taken'))['time_taken__avg'] or 0
    
    # For difficulty rating, use existing data or estimate based on time and success
    # Lower success rate and higher time could indicate higher difficulty
    ai_difficulty = 100 - ai_success_rate if ai_success_rate > 0 else 65
    manual_difficulty = 100 - manual_success_rate if manual_success_rate > 0 else 60
    
    ai_questions_data = [
        round(ai_success_rate) if ai_success_rate > 0 else 75, 
        round(ai_avg_time) if ai_avg_time > 0 else 40, 
        round(ai_difficulty)
    ]
    manual_questions_data = [
        round(manual_success_rate) if manual_success_rate > 0 else 70, 
        round(manual_avg_time) if manual_avg_time > 0 else 45, 
        round(manual_difficulty)
    ]
    
    # AI usage patterns
    usage_patterns_labels = ["Morning", "Afternoon", "Evening", "Night"]
    usage_patterns_data = [
        SMODEL.AIChatHistory.objects.filter(timestamp__hour__lt=12, timestamp__gte=start_date).count(),
        SMODEL.AIChatHistory.objects.filter(timestamp__hour__gte=12, timestamp__hour__lt=17, timestamp__gte=start_date).count(),
        SMODEL.AIChatHistory.objects.filter(timestamp__hour__gte=17, timestamp__hour__lt=21, timestamp__gte=start_date).count(),
        SMODEL.AIChatHistory.objects.filter(timestamp__hour__gte=21, timestamp__gte=start_date).count()
    ]
    
    # Popular topics in AI queries
    # In a real implementation, you would analyze the content of questions
    # Here we're using sample data
    popular_topics_labels = ["Math", "Science", "History", "Language", "Other"]
    popular_topics_data = [25, 30, 15, 20, 10]  # Sample percentages
    
    # AI impact on performance
    # Sample data - would compare scores before/after AI usage
    ai_impact_data = [65, 78]  # Before and after scores
    
    context = {
        'courses': courses,
        'selected_course': course_id,
        'date_range': date_range,
        'total_students': total_students,
        'active_students': active_students,
        'total_exams': total_exams,
        'ai_exams': ai_exams,
        'avg_score': avg_score,
        'exams_taken': exams_taken,
        'total_ai_queries': total_ai_queries,
        'students_using_ai': students_using_ai,
        
        # Chart data
        'date_labels': json.dumps(date_labels),
        'student_progress_data': json.dumps(student_progress_data),
        'performance_distribution': json.dumps(performance_ranges),
        'topic_labels': json.dumps(topic_labels),
        'topic_mastery_data': json.dumps(topic_mastery_data),
        'struggle_area_labels': json.dumps(struggle_area_labels),
        'struggle_areas_data': json.dumps(struggle_areas_data),
        'question_difficulty_data': json.dumps(question_difficulty_data),
        'time_per_question_labels': json.dumps(time_per_question_labels),
        'time_per_question_data': json.dumps(time_per_question_data),
        'course_names': json.dumps(course_names),
        'success_rates_data': json.dumps(success_rates_data),
        'ai_questions_data': json.dumps(ai_questions_data),
        'manual_questions_data': json.dumps(manual_questions_data),
        'ai_query_volume_data': json.dumps(ai_query_volume_data),
        'popular_topics_labels': json.dumps(popular_topics_labels),
        'popular_topics_data': json.dumps(popular_topics_data),
        'usage_patterns_labels': json.dumps(usage_patterns_labels),
        'usage_patterns_data': json.dumps(usage_patterns_data),
        'ai_impact_data': json.dumps(ai_impact_data),
    }
    
    return render(request, 'teacher/analytics_dashboard.html', context)

# AI Exam Generation views
@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_ai_exam_view(request):
    exam_form = forms.AIExamGenerationForm()
    context = {'exam_form': exam_form}
    
    if request.method == 'POST':
        exam_form = forms.AIExamGenerationForm(request.POST, request.FILES)
        if exam_form.is_valid():
            try:
                # Process the form data
                course = exam_form.cleaned_data['course']
                title = exam_form.cleaned_data['title']
                description = exam_form.cleaned_data['description']
                difficulty = exam_form.cleaned_data['difficulty']
                num_questions = exam_form.cleaned_data['num_questions']
                time_limit = exam_form.cleaned_data['time_limit']
                
                # Handle uploaded reference material if provided
                reference_text = None
                if 'reference_material' in request.FILES:
                    reference_file = request.FILES['reference_material']
                    file_name = reference_file.name.lower()
                    
                    # Extract text from the uploaded file based on its type
                    if file_name.endswith('.pdf'):
                        pdf_reader = PyPDF2.PdfReader(BytesIO(reference_file.read()))
                        reference_text = ""
                        for page in pdf_reader.pages:
                            reference_text += page.extract_text()
                            
                    elif file_name.endswith('.docx') or file_name.endswith('.doc'):
                        reference_text = docx2txt.process(reference_file)
                        
                    elif file_name.endswith('.txt'):
                        reference_text = reference_file.read().decode('utf-8')
                
                # Generate AI questions
                questions = get_ai_exam_questions(
                    course=course,
                    difficulty=difficulty,
                    num_questions=num_questions,
                    reference_text=reference_text
                )
                
                # Save questions and exam data in session for review
                request.session['ai_exam_data'] = {
                    'course_id': course.id,
                    'title': title,
                    'description': description,
                    'difficulty': difficulty,
                    'time_limit': time_limit,
                    'questions': questions if isinstance(questions, list) else []
                }
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': True,
                        'redirect_url': reverse('teacher-review-ai-exam')
                    })
                else:
                    return redirect('teacher-review-ai-exam')
                    
            except Exception as e:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': False,
                        'error': str(e)
                    }, status=500)
                else:
                    context['error'] = str(e)
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'error': 'Form is invalid. Please check your inputs.'
                }, status=400)
            else:
                context['error'] = "Form is invalid. Please check your inputs."
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': False,
            'error': 'Invalid request method'
        }, status=400)
    else:
        return render(request, 'teacher/teacher_ai_exam.html', context)

@login_required(login_url='teacherlogin')
@user_passes_test(is_teacher)
def teacher_review_ai_exam_view(request):
    # Get exam data from session
    exam_data = request.session.get('ai_exam_data', None)
    
    if not exam_data:
        return redirect('teacher-ai-exam')
    
    course = QMODEL.Course.objects.get(id=exam_data['course_id'])
    
    context = {
        'course': course,
        'title': exam_data['title'],
        'description': exam_data['description'],
        'difficulty': exam_data['difficulty'],
        'time_limit': exam_data['time_limit'],
        'questions': exam_data['questions']
    }
    
    if request.method == 'POST':
        action = request.POST.get('action', '')
        
        if action == 'save' or action == 'approve':
            # Save edited questions from form
            updated_questions = []
            question_count = int(request.POST.get('question_count', 0))
            
            for i in range(question_count):
                question = {
                    'question': request.POST.get(f'question_{i}', ''),
                    'option1': request.POST.get(f'option1_{i}', ''),
                    'option2': request.POST.get(f'option2_{i}', ''),
                    'option3': request.POST.get(f'option3_{i}', ''),
                    'option4': request.POST.get(f'option4_{i}', ''),
                    'answer': request.POST.get(f'answer_{i}', ''),
                    'marks': int(request.POST.get(f'marks_{i}', 1))
                }
                updated_questions.append(question)
            
            # Save AI-generated exam and its questions
            exam = save_ai_generated_exam(
                course=course,
                title=exam_data['title'],
                description=exam_data['description'],
                difficulty=exam_data['difficulty'],
                questions=updated_questions,
                time_limit=exam_data['time_limit']
            )
            
            if action == 'approve':
                # Mark as approved
                exam.approved = True
                exam.save()
            
            # Clear session data
            if 'ai_exam_data' in request.session:
                del request.session['ai_exam_data']
            
            return redirect('teacher-view-exam')
        
    return render(request, 'teacher/teacher_review_ai_exam.html', context)
