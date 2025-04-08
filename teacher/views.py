from django.shortcuts import render,redirect,reverse
from . import forms,models
from django.db.models import Sum
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
