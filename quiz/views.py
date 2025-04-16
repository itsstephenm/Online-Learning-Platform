from django.shortcuts import render, redirect, reverse, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.core.mail import send_mail
from django.conf import settings
from datetime import date, timedelta
from django.db.models import Sum, Q
from django.contrib.auth.models import Group, User
from . import forms, models
from teacher import models as TMODEL
from student import models as SMODEL
from teacher import forms as TFORM
from student import forms as SFORM
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .ai_utils import import_from_csv, get_chart_data, process_nl_query, predict_adoption_level, process_csv_data, train_model, make_prediction, generate_insights_from_data, get_data_counts, get_prediction_details
from .models import AIAdoptionData, AIPrediction, NLQuery, InsightTopic, AIInsight, AIModel
from django.contrib.admin.views.decorators import staff_member_required
import json
import pandas as pd
import os
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import tempfile

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
    context = {
        'total_teacher': TMODEL.Teacher.objects.filter(status=True).count(),
        'pending_teacher': TMODEL.Teacher.objects.filter(status=False).count(),
        'salary': TMODEL.Teacher.objects.filter(status=True).aggregate(Sum('salary'))['salary__sum'],
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
    context = {
        'total_student': SMODEL.Student.objects.count(),
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
    return render(request, 'quiz/admin_course.html')

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
    return render(request, 'quiz/admin_question.html')

@login_required(login_url='adminlogin')
def admin_add_question_view(request):
    questionForm = forms.QuestionForm()
    if request.method == 'POST':
        questionForm = forms.QuestionForm(request.POST)
        if questionForm.is_valid():
            question = questionForm.save(commit=False)
            course = models.Course.objects.get(id=request.POST.get('courseID'))
            question.course = course
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
    # Temporary remove the user_passes_test for debugging
    dict = {
        'total_student': SMODEL.Student.objects.all().count(),
        'total_teacher': TMODEL.Teacher.objects.all().count(),
        'total_course': models.Course.objects.count(),
        'total_question': models.Question.objects.count(),
    }
    return render(request, 'quiz/ai_prediction_dashboard.html', context=dict)

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
    """
    Dashboard for AI prediction features
    """
    # Get stats
    models = AIModel.objects.all().order_by('-created_date')
    active_model = models.filter(is_active=True).first()
    training_data = AIAdoptionData.objects.all().order_by('-upload_date')
    recent_predictions = AIPrediction.objects.all().order_by('-prediction_date')[:10]
    featured_insights = AIInsight.objects.filter(is_featured=True).order_by('-created_date')[:4]
    
    context = {
        'models': models,
        'active_model': active_model,
        'training_data': training_data,
        'recent_predictions': recent_predictions,
        'featured_insights': featured_insights,
        'total_models': models.count(),
        'total_predictions': AIPrediction.objects.count(),
        'total_training_files': training_data.count(),
    }
    
    return render(request, 'quiz/ai_dashboard.html', context)

@login_required
def upload_training_data(request):
    """
    Upload CSV file for model training
    """
    if request.method == 'POST' and request.FILES.get('training_file'):
        file = request.FILES['training_file']
        
        # Save the file
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # Create AIAdoptionData object
        adoption_data = AIAdoptionData.objects.create(
            file_name=file.name,
            uploaded_by=request.user,
            is_processed=False
        )
        
        # Process the file
        success, message = process_csv_data(file_path, adoption_data.pk)
        
        if success:
            return redirect('ai_data_detail', data_id=adoption_data.pk)
        else:
            return render(request, 'quiz/upload_training_data.html', {'error': message})
    
    return render(request, 'quiz/upload_training_data.html')

@login_required
def ai_data_detail(request, data_id):
    """
    View details of uploaded training data
    """
    data = get_object_or_404(AIAdoptionData, pk=data_id)
    
    if request.method == 'POST' and 'train_model' in request.POST:
        # Train model
        success, message = train_model(data_id)
        
        if success:
            return redirect('ai_model_detail', model_id=message)
        else:
            return render(request, 'quiz/ai_data_detail.html', {'data': data, 'error': message})
    
    context = {
        'data': data,
        'can_train': data.is_processed and not data.processing_errors
    }
    
    return render(request, 'quiz/ai_data_detail.html', context)

@login_required
def ai_model_detail(request, model_id):
    """
    View details of a trained model
    """
    model = get_object_or_404(AIModel, pk=model_id)
    
    # Get insights for this model
    insights = AIInsight.objects.filter(source_model=model).order_by('-created_date')
    
    context = {
        'model': model,
        'insights': insights,
        'training_data': model.training_data,
        'predictions_count': AIPrediction.objects.filter(model=model).count()
    }
    
    return render(request, 'quiz/ai_model_detail.html', context)

@login_required
def make_new_prediction(request):
    """
    Make a new prediction
    """
    # Get active model
    active_model = AIModel.objects.filter(is_active=True).first()
    
    if not active_model:
        return render(request, 'quiz/make_prediction.html', 
                      {'error': 'No active model found. Please train a model first.'})
    
    if request.method == 'POST':
        # Get form data
        input_data = {
            'level_of_study': request.POST.get('level_of_study'),
            'faculty': request.POST.get('faculty'),
            'ai_familiarity': float(request.POST.get('ai_familiarity', 0)),
            'uses_ai_tools': request.POST.get('uses_ai_tools'),
            'tools_used': request.POST.get('tools_used', ''),
            'usage_frequency': request.POST.get('usage_frequency'),
            'improves_learning': request.POST.get('improves_learning'),
            'challenges': request.POST.get('challenges', '')
        }
        
        # Make prediction
        success, result = make_prediction(request.user.id, input_data)
        
        if success:
            return redirect('prediction_result', prediction_id=result)
        else:
            return render(request, 'quiz/make_prediction.html', {'error': result})
    
    context = {
        'active_model': active_model
    }
    
    return render(request, 'quiz/make_prediction.html', context)

@login_required
def prediction_result(request, prediction_id):
    """
    View prediction result
    """
    success, result = get_prediction_details(prediction_id)
    
    if not success:
        return render(request, 'quiz/prediction_result.html', {'error': result})
    
    return render(request, 'quiz/prediction_result.html', {'prediction': result})

@login_required
def all_predictions(request):
    """
    View all predictions
    """
    predictions = AIPrediction.objects.all().order_by('-prediction_date')
    
    context = {
        'predictions': predictions
    }
    
    return render(request, 'quiz/all_predictions.html', context)

@login_required
def all_insights(request):
    """
    View all insights
    """
    insights = AIInsight.objects.all().order_by('-created_date')
    
    context = {
        'insights': insights
    }
    
    return render(request, 'quiz/all_insights.html', context)

# API endpoints for AJAX
@csrf_exempt
@login_required
def api_activate_model(request, model_id):
    """
    Activate a model
    """
    if request.method == 'POST':
        try:
            # Get the model
            model = AIModel.objects.get(pk=model_id)
            
            # Deactivate all models
            AIModel.objects.all().update(is_active=False)
            
            # Activate this model
            model.is_active = True
            model.save()
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
@login_required
def api_delete_model(request, model_id):
    """
    Delete a model
    """
    if request.method == 'POST':
        try:
            # Get the model
            model = AIModel.objects.get(pk=model_id)
            
            # Delete model file
            if os.path.exists(model.model_file_path):
                os.remove(model.model_file_path)
            
            # Delete model object
            model.delete()
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
@login_required
def api_delete_training_data(request, data_id):
    """
    Delete training data
    """
    if request.method == 'POST':
        try:
            # Get the data
            data = AIAdoptionData.objects.get(pk=data_id)
            
            # Check if there are models using this data
            if AIModel.objects.filter(training_data=data).exists():
                return JsonResponse({
                    'success': False, 
                    'error': 'Cannot delete training data that is being used by models'
                })
            
            # Delete file
            file_path = os.path.join(settings.MEDIA_ROOT, data.file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete data object
            data.delete()
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
@login_required
def query_ai(request):
    """
    API endpoint to handle natural language queries about the AI data
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query_text = data.get('query', '')
            
            if not query_text:
                return JsonResponse({'error': 'Query is required'}, status=400)
            
            # Process the query
            result = process_nl_query(query_text, request.user)
            
            # Save the query to the database
            nl_query = NLQuery(
                user=request.user,
                query=query_text,
                processed_query=query_text,  # In a real system, this might be different
                response=result.get('response', ''),
                response_type=result.get('response_type', 'text'),
                chart_data=json.dumps(result.get('chart_data', [])) if result.get('chart_data') else None
            )
            nl_query.save()
            
            # Return the result
            return JsonResponse({
                'success': True,
                'response': result.get('response', ''),
                'response_type': result.get('response_type', 'text'),
                'chart_data': result.get('chart_data'),
                'chart_type': result.get('chart_type'),
                'chart_title': result.get('chart_title')
            })
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)



    
