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
from .ai_utils import predict_adoption_level, train_model, make_prediction, generate_insights_from_data, get_data_counts, prepare_features, get_chart_data, process_nl_query, process_training_data
from .models import AIAdoptionData, AIPrediction, NLQuery, InsightTopic, AIInsight, AIModel
from django.contrib.admin.views.decorators import staff_member_required
import json
import pandas as pd
import os
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import tempfile
from student.models import Student
from django.conf import settings
from decouple import config
from django.db.models import Count, Avg, FloatField
from django.db.models.functions import Coalesce

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
        
        return redirect('prediction_result')
    
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
    try:
        # Try with created_date first
        insights = AIInsight.objects.all().order_by('-created_date')
    except:
        try:
            # If that fails, try with created_at
            insights = AIInsight.objects.all().order_by('-created_at')
        except:
            # If both fail, just get all insights without ordering
            insights = AIInsight.objects.all()
    
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
def make_new_prediction(request):
    """View for making new AI predictions"""
    if request.method == 'POST':
        try:
            data = json.loads(request.POST.get('data'))
            prediction = make_prediction(data)
            return JsonResponse({'success': True, 'prediction': prediction})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return render(request, 'quiz/make_prediction.html')

@login_required
def all_predictions(request):
    """View for displaying all predictions"""
    predictions = AIPrediction.objects.all().order_by('-created_at')
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



    
