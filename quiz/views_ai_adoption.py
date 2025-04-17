from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_http_methods
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db.models import Count, Avg
from django.conf import settings
from collections import Counter
import os
import time
import logging
import json
import pandas as pd

# Import models and utilities
from .models import CSVUpload, AIAdoptionData, AIModel
from .ai_data_utils import import_from_csv, train_ai_model

# Configure logging
logger = logging.getLogger(__name__)

@login_required(login_url='adminlogin')
def ai_upload_data_view(request):
    """View for uploading and processing CSV survey data"""
    # Get upload history
    upload_history = CSVUpload.objects.all().order_by('-created_at')[:10]
    
    # Get stats
    total_records = AIAdoptionData.objects.count()
    model_count = AIModel.objects.count()
    
    # Get best accuracy from any model
    try:
        best_model = AIModel.objects.all().order_by('-accuracy').first()
        best_accuracy = f"{best_model.accuracy * 100:.1f}%" if best_model else "0%"
    except:
        best_accuracy = "0%"
    
    # Get last upload date
    try:
        last_upload = upload_history.first().created_at.strftime("%b %d, %Y") if upload_history.exists() else "None"
    except:
        last_upload = "None"
    
    context = {
        'upload_history': upload_history,
        'total_records': total_records,
        'model_count': model_count,
        'best_accuracy': best_accuracy,
        'last_upload': last_upload
    }
    
    return render(request, 'quiz/ai_upload_data.html', context)

@login_required(login_url='adminlogin')
def ai_model_metrics_view(request):
    """View for displaying model performance metrics"""
    # Get all models
    models = AIModel.objects.all().order_by('-created_date')
    
    # Get active model
    try:
        active_model = AIModel.objects.filter(is_active=True).latest('created_date')
        
        # Get feature importance data for charts
        feature_importances = active_model.feature_importance
        # Sort by importance value (descending)
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        feature_labels = [item[0] for item in sorted_importances[:10]]  # Top 10 features
        feature_values = [item[1] for item in sorted_importances[:10]]
        
        # Get ROC curve data
        roc_curve_data = active_model.roc_curve_data
        # Use the first class's ROC curve data for the chart
        first_class = list(roc_curve_data.keys())[0] if roc_curve_data else None
        roc_curve_x = roc_curve_data[first_class]['fpr'] if first_class else []
        roc_curve_y = roc_curve_data[first_class]['tpr'] if first_class else []
        roc_auc = roc_curve_data[first_class]['auc'] if first_class else 0.5
        
        # Get confusion matrix
        confusion_matrix = active_model.confusion_matrix
    except AIModel.DoesNotExist:
        active_model = None
        feature_labels = []
        feature_values = []
        roc_curve_x = []
        roc_curve_y = []
        roc_auc = 0.5
        confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    context = {
        'models': models,
        'active_model': active_model,
        'feature_importance_labels': json.dumps(feature_labels),
        'feature_importance_values': json.dumps(feature_values),
        'roc_curve_x': json.dumps(roc_curve_x),
        'roc_curve_y': json.dumps(roc_curve_y),
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix
    }
    
    return render(request, 'quiz/ai_model_metrics.html', context)

@login_required
@require_http_methods(['POST'])
def upload_csv_view(request):
    try:
        csv_file = request.FILES.get('csv_file')
        if not csv_file:
            return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
        
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'status': 'error', 'message': 'File must be a CSV'})
        
        # Create upload record
        upload_record = CSVUpload.objects.create(
            filename=csv_file.name,
            uploaded_by=request.user,
            status='processing'
        )
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Column mapping
        column_mapping = {
            'Email Address': 'email',
            'Faculty': 'faculty',
            'Level of Study': 'level_of_study',
            'How familiar are you with AI tools?': 'ai_familiarity',
            'Do you use AI tools for learning?': 'uses_ai_tools',
            'Which AI tools do you use?': 'tools_used',
            'How often do you use AI tools?': 'usage_frequency',
            'What challenges do you face using AI tools?': 'challenges',
            'Any suggestions for improving AI tools?': 'suggestions',
            'Does AI improve your learning?': 'improves_learning'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Clean and transform data
        df['email_domain'] = df['email'].apply(lambda x: x.split('@')[1] if isinstance(x, str) and '@' in x else '')
        
        # Map AI familiarity to 1-5 scale
        familiarity_mapping = {
            'Not at all familiar': 1,
            'Slightly familiar': 2,
            'Moderately familiar': 3,
            'Very familiar': 4,
            'Extremely familiar': 5
        }
        df['ai_familiarity'] = df['ai_familiarity'].map(familiarity_mapping)
        
        # Map usage frequency
        frequency_mapping = {
            'Never': 'never',
            'Rarely': 'rarely',
            'Sometimes': 'sometimes',
            'Often': 'often',
            'Very often': 'very_often'
        }
        df['usage_frequency'] = df['usage_frequency'].map(frequency_mapping)
        
        # Clean yes/no responses
        df['uses_ai_tools'] = df['uses_ai_tools'].str.lower().map({'yes': True, 'no': False})
        df['improves_learning'] = df['improves_learning'].str.lower().apply(
            lambda x: 'yes' if 'yes' in str(x) else ('no' if 'no' in str(x) else 'maybe')
        )
        
        # Save records
        records = []
        for _, row in df.iterrows():
            record = AIAdoptionData(
                email_domain=row['email_domain'],
                faculty=row['faculty'],
                level_of_study=row['level_of_study'],
                ai_familiarity=row['ai_familiarity'],
                uses_ai_tools=row['uses_ai_tools'],
                tools_used=row['tools_used'],
                usage_frequency=row['usage_frequency'],
                challenges=row['challenges'],
                suggestions=row['suggestions'],
                improves_learning=row['improves_learning'],
                upload_batch=upload_record
            )
            records.append(record)
        
        # Bulk create records
        AIAdoptionData.objects.bulk_create(records)
        
        # Update upload record
        upload_record.record_count = len(records)
        upload_record.status = 'completed'
        upload_record.insights = {
            'total_records': len(records),
            'ai_users': df['uses_ai_tools'].sum(),
            'avg_familiarity': df['ai_familiarity'].mean(),
            'improves_learning_yes': (df['improves_learning'] == 'yes').sum(),
            'improves_learning_no': (df['improves_learning'] == 'no').sum(),
            'improves_learning_maybe': (df['improves_learning'] == 'maybe').sum()
        }
        upload_record.save()
        
        # Train model if requested
        if request.POST.get('train_model') == 'true':
            result = train_ai_model(upload_record.id)
            if not result['success']:
                return JsonResponse({
                    'status': 'success',
                    'upload_id': upload_record.id,
                    'warning': f"Upload successful but model training failed: {result['error']}"
                })
        
        return JsonResponse({
            'status': 'success',
            'upload_id': upload_record.id,
            'message': 'Data uploaded and processed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in upload_csv_view: {str(e)}")
        if 'upload_record' in locals():
            upload_record.status = 'failed'
            upload_record.error_message = str(e)
            upload_record.save()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

@login_required(login_url='adminlogin')
def upload_history_view(request):
    """View for getting upload history via AJAX"""
    upload_history = CSVUpload.objects.all().order_by('-created_at')[:10]
    
    history_data = []
    for upload in upload_history:
        history_data.append({
            'id': upload.id,
            'filename': upload.original_filename,
            'record_count': upload.record_count,
            'status': upload.status,
            'created_at': upload.created_at.strftime("%b %d, %Y %H:%M")
        })
    
    return JsonResponse({
        'status': 'success',
        'history': history_data
    })

@login_required(login_url='adminlogin')
@require_POST
def train_model_view(request):
    """View for training a new model"""
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
                    'training_time': results.get('training_time', 0),
                    'algorithm': results.get('algorithm', algorithm)
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
def ai_prediction_dashboard_view(request):
    """View for displaying the AI prediction dashboard"""
    # This function uses the existing template
    return render(request, 'quiz/ai_prediction_dashboard.html') 