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
    """View for processing CSV uploads with improved error handling"""
    try:
        # Start timing for performance tracking
        start_time = time.time()
        
        # Check if this is a standard form submission or AJAX
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'
        
        # Debug logging
        logger.info(f"CSV upload request received. AJAX: {is_ajax}, POST keys: {request.POST.keys()}, FILES keys: {request.FILES.keys()}")
        
        # Get the CSV file from the request
        csv_file = request.FILES.get('csv_file')
        
        # Validate file exists
        if not csv_file:
            error_msg = 'No file uploaded. Please select a CSV file.'
            logger.error(f"Upload failed: {error_msg}")
            
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': error_msg})
            else:
                messages.error(request, error_msg)
                return redirect('ai_upload_data')
        
        # Validate file type
        if not csv_file.name.endswith('.csv'):
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': 'File must be a CSV'})
            else:
                messages.error(request, 'File must be a CSV')
                return redirect('ai_upload_data')
        
        # Create upload record
        upload_record = CSVUpload.objects.create(
            filename=csv_file.name,
            original_filename=csv_file.name,
            uploaded_by=request.user,
            status='processing'
        )
        
        # Log the upload
        logger.info(f"Processing CSV upload: {csv_file.name} (ID: {upload_record.id})")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Validate file has data
            if len(df) == 0:
                upload_record.status = 'failed'
                upload_record.error_message = 'CSV file is empty'
                upload_record.save()
                
                if is_ajax:
                    return JsonResponse({'status': 'error', 'message': 'The uploaded CSV file is empty.'})
                else:
                    messages.error(request, 'The uploaded CSV file is empty.')
                    return redirect('ai_upload_data')
            
            # Process the column mapping
            column_mapping = {
                'email_domain': 'email_domain',
                'faculty': 'faculty',
                'level_of_study': 'level_of_study',
                'ai_familiarity': 'ai_familiarity',
                'uses_ai_tools': 'uses_ai_tools',
                'tools_used': 'tools_used',
                'usage_frequency': 'usage_frequency',
                'challenges': 'challenges',
                'suggestions': 'suggestions',
                'improves_learning': 'improves_learning',
                # Add mappings for common variations
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
            
            # Check if we need to rename columns
            rename_cols = {}
            for col_name, mapped_name in column_mapping.items():
                if col_name in df.columns and col_name != mapped_name:
                    rename_cols[col_name] = mapped_name
            
            # Rename columns if needed
            if rename_cols:
                df = df.rename(columns=rename_cols)
            
            # Process clean_data parameter
            clean_data = request.POST.get('clean_data', 'false').lower() in ('true', 'on', 'yes', '1')
            
            # Clean and transform data if requested
            if clean_data:
                # Handle email_domain if email column exists and email_domain doesn't
                if 'email' in df.columns and 'email_domain' not in df.columns:
                    df['email_domain'] = df['email'].apply(
                        lambda x: x.split('@')[1] if isinstance(x, str) and '@' in x else ''
                    )
                
                # Map AI familiarity to 1-5 scale if needed
                if 'ai_familiarity' in df.columns and df['ai_familiarity'].dtype == 'object':
                    familiarity_mapping = {
                        'Not at all familiar': 1,
                        'Slightly familiar': 2,
                        'Moderately familiar': 3,
                        'Very familiar': 4,
                        'Extremely familiar': 5
                    }
                    df['ai_familiarity'] = df['ai_familiarity'].map(familiarity_mapping).fillna(3)
                
                # Map usage frequency if needed
                if 'usage_frequency' in df.columns and df['usage_frequency'].dtype == 'object':
                    frequency_mapping = {
                        'Never': 'never',
                        'Rarely': 'rarely',
                        'Monthly': 'monthly',
                        'Weekly': 'weekly',
                        'Daily': 'daily',
                        'Sometimes': 'sometimes',
                        'Often': 'often',
                        'Very often': 'very_often'
                    }
                    df['usage_frequency'] = df['usage_frequency'].str.lower().map(frequency_mapping).fillna('never')
                
                # Clean yes/no responses if needed
                if 'uses_ai_tools' in df.columns:
                    df['uses_ai_tools'] = df['uses_ai_tools'].astype(str).str.lower().apply(
                        lambda x: True if x in ('yes', 'true', 'y', '1', 'on') else False
                    )
                
                if 'improves_learning' in df.columns:
                    df['improves_learning'] = df['improves_learning'].astype(str).str.lower().apply(
                        lambda x: 'yes' if x in ('yes', 'true', 'y', '1', 'on') 
                            else ('no' if x in ('no', 'false', 'n', '0', 'off') else 'maybe')
                    )
            
            # Save records
            records = []
            for _, row in df.iterrows():
                # Use get() to handle missing columns
                record = AIAdoptionData(
                    email_domain=row.get('email_domain', ''),
                    faculty=row.get('faculty', ''),
                    level_of_study=row.get('level_of_study', ''),
                    ai_familiarity=row.get('ai_familiarity', 3),
                    uses_ai_tools=row.get('uses_ai_tools', False),
                    tools_used=row.get('tools_used', ''),
                    usage_frequency=row.get('usage_frequency', 'never'),
                    challenges=row.get('challenges', ''),
                    suggestions=row.get('suggestions', ''),
                    improves_learning=row.get('improves_learning', 'maybe'),
                    upload_batch=upload_record
                )
                records.append(record)
            
            # Bulk create records
            AIAdoptionData.objects.bulk_create(records)
            
            # Update upload record
            upload_record.record_count = len(records)
            upload_record.status = 'completed'
            
            # Create insights data
            ai_users = len([r for r in records if r.uses_ai_tools])
            avg_familiarity = sum(r.ai_familiarity for r in records) / len(records) if records else 0
            
            upload_record.insights = {
                'total_records': len(records),
                'ai_users': ai_users,
                'avg_familiarity': avg_familiarity,
                'improves_learning_yes': len([r for r in records if r.improves_learning == 'yes']),
                'improves_learning_no': len([r for r in records if r.improves_learning == 'no']),
                'improves_learning_maybe': len([r for r in records if r.improves_learning == 'maybe'])
            }
            upload_record.save()
            
            # Extract useful insights for display
            insights = [
                f"Total records: {len(records)}",
                f"AI tool users: {ai_users} ({ai_users/len(records)*100:.1f}%)" if records else "No records",
                f"Average AI familiarity: {avg_familiarity:.1f}/5.0" if records else "No data"
            ]
            
            # Train model if requested
            train_model = request.POST.get('train_model', 'false').lower() in ('true', 'on', 'yes', '1')
            
            model_trained = False
            model_accuracy = 0
            
            if train_model:
                try:
                    result = train_ai_model(upload_record.id)
                    if result['success']:
                        model_trained = True
                        model_accuracy = result['accuracy']
                        logger.info(f"Successfully trained model for upload {upload_record.id}")
                        insights.append(f"Model trained with accuracy: {model_accuracy:.2f}")
                    else:
                        error_msg = f"Upload successful but model training failed: {result.get('error')}"
                        logger.warning(f"Training failed for upload {upload_record.id}: {result.get('error')}")
                        if is_ajax:
                            # Continue processing even if training failed
                            pass
                        else:
                            messages.warning(request, error_msg)
                except Exception as e:
                    logger.error(f"Error training model: {str(e)}", exc_info=True)
                    if not is_ajax:
                        messages.warning(request, f"Upload successful but model training failed: {str(e)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get updated stats
            total_records = AIAdoptionData.objects.count()
            model_count = AIModel.objects.count()
            
            try:
                best_model = AIModel.objects.all().order_by('-accuracy').first()
                best_accuracy = f"{best_model.accuracy * 100:.1f}%" if best_model else "0%"
            except:
                best_accuracy = "0%"
            
            try:
                last_upload = CSVUpload.objects.all().order_by('-created_at').first()
                last_upload_date = last_upload.created_at.strftime("%b %d, %Y") if last_upload else "None"
            except:
                last_upload_date = "None"
            
            # Return appropriate response based on request type
            if is_ajax:
                return JsonResponse({
                    'status': 'success',
                    'upload_id': upload_record.id,
                    'message': 'Data uploaded and processed successfully',
                    'record_count': len(records),
                    'processing_time': processing_time,
                    'model_trained': model_trained,
                    'accuracy': model_accuracy,
                    'insights': insights,
                    'total_records': total_records,
                    'model_count': model_count,
                    'best_accuracy': best_accuracy,
                    'last_upload': last_upload_date
                })
            else:
                messages.success(request, 'Data uploaded and processed successfully')
                return redirect('ai_upload_data')
                
        except pd.errors.EmptyDataError:
            upload_record.status = 'failed'
            upload_record.error_message = 'CSV file is empty'
            upload_record.save()
            logger.error(f"Empty CSV file uploaded (ID: {upload_record.id})")
            
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': 'The uploaded CSV file is empty.'})
            else:
                messages.error(request, 'The uploaded CSV file is empty.')
                return redirect('ai_upload_data')
                
        except pd.errors.ParserError as e:
            upload_record.status = 'failed'
            upload_record.error_message = f'CSV parsing error: {str(e)}'
            upload_record.save()
            logger.error(f"CSV parsing error (ID: {upload_record.id}): {str(e)}")
            
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': f'Error parsing CSV file: {str(e)}'})
            else:
                messages.error(request, f'Error parsing CSV file: {str(e)}')
                return redirect('ai_upload_data')
            
    except Exception as e:
        # Log the error
        logger.error(f"Error in upload_csv_view: {str(e)}", exc_info=True)
        
        # Update upload record if it was created
        if 'upload_record' in locals():
            upload_record.status = 'failed'
            upload_record.error_message = str(e)
            upload_record.save()
        
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'
        if is_ajax:
            return JsonResponse({
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}'
            })
        else:
            messages.error(request, f'Error: {str(e)}')
            return redirect('ai_upload_data')

@login_required(login_url='adminlogin')
def upload_history_view(request):
    """View for getting upload history via AJAX"""
    upload_history = CSVUpload.objects.all().order_by('-created_at')[:10]
    
    # Prepare history data
    history_data = []
    for upload in upload_history:
        history_data.append({
            'id': upload.id,
            'filename': upload.original_filename,
            'record_count': upload.record_count,
            'status': upload.status,
            'created_at': upload.created_at.strftime("%b %d, %Y %H:%M")
        })
    
    # Get stats for response
    total_records = AIAdoptionData.objects.count()
    model_count = AIModel.objects.count()
    
    try:
        best_model = AIModel.objects.all().order_by('-accuracy').first()
        best_accuracy = f"{best_model.accuracy * 100:.1f}%" if best_model else "0%"
    except:
        best_accuracy = "0%"
    
    try:
        last_upload = upload_history.first()
        last_upload_date = last_upload.created_at.strftime("%b %d, %Y") if last_upload else "None"
    except:
        last_upload_date = "None"
    
    # Return combined response
    return JsonResponse({
        'status': 'success',
        'history': history_data,
        'stats': {
            'total_records': total_records,
            'model_count': model_count,
            'best_accuracy': best_accuracy,
            'last_upload': last_upload_date
        }
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
        
        # Get updated stats
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
            last_upload = CSVUpload.objects.all().order_by('-created_at').first()
            last_upload_date = last_upload.created_at.strftime("%b %d, %Y") if last_upload else "None"
        except:
            last_upload_date = "None"
            
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success',
                'total_records': total_records,
                'model_count': model_count,
                'best_accuracy': best_accuracy,
                'last_upload': last_upload_date
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