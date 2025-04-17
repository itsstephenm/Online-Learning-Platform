from django.urls import path
from . import views

app_name = 'quiz'  # Add namespace

urlpatterns = [
    # AI Dashboard URL - main entry point
    path('dashboard/', views.ai_dashboard, name='ai_dashboard'),
    path('ai_adoption_dashboard/', views.ai_adoption_dashboard_view, name='ai_adoption_dashboard'),  # New dashboard URL
    
    # AI Prediction System URLs
    path('ai/upload/', views.upload_csv, name='upload_csv'),
    path('ai/train/', views.train_ai_model, name='train_ai_model'),
    path('ai/predict/', views.predict_adoption_level, name='predict_adoption'),
    path('ai/prediction/<int:prediction_id>/', views.prediction_result, name='prediction_result'),
    path('ai/insights/', views.insights_view, name='insights_view'),
    path('ai/query/', views.query_ai, name='query_ai'),
    path('ai/make-prediction/', views.make_prediction_view, name='make_new_prediction'),  # Added URL for make_new_prediction
    
    # Training data management
    path('ai/upload-training-data/', views.upload_training_data, name='upload_training_data'),
    path('ai/data/<int:data_id>/', views.ai_data_detail, name='ai_data_detail'),
    path('ai/model/<int:model_id>/', views.ai_model_detail, name='ai_model_detail'),
    
    # Predictions
    path('ai/predictions/', views.all_predictions, name='all_predictions'),
    
    # API endpoints
    path('api/ai/activate-model/<int:model_id>/', views.api_activate_model, name='api_activate_model'),
    path('api/ai/delete-model/<int:model_id>/', views.api_delete_model, name='api_delete_model'),
    path('api/ai/delete-training-data/<int:data_id>/', views.api_delete_training_data, name='api_delete_training_data'),
    path('nl_query/', views.nl_query_view, name='nl_query'),
    
    # New API endpoints for AI dashboard
    path('get-nl-query/', views.get_nl_query_view, name='get_nl_query'),
    path('run-sql-query/', views.run_sql_query_view, name='run_sql_query'),
    
    # AJAX endpoints for the new dashboard
    path('api/upload-csv/', views.ajax_upload_csv, name='ajax_upload_csv'),
    path('api/train-model/', views.ajax_train_model, name='ajax_train_model'),
    
    # Add a new URL pattern for teacher access to prediction dashboard
    path('teacher/ai-prediction-dashboard/', views.teacher_ai_prediction_dashboard_view, name='teacher_ai_prediction_dashboard'),
] 