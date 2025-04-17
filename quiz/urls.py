from django.urls import path
from . import views

app_name = 'quiz'  # Add namespace

urlpatterns = [
    # AI Dashboard URL - main entry point
    path('dashboard/', views.ai_dashboard, name='ai_dashboard'),
    path('ai_adoption_dashboard/', views.ai_adoption_dashboard_view, name='ai_adoption_dashboard'),  # New dashboard URL
    
    # AI Prediction System URLs
    path('ai/upload/', views.upload_csv_ai, name='upload_csv_ai'),
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
    
    # AI Adoption System URLs
    path('admin/ai-adoption/upload/', views.ai_upload_data_view, name='ai_upload_data'),
    path('admin/ai-adoption/view-data/', views.ai_view_data_list_view, name='ai_view_data_list'),
    path('admin/ai-adoption/view-data/<int:dataset_id>/', views.ai_view_data_detail_view, name='ai_view_data_detail'),
    path('admin/ai-adoption/model-metrics/', views.ai_model_metrics_view, name='ai_model_metrics'),
    path('admin/ai-adoption/predict/', views.ai_prediction_form_view, name='ai_prediction_form'),
    
    # AJAX endpoints for AI adoption system
    path('admin/ai-adoption/upload-csv/', views.upload_csv, name='upload_csv'),
    path('admin/ai-adoption/load-more-data/<int:dataset_id>/', views.load_more_data_view, name='load_more_data_view'),
    path('admin/ai-adoption/generate-chart/<int:dataset_id>/', views.generate_chart_view, name='generate_chart_view'),
    path('admin/ai-adoption/generate-insights/<int:dataset_id>/', views.generate_insights_view, name='generate_insights_view'),
    path('admin/ai-adoption/export-csv/<int:dataset_id>/', views.export_csv_view, name='export_csv_view'),
    path('admin/ai-adoption/train-model/', views.train_model_view, name='train_model_view'),
    path('admin/ai-adoption/activate-model/<int:model_id>/', views.activate_model_view, name='activate_model_view'),
    path('admin/ai-adoption/delete-model/<int:model_id>/', views.delete_model_view, name='delete_model_view'),
    path('admin/ai-adoption/predict-api/', views.ai_predict_view, name='ai_predict'),
    path('admin/ai-adoption/delete-dataset/<int:dataset_id>/', views.delete_dataset_view, name='delete_dataset_view'),
    
    # Updated URL patterns with correct prefix
    path('admin/ai-adoption/upload/', views.ai_upload_data_view, name='ai_upload_data'),
    path('admin/ai-adoption/upload-csv/', views.upload_csv, name='upload_csv'),
    path('admin/ai-adoption/upload-history/', views.upload_history_view, name='upload_history'),
    path('admin/ai-adoption/delete-upload/<int:upload_id>/', views.delete_upload, name='delete_upload'),
] 