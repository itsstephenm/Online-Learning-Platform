from django.urls import path
from . import views

urlpatterns = [
    # Data upload and management
    path('admin/ai-adoption/upload-data/', views.ai_upload_data_view, name='ai_upload_data'),
    path('admin/ai-adoption/upload-csv/', views.upload_csv_view, name='upload_csv_view'),
    path('admin/ai-adoption/upload-history/', views.upload_history_view, name='upload_history'),
    path('admin/ai-adoption/data-detail/<int:data_id>/', views.ai_data_detail_view, name='ai_data_detail'),
    path('admin/ai-adoption/delete-data/<int:upload_id>/', views.delete_upload_view, name='delete_upload'),
    
    # Model management and metrics
    path('admin/ai-adoption/model-metrics/', views.ai_model_metrics_view, name='ai_model_metrics'),
    path('admin/ai-adoption/train-model/', views.train_model_view, name='train_model'),
    path('admin/ai-adoption/activate-model/<int:model_id>/', views.activate_model_view, name='activate_model'),
    path('admin/ai-adoption/delete-model/<int:model_id>/', views.delete_model_view, name='delete_model'),
    
    # Dashboard and prediction
    path('admin/ai-adoption/dashboard/', views.ai_prediction_dashboard_view, name='ai_prediction_dashboard'),
] 