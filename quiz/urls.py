from django.urls import path
from . import views

urlpatterns = [
    # Existing URL patterns
    # AI Prediction System URLs
    path('ai/dashboard/', views.ai_dashboard, name='ai_dashboard'),
    path('ai/upload/', views.upload_csv, name='upload_csv'),
    path('ai/train/', views.train_ai_model, name='train_ai_model'),
    path('ai/predict/', views.predict_adoption, name='predict_adoption'),
    path('ai/prediction/<int:prediction_id>/', views.prediction_result, name='prediction_result'),
    path('ai/insights/', views.insights_view, name='insights_view'),
    path('ai/insights/<int:topic_id>/', views.insights_view, name='insights_topic_view'),
    path('ai/query/', views.query_ai, name='query_ai'),

    # AI Prediction URLs
    path('ai/dashboard/', views.ai_dashboard, name='ai_dashboard'),
    path('ai/upload-training-data/', views.upload_training_data, name='upload_training_data'),
    path('ai/data/<int:data_id>/', views.ai_data_detail, name='ai_data_detail'),
    path('ai/model/<int:model_id>/', views.ai_model_detail, name='ai_model_detail'),
    path('ai/make-prediction/', views.make_new_prediction, name='make_new_prediction'),
    path('ai/prediction/<int:prediction_id>/', views.prediction_result, name='prediction_result'),
    path('ai/predictions/', views.all_predictions, name='all_predictions'),
    path('ai/insights/', views.all_insights, name='all_insights'),
    
    # AI API endpoints
    path('api/ai/activate-model/<int:model_id>/', views.api_activate_model, name='api_activate_model'),
    path('api/ai/delete-model/<int:model_id>/', views.api_delete_model, name='api_delete_model'),
    path('api/ai/delete-training-data/<int:data_id>/', views.api_delete_training_data, name='api_delete_training_data'),
    path('nl_query/', views.nl_query_view, name='nl_query'),
] 