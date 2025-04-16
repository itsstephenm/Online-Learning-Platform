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
] 