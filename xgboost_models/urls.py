from django.urls import path
from .apis.xgboost_model_view import XGBoostModelsView
from .apis.xgboost_prediction_view import XGBoostPredictionAPIView

urlpatterns = [
    # CRUD operations for XGBoost models
    path('xgboost-models/', XGBoostModelsView.as_view(), name='xgboost_models'),
    path('xgboost-models/<int:pk>/', XGBoostModelsView.as_view(), name='xgboost_model_detail'),

    # Prediction API for a specific model
    path('xgboost-models/predict/<int:model_id>/', XGBoostPredictionAPIView.as_view(), name='xgboost_predict'),
]
