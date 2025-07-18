from rest_framework import serializers
from ..models import XGBoostModels

class XGBoostModelsSerializer(serializers.ModelSerializer):
    class Meta:
        model = XGBoostModels
        fields = [
            'id',
            'name',
            'description',
            'instrument_key',
            'interval',
            'from_date',
            'to_date',
            'prediction_step',
            'train_split',
            'eta',
            'subsample',
            'colsample_bytree',
            'seed',
            'model_file',
            'model_structure_file',
            'selected_features',
            'training_mse',
            'training_mae',
            'created_at',
            'modified_at'
        ]
        read_only_fields = ['id', 'model_file', 'model_structure_file', 'training_mse', 'training_mae', 'created_at', 'modified_at']
