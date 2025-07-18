from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .serializers import XGBoostModelsSerializer
from ..tasks import async_train_model  # Import the Celery task
from ..services.xgboost_model_service import XGBoostModelService
from ..models import XGBoostModels


class XGBoostModelsView(APIView):
    """
    API for handling XGBoost models. Supports CRUD operations and model training.
    """

    def get(self, request, pk=None):
        """
        Retrieve one or all XGBoost models.
        """
        if pk:
            stock_model = get_object_or_404(XGBoostModelService.get_model_by_id, pk=pk)
            serializer = XGBoostModelsSerializer(stock_model)
        else:
            stock_models = XGBoostModelService.get_all_models()
            serializer = XGBoostModelsSerializer(stock_models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        """
        Create and start training a new XGBoost model.
        """
        serializer = XGBoostModelsSerializer(data=request.data)
        if serializer.is_valid():
            validated_data = serializer.validated_data

            # Save the model entry in the database with initial data
            created_model = XGBoostModelService.save_model_info(validated_data)

            # Start the async task
            async_train_model.delay(
                model_id=created_model.id,
                instrument_key=validated_data['instrument_key'],
                interval=validated_data['interval'],
                from_date=str(validated_data['from_date']),
                to_date=str(validated_data['to_date']),
                prediction_step=validated_data['prediction_step'],
                train_split=validated_data['train_split'],
                max_depth=validated_data['max_depth'],
                eta=validated_data['eta'],
                subsample=validated_data['subsample'],
                colsample_bytree=validated_data['colsample_bytree'],
                seed=validated_data['seed'],
                name=validated_data['name'],
                lagging_count=validated_data['lagging_count'],
                num_boost_round=validated_data['num_boost_round'],
                early_stopping_rounds=validated_data['early_stopping_rounds'],
                verbose_eval=validated_data['verbose_eval'],
                selected_features=validated_data['selected_features']
            )

            return Response(
                {"message": "Model training started", "model_id": created_model.id},
                status=status.HTTP_202_ACCEPTED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        """
        Update an existing XGBoost model.
        """
        stock_model = get_object_or_404(XGBoostModelService.get_model_by_id, pk=pk)
        serializer = XGBoostModelsSerializer(stock_model, data=request.data, partial=True)
        if serializer.is_valid():
            updated_model = XGBoostModelService.update_model_info(pk, serializer.validated_data)
            return Response(XGBoostModelsSerializer(updated_model).data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Delete an existing XGBoost model.
        """
        XGBoostModelService.delete_model_by_id(pk)
        return Response(status=status.HTTP_204_NO_CONTENT)
