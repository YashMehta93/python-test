from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from ..models import XGBoostModels
from ..services.xgboost_prediction_service import load_model, process_ohlcv_data, predict_close_price

class XGBoostPredictionAPIView(APIView):
    """
    API for predicting the stock close price using an XGBoost model.
    """

    def post(self, request, model_id):
        """
        Takes a model ID and OHLCV data to predict the close price.
        """
        # Fetch the model from the database
        xgboost_model = get_object_or_404(XGBoostModels, id=model_id)

        # Validate and process the input data
        ohlcv_data = request.data.get("ohlcv_data")
        if not ohlcv_data:
            return Response({"error": "OHLCV data is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Process the OHLCV data
            processed_data = process_ohlcv_data(ohlcv_data, xgboost_model)

            # Load the trained model
            model = load_model(xgboost_model.model_file)

            # Make the prediction
            predicted_close = predict_close_price(model, processed_data)

            # Return the predicted close price
            return Response({"predicted_close_price": predicted_close}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
