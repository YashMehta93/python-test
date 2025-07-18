import pickle
import pandas as pd
from ..ml.data_preparer import calculate_indicators
from ..utils.utils import flatten_categorized_features,categorize_features
import xgboost as xgb

def load_model(model_file_path):
    """
    Loads the trained XGBoost model from the file path.
    """
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
    return model

def process_ohlcv_data(ohlcv_data, model_obj):
    """
    Processes the OHLCV data and calculates the required indicators.
    """
    # Convert OHLCV JSON to a Pandas DataFrame
    df = pd.DataFrame(ohlcv_data)

    feature_columns = flatten_categorized_features(model_obj.selected_features)

    # Ensure data is properly formatted
    if not set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
        raise ValueError("OHLCV data must contain open, high, low, close, and volume fields.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert timestamp to datetime
    df.set_index("timestamp", inplace=True) 

    # Calculate indicators based on the model requirements
    df, full_features = calculate_indicators(df=df, selected_features=feature_columns, counts=model_obj.lagging_count, scaling_method='standardization')

    df.reset_index(inplace=True)

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure DataFrame is sorted by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Handle missing values (forward fill) and drop any remaining NaNs
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    df.set_index('timestamp', inplace=True)

    # Check for missing columns and add if necessary
    missing_cols = set(full_features) - set(df.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        df = df.reindex(columns=df.columns.tolist() + list(missing_cols), fill_value=0)

    return df

def predict_close_price(model, prepared_data):
    """
    Predicts the close price using the trained model.
    """
    # Perform the prediction
    dmatrix = xgb.DMatrix(prepared_data)
    predictions = model.predict(dmatrix)
    return predictions[-1]  # Return the last prediction
