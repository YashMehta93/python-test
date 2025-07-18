import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from ..repository.ohlcv_repo import fetch_data
from .data_preparer import calculate_indicators, get_full_features
from ..models import XGBoostModels
from ..utils.utils import categorize_features, flatten_categorized_features



def train_model(
        instrument_key,
        interval,
        from_date,
        to_date,
        prediction_step,
        train_split,
        max_depth,
        eta,
        subsample,
        colsample_bytree,
        seed,
        name,
        lagging_count,
        num_boost_round,
        early_stopping_rounds,
        verbose_eval,
        selected_features):
    models_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'xgboost_models')
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Prepare selected features for model training
    if selected_features:
        feature_columns = flatten_categorized_features(selected_features)
    else:
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'sma_10', 'ema_12', 'ema_26', 'rsi_14', 'macd', 'macd_signal',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'atr_14', 'obv', 'force_index', 'adx_14',
            'aroon_up', 'aroon_down', 'parabolic_sar', 'dmi_plus', 'dmi_minus', 'z_score', 'kurtosis_10', 'skewness_10',
            'sharpe_ratio', 'price_above_resistance', 'price_below_support', 'day_of_week', 'month', 'sma_cross', 'ema_cross', 'ar_close', 'ar_high'
        ]

    # Fetch OHLCV data using fetch_data method
    df = fetch_data(instrument_key, from_date, to_date, interval)
    df, full_features = calculate_indicators(df, feature_columns, lagging_count, 'standardization')  # Calculate indicators using the existing method
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

    # Prepare features (X) and target (y)
    X = df[full_features]
    y = df['close'].shift(-prediction_step)  # Predicting the next 30-minute close price

    # Drop the last row due to target shift
    X = X[:-prediction_step]
    y = y.dropna()

    # Ensure the index alignment
    y = y.iloc[:len(X)]

    # Chronological Train-Test Split (80% train, 20% test)
    split_index = int(len(X) * train_split)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters with early stopping
    params = {
        'objective': 'reg:squarederror',
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'seed': seed
    }

    # Train the model with early stopping
    eval_set = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=eval_set, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

    predictions = model.predict(dtest)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Save the trained model to a file
    model_filename = f"{name}_{instrument_key}_{interval}_{from_date}_{to_date}.pkl"
    model_structure_filename = f"{model_filename}_structure.json"
    model_path = os.path.join(models_folder, model_filename)
    model_structure_path = os.path.join(models_folder, model_structure_filename)

    if os.path.exists(model_path):
        try:
            # If the file exists, remove it
            os.remove(model_path)
            print(f"Existing model file '{model_filename}' has been deleted.")
        except Exception as e:
            print(f"Error deleting the model file '{model_filename}': {e}")

    model.save_model(model_path)
    model.dump_model(model_structure_path, dump_format='json')
    
    print(f"Model saved at {model_path}, Structure saved at {model_structure_path}")

    categorized_features = categorize_features(feature_columns, get_full_features())

    xgboost_model = XGBoostModels(
        name=name,
        instrument_key=instrument_key,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
        prediction_step=prediction_step,
        train_split=train_split,
        eta=eta,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        seed=seed,
        max_depth=max_depth,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        model_path=model_path,
        model_structure_path=model_structure_path,
        selected_features=dict(categorized_features),  # Store full feature list in the DB
        training_mse=mse,
        training_mae=mae
    )
    
    return xgboost_model