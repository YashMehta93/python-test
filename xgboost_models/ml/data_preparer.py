import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


FEATURE_FUNCTIONS = {
    # 2. Price Changes & Returns
    "price_changes_and_returns": lambda df: {
        "daily_return": df['close'].pct_change(),
        "log_return": np.log(df['close'] / df['close'].shift(1)),
        "high_low_diff": df['high'] - df['low'],
        "open_close_diff": df['open'] - df['close']
    },
    # 3. Moving Averages
    "moving_averages": lambda df: {
        "sma_10": talib.SMA(df['close'], timeperiod=10),
        "ema_12": talib.EMA(df['close'], timeperiod=12),
        "ema_26": talib.EMA(df['close'], timeperiod=26),
        "wma_20": talib.WMA(df['close'], timeperiod=20),
        "sma_50": talib.SMA(df['close'], timeperiod=50),
        "sma_200": talib.SMA(df['close'], timeperiod=200),
        "ichimoku_tenkan_sen": (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2,
        "ichimoku_kijun_sen": (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    },
    # 4. Momentum Indicators
    "momentum_indicators": lambda df: {
        "rsi_14": talib.RSI(df['close'], timeperiod=14),
        "macd": talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0],
        "macd_signal": talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)[1],
        "macd_hist": talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2],
        "stoch_k": talib.STOCH(df['high'], df['low'], df['close'])[0],
        "stoch_d": talib.STOCH(df['high'], df['low'], df['close'])[1],
        "cci_20": talib.CCI(df['high'], df['low'], df['close'], timeperiod=20),
        "roc_10": talib.ROC(df['close'], timeperiod=10),
        "williamsr_14": talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14),
        "mfi_14": talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    },
    # 5. Volatility Indicators
    "volatility_indicators": lambda df: {
        "bollinger_upper": talib.BBANDS(df['close'], timeperiod=20)[0],
        "bollinger_middle": talib.BBANDS(df['close'], timeperiod=20)[1],
        "bollinger_lower": talib.BBANDS(df['close'], timeperiod=20)[2],
        "atr_14": talib.ATR(df['high'], df['low'], df['close'], timeperiod=14),
        "donchian_high": df['high'].rolling(window=20).max(),
        "donchian_low": df['low'].rolling(window=20).min(),
        "choppiness_index": 100 * np.log10((talib.ATR(df['high'], df['low'], df['close'], timeperiod=14) /
                                           talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).rolling(14).mean())) / np.log10(14),
        "natr_14": talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    },
    # 6. Volume-Based Indicators
    "volume_indicators": lambda df: {
        "obc": talib.OBV(df['close'], df['volume']),
        "chaikin_money_flow": talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10),
        "accum_dist": talib.AD(df['high'], df['low'], df['close'], df['volume']),
        "force_index": df['close'].diff(1) * df['volume']
    },
    # 7. Support and Resistance Levels
    "support_and_resistance": lambda df: {
        "pivot_point": (df['high'] + df['low'] + df['close']) / 3,
        "fibonacci_38.2": df['low'] + 0.382 * (df['high'] - df['low']),
        "fibonacci_61.8": df['low'] + 0.618 * (df['high'] - df['low']),
        "support_1": (2 * ((df['high'] + df['low'] + df['close']) / 3)) - df['high'],
        "resistance_1": (2 * ((df['high'] + df['low'] + df['close']) / 3)) - df['low']
    },
    # 8. Candlestick Patterns
    "candlestick_patterns": lambda df: {
        "doji": talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']),
        "engulfing": talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']),
        "hammer": talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']),
        "shooting_star": talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']),
        "three_black_crows": talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    },
    # 9. Trend Analysis
    "trend_analysis": lambda df: {
        "adx_14": talib.ADX(df['high'], df['low'], df['close'], timeperiod=14),
        "aroon_up": talib.AROON(df['high'], df['low'], timeperiod=25)[0],
        "aroon_down": talib.AROON(df['high'], df['low'], timeperiod=25)[1],
        "parabolic_sar": talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2),
        "dmi_plus": talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14),
        "dmi_minus": talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    },
    # 10. Statistical Features
    "statistical_features": lambda df: {
        "z_score": (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std(),
        "kurtosis_10": df['close'].rolling(10).apply(kurtosis),
        "skewness_10": df['close'].rolling(10).apply(skew),
        "sharpe_ratio": df['daily_return'].rolling(window=20).mean() / df['daily_return'].rolling(window=20).std()
    },
    # 11. Breakout Indicators
    "breakout_indicators": lambda df: {
        "price_above_resistance": np.where(df['close'] > df['resistance_1'], 1, 0),
        "price_below_support": np.where(df['close'] < df['support_1'], 1, 0)
    },
    # 12. Time-Based Features
    "time_features": lambda df: {
        "day_of_week": df.index.dayofweek,
        "month": df.index.month,
        "hour": df.index.hour
    },
    # 14. Moving Average Crossovers
    "moving_average_crossovers": lambda df: {
        "sma_cross": np.where(df['sma_10'] > df['sma_50'], 1, 0),
        "ema_cross": np.where(df['ema_12'] > df['ema_26'], 1, 0)
    },
    # 15. Autoregressive Features
    "autoregressive_features": lambda df: {
        "ar_close": df['close'].shift(1) - df['close'].shift(2),
        "ar_high": df['high'].shift(1) - df['high'].shift(2)
    }
}

def get_full_features():
    return FEATURE_FUNCTIONS

def calculate_indicators(df, selected_features, counts, scaling_method='standardization'):

    for feature in selected_features:
        # Check if the feature exists in any category
        found = False
        for category, features in FEATURE_FUNCTIONS.items():
            if feature in features:
                df[feature] = features[feature](df)
                found = True
                break
        if not found:
            print(f"Feature '{feature}' not found in FEATURE_FUNCTIONS. Skipping.")

    # Generate lagged features if specified
    lagged_features = []
    if counts > 0:
        feature_columns = df.columns  # Include all current features
        for i in range(1, counts + 1):
            for col in feature_columns:
                lagged_col_name = f"{col}_lag_{i}"
                df[lagged_col_name] = df[col].shift(i)
                lagged_features.append(lagged_col_name)

    updated_selected_features = selected_features + lagged_features

    df.fillna(df.mean(), inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    df.fillna(method='ffill', inplace=True) 
    df = df.apply(pd.to_numeric, errors='coerce')

    if scaling_method == 'standardization':
        scaler = StandardScaler()
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaling_method. Choose 'standardization' or 'normalization'.")
    
    df[updated_selected_features] = scaler.fit_transform(df[updated_selected_features])

    return df, updated_selected_features