import pandas as pd
import numpy as np

# Function that takes in stock time series data and adds appropiate features

def stock_feat(df, date_col: str = "Date", close_col: str = "Close", vol_col: str = "Volume"):
    #### Daily stock data

    required_cols = [date_col, close_col, vol_col]
    assert all(col in df.columns for col in required_cols), \
        "Stock data does not have necessary columns"

    df = df.copy()
    
    # Cut data to when volume started to be tracked
    df = df[df[vol_col] != 0]

    # daily return %
    df['Daily_Return_%'] = df[close_col].pct_change() * 100
    
    #20 day simple moving average (SMA_20)
    df['SMA_20'] = df[close_col].rolling(20).mean()
    
    # SMA_50
    df['SMA_50'] = df[close_col].rolling(50).mean()
    
    # Volatility 20 day window
    df['volatility_20'] = df[close_col].pct_change().rolling(20).std() * 100
    
    #EMA 12 and 26, weight applied to price nth step back is alpha(1-alpha)^n
    df["EMA_12"] = df[close_col].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df[close_col].ewm(span=26, adjust=False).mean()
    
    # MACD and signal line
    # This measures the "momentum". Price from smaller window back minus price based on larger window back
    # Signal line smoothes this metric
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Bollinger band width
    df["BB_Mid"] = df[close_col].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    
    df["BB_Upper"] = df[close_col] + 2 * df["BB_Std"]
    df["BB_Lower"] = df[close_col] - 2 * df["BB_Std"]
    
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    return df

def regime_labels(df, N = 21, threshold = 0.05):
    y = pd.Series(name = 'Regime')
    for i in range(len(df) - N):

        # return over the next N days
        forward_return = (df.iloc[i+N]['Close'] / df.iloc[i]['Close']) - 1

        if forward_return >= threshold:
            y[i] = 2 # Bull
        elif forward_return <= -threshold:
            y[i] = 0 # Bear
        else:
            y[i] = 1 # Sideways (not significant)

    return y

def train_validate_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_range: tuple, # all of these pass as datetime
    val_range: tuple,
    test_range: tuple,
    date_col: str = "Date"
):
    
    X = X.copy()

    X[date_col] = pd.to_datetime(X[date_col])

    # helper func to create masks
    def _mask_maker(df, date_range):
        start, end = date_range
        return (df[date_col] >= start) & (df[date_col] <= end)

    # Create masks using function
    train_mask = _mask_maker(X, train_range)
    val_mask = _mask_maker(X, val_range)
    test_mask = _mask_maker(X, test_range)

    y = y.loc[X.index]

    assert len(y) == len(X), "dimension error"
    assert not y.isna().any(), "y NaNs"

    # build the splits
    X_train = X.loc[train_mask].drop(columns=[date_col])
    y_train = y.loc[train_mask]

    X_val = X.loc[val_mask].drop(columns=[date_col])
    y_val = y.loc[val_mask]

    X_test = X.loc[test_mask].drop(columns=[date_col])
    y_test = y.loc[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test