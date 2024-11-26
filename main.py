import pandas as pd
import zipfile
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
BASE_PATH = "histdata_free/"
import zipfile
import pandas as pd
import numpy as np


def load_eurusd_data(base_path: str, year: int) -> pd.DataFrame:
    """
    Unzips the specified EURUSD data file and loads it into a pandas DataFrame.

    Args:
        base_path: The path to the directory containing the data file.
        year: The year of the data to load.

    Returns:
        A pandas DataFrame containing the EURUSD data.
    """
    file_path = os.path.join(base_path, f"HISTDATA_COM_NT_EURUSD_M1{year}.zip")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(base_path)

    csv_file_path = os.path.join(base_path, f"DAT_NT_EURUSD_M1_{year}.csv")
    df = pd.read_csv(
        csv_file_path,
        delimiter=";",
        names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
        index_col=False,
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y%m%d %H%M%S")
    df.set_index("DateTime", inplace=True, drop=True)
    return df

def get_ewm_macd(data: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    return data["Close"].ewm(span=short_window).mean() - data["Close"].ewm(span=long_window).mean()

def get_pinbar(
    data: pd.DataFrame, body_threshold: float = 0.2, wick_threshold: float = 0.7
) -> pd.Series:
    """
    Calculates the pinbar feature.

    Args:
        data: The input data DataFrame with columns "Open", "High", "Low", "Close".

    Returns:
        A pandas Series containing +1 for bullish pinbars, -1 for bearish pinbars,
        and 0 otherwise.
    """

    # Calculate the range and body size
    data["Range"] = data["High"] - data["Low"]
    data["Body"] = abs(data["Close"] - data["Open"])

    # Calculate the upper and lower shadows (wicks)
    data["UpperShadow"] = data["High"] - data[["Close", "Open"]].max(axis=1)
    data["LowerShadow"] = data[["Close", "Open"]].min(axis=1) - data["Low"]

    # Identify pinbars based on thresholds
    # is_pinbar = (data["Body"] / data["Range"] <= body_threshold) & (
    #     (data["UpperShadow"] / data["Range"] >= wick_threshold)
    #     | (data["LowerShadow"] / data["Range"] >= wick_threshold)
    # )

    # ... (previous code for range, body, shadows remains the same) ...

    # Identify bullish and bearish pinbars
    is_bullish_pinbar = (
        (data["Body"] / data["Range"] <= body_threshold)
        & (data["LowerShadow"] / data["Range"] >= wick_threshold)
        & (data["Close"] > data["Open"])  # Close is higher than Open (bullish)
    )

    is_bearish_pinbar = (
        (data["Body"] / data["Range"] <= body_threshold)
        & (data["UpperShadow"] / data["Range"] >= wick_threshold)
        & (data["Close"] < data["Open"])  # Close is lower than Open (bearish)
    )

    # Combine the results
    pinbar_direction = pd.Series(0, index=data.index)  # Initialize with 0
    pinbar_direction[is_bullish_pinbar] = 1
    pinbar_direction[is_bearish_pinbar] = -1

    return pinbar_direction  # Convert to 1 for pinbars, 0 otherwise


def get_garman_klass(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the Garman-Klass volatility.

    Args:
        data: The input data DataFrame with columns "Open", "High", "Low", "Close".

    Returns:
        A pandas Series containing the Garman-Klass volatility values.
    """

    return np.sqrt(
        0.5 * (np.log(data["High"] / data["Low"]) ** 2)
        - (2 * np.log(2) - 1) * (np.log(data["Close"] / data["Open"]) ** 2)
    )

def rolling_regression_forecast(
    features_df: pd.DataFrame, window_size: int = 10  # 1 week in minutes
) -> pd.DataFrame:
    """
    Performs rolling regression to forecast forward returns.

    Args:
        features_df: DataFrame with features and 'fwd_rets'.
        window_size: Size of the rolling window in minutes (default: 1 week).

    Returns:
        DataFrame with forecasts and performance metrics.
    """

    # Assuming your DataFrame is called features_df
    # and it has a DateTimeIndex
    # It should contain 'fwd_rets' and other feature columns

    # Ensure index is unique before proceeding
    # If there are duplicates, keep the first occurrence
    features_df = features_df[~features_df.index.duplicated(keep="first")]

    results_df = pd.DataFrame(index=features_df.index)
    results_df["y_true"] = features_df["fwd_rets"]  # Store true values
    results_df["y_pred"] = np.nan  # Initialize predictions with NaN

    start_index = 0
    while start_index + window_size < len(features_df):
        # Get data for the current window
        window_data = features_df.iloc[start_index : start_index + window_size]

        # Separate features (X) and target (y)
        X = window_data.drop(columns=["fwd_rets"])
        y = window_data["fwd_rets"]

        # Fit the regression model
        X = sm.add_constant(X)  # Add a constant term
        model = sm.OLS(y, X).fit()

        # Forecast for the next week
        next_week_data = features_df.iloc[
            start_index + window_size : start_index + 2 * window_size
        ]

        next_week_X = sm.add_constant(
            next_week_data.drop(columns=["fwd_rets"])
        )  # Add a constant term and drop target for prediction

        # using .values and aligning indices will avoid reindexing issues
        # Get the common index between next_week_data and results_df
        common_index = next_week_data.index.intersection(results_df.index)

        # Assign predictions to the common index in results_df
        results_df.loc[common_index, "y_pred"] = model.predict(
            next_week_X
        ).values  # Store predictions

        start_index += window_size  # Move to the next window

    # Calculate performance metrics
    results_df = results_df.dropna()  # Remove rows with NaN predictions
    mse = mean_squared_error(results_df["y_true"], results_df["y_pred"])
    mape = mean_absolute_percentage_error(results_df["y_true"], results_df["y_pred"])
    correlation = results_df["y_true"].corr(results_df["y_pred"])

    print(f"Out-of-sample MSE: {mse}")
    print(f"Out-of-sample MAPE: {mape}")
    print(f"Correlation between y and y^: {correlation}")
    return results_df


