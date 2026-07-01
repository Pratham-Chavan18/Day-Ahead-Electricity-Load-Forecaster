"""
Day-Ahead Electricity Load Forecaster

End-to-end pipeline for forecasting hourly electricity load 24 hours ahead
using the Individual Household Electric Power Consumption dataset from UCI.

Steps:
1. Data Loading & Cleaning
2. Resampling to Hourly Data
3. Advanced Feature Engineering (Cyclical encodings, Lags, Rolling statistics)
4. Chronological Train/Test Split & Leakage Prevention
5. Baseline Evaluation & XGBoost Regressor Training
6. Visualization & Performance Reporting
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')


def locate_data_file() -> str:
    """Locates the raw dataset file across common project structures."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, 'data', 'household_power_consumption.txt'),
        os.path.join(base_dir, 'electricity-load-forecaster', 'data', 'household_power_consumption.txt'),
        os.path.join(os.path.dirname(base_dir), 'data', 'household_power_consumption.txt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find household_power_consumption.txt. Ensure data/household_power_consumption.txt exists."
    )


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw minute-level UCI dataset, handles missing values represented as '?',
    and parses Date/Time into a continuous DatetimeIndex.
    """
    print(f"Step 1: Loading raw data from {filepath}...")
    t0 = time.time()
    
    # Specify dtypes for memory efficiency and fast read speeds
    dtypes = {
        'Global_active_power': 'float32',
        'Global_reactive_power': 'float32',
        'Voltage': 'float32',
        'Global_intensity': 'float32',
        'Sub_metering_1': 'float32',
        'Sub_metering_2': 'float32',
        'Sub_metering_3': 'float32'
    }
    
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        dtype=dtypes,
        low_memory=False
    )
    
    print(f"Loaded {len(df):,} minute-level observations in {time.time() - t0:.2f}s.")
    
    print("Parsing Date and Time into DatetimeIndex...")
    t1 = time.time()
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.sort_index(inplace=True)
    print(f"Datetime conversion completed in {time.time() - t1:.2f}s.")
    
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples minute-level data to hourly blocks using domain-appropriate aggregations:
    - Energy metrics (Active power, Reactive power, Sub-meterings) are summed.
    - Instantaneous metrics (Voltage, Intensity) are averaged.
    """
    print("Step 2: Resampling minute-level reads to hourly consumption...")
    agg_dict = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum'
    }
    
    resampled = df.resample('1h').agg(agg_dict)
    
    # Forward/backward fill minor isolated gaps created by resampling if any
    resampled.interpolate(method='linear', limit=3, inplace=True)
    resampled.dropna(inplace=True)
    
    print(f"Hourly dataset shape: {resampled.shape[0]:,} hours ({resampled.index.min()} to {resampled.index.max()}).")
    return resampled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers advanced time-series features while strictly preventing future data leakage:
    - Cyclical Sine/Cosine time encodings (hour, day of week, day of month, month).
    - Historical lag features (strictly >= 24h for 24h day-ahead forecasting).
    - Rolling window trends and historical same-hour aggregates shifted by 24h.
    """
    print("Step 3: Engineering time-series features...")
    df_feat = df.copy()
    
    # 1. Cyclical Time Encoding
    hours = df_feat.index.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    dows = df_feat.index.dayofweek
    df_feat['dow_sin'] = np.sin(2 * np.pi * dows / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * dows / 7)
    
    days = df_feat.index.day
    df_feat['day_sin'] = np.sin(2 * np.pi * days / 31)
    df_feat['day_cos'] = np.cos(2 * np.pi * days / 31)
    
    months = df_feat.index.month
    df_feat['month_sin'] = np.sin(2 * np.pi * months / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * months / 12)
    
    # 2. Lag Features (Usage at same hour over past 7 days: 24h, 48h, ..., 168h ago)
    for d in range(1, 8):
        df_feat[f'lag_{d*24}'] = df_feat['Global_active_power'].shift(d * 24)
        
    # 3. Same-Hour Historical Statistics (Mean/Std across recent previous days at the same hour)
    same_hr_cols = [f'lag_{d*24}' for d in range(1, 8)]
    df_feat['same_hr_mean_3d'] = df_feat[[f'lag_{d*24}' for d in range(1, 4)]].mean(axis=1)
    df_feat['same_hr_mean_7d'] = df_feat[same_hr_cols].mean(axis=1)
    df_feat['same_hr_std_7d'] = df_feat[same_hr_cols].std(axis=1)
    df_feat['same_hr_min_7d'] = df_feat[same_hr_cols].min(axis=1)
    df_feat['same_hr_max_7d'] = df_feat[same_hr_cols].max(axis=1)
    
    # 4. Recent Rolling Trends (Strictly applied to shift(24) to ensure 0 leakage into forecast horizon)
    shifted_24 = df_feat['Global_active_power'].shift(24)
    df_feat['roll_mean_6'] = shifted_24.rolling(window=6).mean()
    df_feat['roll_mean_12'] = shifted_24.rolling(window=12).mean()
    df_feat['roll_mean_24'] = shifted_24.rolling(window=24).mean()
    df_feat['roll_std_24'] = shifted_24.rolling(window=24).std()
    df_feat['roll_mean_168'] = shifted_24.rolling(window=168).mean()
    
    # Drop initial rows with NaNs resulting from 168h lag shifts
    df_feat.dropna(inplace=True)
    print(f"Feature engineering completed. Total features generated: {df_feat.shape[1] - 1}.")
    return df_feat


def train_and_evaluate(df_feat: pd.DataFrame, split_date: str = '2010-01-01'):
    """
    Performs chronological train/test split, drops concurrent leakage columns,
    evaluates naive 24h baseline, and trains an optimized XGBoost regressor.
    """
    print(f"Step 4: Performing chronological train/test split at {split_date}...")
    train_df = df_feat[df_feat.index < split_date]
    test_df = df_feat[df_feat.index >= split_date]
    
    # Drop features occurring simultaneously with the target variable to prevent leakage
    leakage_cols = [
        'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    feature_cols = [col for col in df_feat.columns if col not in ['Global_active_power'] + leakage_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Global_active_power']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Global_active_power']
    
    print(f"Training set: {len(X_train):,} hours | Testing set: {len(X_test):,} hours")
    
    print("Step 5: Evaluating Naive Baseline ('Tomorrow looks like Today')...")
    # Naive baseline is exactly lag_24 (usage at the exact same hour yesterday)
    naive_preds = test_df['lag_24']
    mae_naive = mean_absolute_error(y_test, naive_preds)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_preds))
    
    print(f"--> Naive Baseline MAE : {mae_naive:.4f} kW-h")
    print(f"--> Naive Baseline RMSE: {rmse_naive:.4f} kW-h")
    
    print("Training XGBoost Regressor...")
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Model training completed in {time.time() - t0:.2f}s.")
    
    xgb_preds = pd.Series(model.predict(X_test), index=X_test.index)
    mae_xgb = mean_absolute_error(y_test, xgb_preds)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
    
    mae_impr = ((mae_naive - mae_xgb) / mae_naive) * 100
    rmse_impr = ((rmse_naive - rmse_xgb) / rmse_naive) * 100
    
    print("=" * 60)
    print("FINAL EVALUATION RESULTS (TEST SET >= 2010-01-01)")
    print("=" * 60)
    print(f"Model            | MAE (kW-h) | RMSE (kW-h) | Improvement")
    print(f"------------------------------------------------------------")
    print(f"Naive Baseline   | {mae_naive:10.4f} | {rmse_naive:11.4f} |        --- ")
    print(f"XGBoost Model    | {mae_xgb:10.4f} | {rmse_xgb:11.4f} | +{mae_impr:.1f}% MAE")
    print("=" * 60)
    
    return y_test, naive_preds, xgb_preds, (mae_naive, rmse_naive, mae_xgb, rmse_xgb)


def generate_visualization(y_test: pd.Series, naive_preds: pd.Series, xgb_preds: pd.Series, metrics: tuple, output_path: str):
    """Generates an informative, polished chart comparing actual usage against forecasts."""
    print("Step 6: Generating high-resolution visualization chart...")
    mae_naive, rmse_naive, mae_xgb, rmse_xgb = metrics
    
    # Select a clear 10-day window in January 2010 to show detailed diurnal behavior
    start_plot = '2010-01-10'
    end_plot = '2010-01-20'
    
    y_slice = y_test.loc[start_plot:end_plot]
    naive_slice = naive_preds.loc[start_plot:end_plot]
    xgb_slice = xgb_preds.loc[start_plot:end_plot]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.2)
    
    # Top plot: Time series comparison
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(y_slice.index, y_slice.values, label='Actual Consumption', color='#1f77b4', linewidth=2.0, alpha=0.9)
    ax0.plot(naive_slice.index, naive_slice.values, label=f'Naive 24h Baseline (MAE: {mae_naive:.2f})', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax0.plot(xgb_slice.index, xgb_slice.values, label=f'XGBoost Day-Ahead Forecast (MAE: {mae_xgb:.2f})', color='#2ca02c', linewidth=2.0, alpha=0.9)
    
    ax0.set_title('Day-Ahead Electricity Load Forecast vs Actual Consumption (Sample Window: Jan 10 - Jan 20, 2010)', fontsize=14, fontweight='bold', pad=15)
    ax0.set_ylabel('Total Active Power (kW-h per hour)', fontsize=12, fontweight='bold')
    ax0.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    ax0.tick_params(labelsize=10)
    
    # Bottom left: Error comparison scatter/residual distribution across test set
    ax1 = fig.add_subplot(gs[1, 0])
    residuals_naive = y_test - naive_preds
    residuals_xgb = y_test - xgb_preds
    
    ax1.hist(residuals_naive, bins=80, range=(-100, 100), alpha=0.5, color='#ff7f0e', label='Naive Residuals')
    ax1.hist(residuals_xgb, bins=80, range=(-100, 100), alpha=0.6, color='#2ca02c', label='XGBoost Residuals')
    ax1.set_title('Test Set Forecast Error Distribution (Residuals = Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Prediction Error (kW-h)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.legend(loc='upper right', frameon=True)
    
    # Bottom right: Bar chart comparison of MAE / RMSE
    ax2 = fig.add_subplot(gs[1, 1])
    models = ['Naive\nBaseline', 'XGBoost\nModel']
    maes = [mae_naive, mae_xgb]
    rmses = [rmse_naive, rmse_xgb]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, maes, width, label='MAE', color='#3498db')
    rects2 = ax2.bar(x + width/2, rmses, width, label='RMSE', color='#e74c3c')
    
    ax2.set_title('Test Set Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (kW-h)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.legend()
    
    # Add value labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax2.set_ylim(0, max(rmses) * 1.25)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Chart successfully saved to {output_path}")


def main():
    print("=" * 60)
    print("STARTING DAY-AHEAD ELECTRICITY LOAD FORECASTING PIPELINE")
    print("=" * 60)
    
    data_path = locate_data_file()
    df_raw = load_and_clean_data(data_path)
    df_hourly = resample_to_hourly(df_raw)
    df_feat = engineer_features(df_hourly)
    y_test, naive_preds, xgb_preds, metrics = train_and_evaluate(df_feat)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_chart = os.path.join(base_dir, 'forecast_results.png')
    generate_visualization(y_test, naive_preds, xgb_preds, metrics, output_chart)
    
    print("Pipeline finished successfully!")


if __name__ == '__main__':
    main()
