"""
AEGIS VFM Baseline Training Pipeline (v3 - Leakage-Safe)
Trains Model A (Total Liquid) and Model B (Gas) using XGBoost.
Strict chronological splitting and zero future-data leakage.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

def engineer_features(df):
    """Generates time-aware features strictly using past/current data."""
    print("Engineering rolling features and deltas...")
    
    # Sort strictly by well and time
    df = df.sort_values(by=['well_id', 'timestamp']).reset_index(drop=True)
    
    grouped = df.groupby('well_id')
    
    # Deltas (Changes over time)
    df['whp_delta_1h'] = grouped['wellhead_pressure_psi'].diff(1)
    df['whp_delta_6h'] = grouped['wellhead_pressure_psi'].diff(6)
    df['whp_delta_24h'] = grouped['wellhead_pressure_psi'].diff(24)
    df['choke_delta_1h'] = grouped['choke_size_64th'].diff(1)
    df['dp_delta_1h'] = grouped['differential_pressure_psi'].diff(1)
    
    # Rolling averages (min_periods=1 prevents NaNs by using whatever history is available)
    df['whp_rolling_6h'] = grouped['wellhead_pressure_psi'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['dp_rolling_6h'] = grouped['differential_pressure_psi'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    # LEAKAGE FIX: Replace bfill() with safe defaults for initial rows
    delta_cols = ['whp_delta_1h', 'whp_delta_6h', 'whp_delta_24h', 'choke_delta_1h', 'dp_delta_1h']
    df[delta_cols] = df[delta_cols].fillna(0.0)
    
    # One-hot encode the well_id natively
    df = pd.get_dummies(df, columns=['well_id'], prefix='well')
    
    # Ensure boolean columns are integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, target_name):
    """Trains an XGBoost model and returns the model + metrics."""
    print(f"\n--- Training Model: {target_name} ---")
    
    # XGBoost API FIX: early_stopping_rounds moved to constructor for modern compatibility
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=20 
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Stopped at iteration {model.best_iteration}")
    
    # Evaluate on the unseen TEST set (Days 51-60)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    
    print(f"Test MAE:  {mae:.1f}")
    print(f"Test RMSE: {rmse:.1f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    return model

def main():
    print("Loading aegis_data_v2.csv...")
    df = pd.read_csv("aegis_data_v2.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = engineer_features(df)
    
    # Exclude metadata, target variables, and future leaks
    exclude_cols = [
        'timestamp', 'flow_station', 
        'is_production_test_hour', 
        'test_oil_bpd', 'test_gas_mscfd', 
        'test_water_bpd', 'test_liquid_bpd', 'test_water_cut_pct',
        'vibration_index'
    ]
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Filter to physical test rows for training
    df_tests = df[df['is_production_test_hour'] == 1].copy()
    
    # Chronological Split
    day_index = (df_tests['timestamp'] - df_tests['timestamp'].min()).dt.days
    
    train_mask = day_index < 40
    val_mask = (day_index >= 40) & (day_index < 50)
    test_mask = day_index >= 50
    
    X_train, y_liq_train, y_gas_train = df_tests.loc[train_mask, features], df_tests.loc[train_mask, 'test_liquid_bpd'], df_tests.loc[train_mask, 'test_gas_mscfd']
    X_val, y_liq_val, y_gas_val = df_tests.loc[val_mask, features], df_tests.loc[val_mask, 'test_liquid_bpd'], df_tests.loc[val_mask, 'test_gas_mscfd']
    X_test, y_liq_test, y_gas_test = df_tests.loc[test_mask, features], df_tests.loc[test_mask, 'test_liquid_bpd'], df_tests.loc[test_mask, 'test_gas_mscfd']
    
    model_liquid = train_and_evaluate(X_train, y_liq_train, X_val, y_liq_val, X_test, y_liq_test, "Total Liquid (bpd)")
    model_gas = train_and_evaluate(X_train, y_gas_train, X_val, y_gas_val, X_test, y_gas_test, "Gas Rate (mscfd)")
    
    print("\nExporting models...")
    joblib.dump(model_liquid, 'aegis_vfm_liquid.joblib')
    joblib.dump(model_gas, 'aegis_vfm_gas.joblib')
    joblib.dump(features, 'aegis_vfm_features.joblib') 
    
    
    # NEW: Export a 48-hour historical buffer for Streamlit initialization
    print("Exporting 48-hour buffer for Streamlit state management...")
    
    # 48 hours gives enough runway to safely calculate 24h deltas immediately
    buffer_df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(hours=48)]
    buffer_df.to_csv('aegis_history_buffer.csv', index=False)
    
    print("Done! Files saved successfully.")

if __name__ == "__main__":
    main()