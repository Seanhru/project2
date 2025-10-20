# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates

# --- Step 2: Load data ---
df = pd.read_csv("AirQualityUCI.csv")

# Columns of interest
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
df = df[cols].copy()

# --- Step 3: Preprocess ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

# Replace negative values and -200 with NaN for all pollutant/climate columns
for col in ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Interpolate missing values to keep temporal continuity
df = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
df[cols[2:]] = df[cols[2:]].interpolate(method='linear', limit_direction='both')

# --- Step 4: Add time features ---
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# --- Step 5: Add lag features (previous hour & previous day) ---
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)']
for pollutant in pollutants:
    df[f'{pollutant}_lag1'] = df[pollutant].shift(1)     # 1 hour lag
    df[f'{pollutant}_lag24'] = df[pollutant].shift(24)   # 24 hour lag

# Drop rows with NaNs created by lagging
df = df.dropna().reset_index(drop=True)

# --- Step 6: Train/test split ---
test_hours = 720  # last month
train = df.iloc[:-test_hours]
test = df.iloc[-test_hours:]

# --- Step 7: Prepare features ---
feature_cols = ['Hour', 'T', 'RH', 'AH', 'DayOfWeek', 'Month'] + \
               [f'{p}_lag1' for p in pollutants] + [f'{p}_lag24' for p in pollutants]
X_train = train[feature_cols]
X_test = test[feature_cols]

# Store results
results = {}

# --- Step 8: Train Random Forests ---
for pollutant in pollutants:
    print(f"\nTraining model for {pollutant}...")
    
    y_train = train[pollutant]
    y_test = test[pollutant]
    
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Store results
    results[pollutant] = {
        'model': rf,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Feature importance plot
    importance = rf.feature_importances_
    plt.figure(figsize=(8,4))
    sns.barplot(x=importance, y=feature_cols, palette="viridis")
    plt.title(f"Feature Importance for {pollutant}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# --- Step 9: Evaluation summary ---
print("\n--- Model Performance Summary ---")
for pollutant, metrics in results.items():
    print(f"{pollutant}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

# --- Step 10: Plot actual vs predicted for each pollutant ---
test_dates = pd.to_datetime(test['Date'])

for pollutant, metrics in results.items():
    plt.figure(figsize=(12,5))
    plt.plot(test_dates, metrics['y_test'].values, label='Actual', linewidth=2)
    plt.plot(test_dates, metrics['y_pred'], linestyle='--', label='Predicted')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.title(f"{pollutant}: Actual vs Predicted Concentration for Next Month")
    plt.xlabel("Date")
    plt.ylabel("Concentration")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
