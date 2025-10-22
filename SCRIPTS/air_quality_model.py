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

# Columns of interest (include NO2(GT))
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
df = df[cols].copy()

# --- Step 3: Preprocess ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

# Replace negative values and -200 with NaN for all pollutant/climate columns
for col in ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Interpolate missing values to maintain temporal continuity
df = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
df[cols[2:]] = df[cols[2:]].interpolate(method='linear', limit_direction='both')

# --- Step 4: Add time features ---
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# --- Step 5: Add lag features (previous hour & previous day) ---
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
for pollutant in pollutants:
    df[f'{pollutant}_lag1'] = df[pollutant].shift(1)     # 1 hour lag
    df[f'{pollutant}_lag24'] = df[pollutant].shift(24)   # 24 hour lag

# Drop rows with NaNs created by lagging
df = df.dropna().reset_index(drop=True)

# --- Step 6: Train/test split ---
test_hours = 720  # Last month = March
train = df.iloc[:-test_hours]
test = df.iloc[-test_hours:]

# --- Step 7: Prepare features ---
feature_cols = ['Hour', 'T', 'RH', 'AH', 'DayOfWeek', 'Month'] + \
               [f'{p}_lag1' for p in pollutants] + [f'{p}_lag24' for p in pollutants]
X_train = train[feature_cols]
X_test = test[feature_cols]

# Save cleaned and processed data used for training
df.to_csv("cleaned_airquality_data.csv", index=False)

# --- Step 8: Train Random Forests ---
results = {}
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

# --- Step 9: Evaluation summary ---
print("\n--- Model Performance Summary (Validation on March Data) ---")
for pollutant, metrics in results.items():
    print(f"{pollutant}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

# Units for labeling
units = {
    'CO(GT)': 'mg/m³',
    'NMHC(GT)': 'µg/m³',
    'C6H6(GT)': 'µg/m³',
    'NOx(GT)': 'ppb',
    'NO2(GT)': 'µg/m³'
}

# --- Step 9b: Create a table figure of validation metrics ---
import matplotlib.pyplot as plt

# --- Prepare data for table, excluding NMHC(GT) ---
metrics_data = []
for pollutant, metrics in results.items():
    if pollutant != 'NMHC(GT)':  # skip NMHC
        metrics_data.append([pollutant, f"{metrics['r2']:.3f}", f"{metrics['rmse']:.3f}"])

# Column labels
col_labels = ["Pollutant", "R² (March)", "RMSE (March)"]

# Create figure
fig, ax = plt.subplots(figsize=(6, 2.5))  # slightly taller figure
ax.axis('off')  # Hide axes

# Create table
table = ax.table(cellText=metrics_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

fig.suptitle("Validation Metrics for March (Actual vs Predicted)", fontsize=14)
plt.tight_layout()
plt.show()


# --- Step 10: Plot Actual vs Predicted for Validation Period ---
test_dates = pd.to_datetime(test['Date'])

for pollutant, metrics in results.items():
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, metrics['y_test'].values, label='Actual', linewidth=2)
    plt.plot(test_dates, metrics['y_pred'], linestyle='--', label='Predicted', color='orange')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.title(f"{pollutant}: Actual vs Predicted Concentration (Validation - March)")
    plt.xlabel("Date")
    plt.ylabel(f"Concentration ({units[pollutant]})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Step 11: Predict next month (future prediction) ---
# Generate feature data for next month (April) based on last known values
future_hours = 720  # 30 days * 24 hours
last_row = df.iloc[-1].copy()

future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(hours=1), periods=future_hours, freq='H')
future_df = pd.DataFrame({
    'Date': future_dates,
    'Hour': future_dates.hour,
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month,
})

# Use last known climate features for simplicity (can be replaced with forecasts)
for col in ['T', 'RH', 'AH']:
    future_df[col] = last_row[col]

# Initialize lag columns
for pollutant in pollutants:
    future_df[f'{pollutant}_lag1'] = np.nan
    future_df[f'{pollutant}_lag24'] = np.nan

# Iteratively predict future values
predictions = {pollutant: [] for pollutant in pollutants}
for i in range(future_hours):
    row_features = []
    for col in feature_cols:
        if col.endswith('_lag1'):
            pollutant = col.replace('_lag1', '')
            row_features.append(df[pollutant].iloc[-1] if i == 0 else predictions[pollutant][-1])
        elif col.endswith('_lag24'):
            pollutant = col.replace('_lag24', '')
            if i < 24:
                row_features.append(df[pollutant].iloc[-24 + i])
            else:
                row_features.append(predictions[pollutant][-24])
        else:
            row_features.append(future_df.iloc[i][col])
    for pollutant in pollutants:
        model = results[pollutant]['model']
        pred = model.predict([row_features])[0]
        predictions[pollutant].append(pred)

# --- Step 12: Plot future predictions ---
for pollutant in pollutants:
    plt.figure(figsize=(12, 5))
    plt.plot(future_dates, predictions[pollutant], label='Predicted Future', color='purple')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.title(f"{pollutant}: Predicted Concentration for Next Month (April)")
    plt.xlabel("Date")
    plt.ylabel(f"Concentration ({units[pollutant]})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
# --- Step 12b: Save future predictions to CSV ---
future_predictions_df = future_df.copy()
for pollutant in pollutants:
    future_predictions_df[pollutant] = predictions[pollutant]

future_predictions_df.to_csv("predicted_airquality_next_month.csv", index=False)
print("Future predictions saved to predicted_air_quality_next_month.csv")
