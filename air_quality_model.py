# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Step 2: Load data ---
df = pd.read_csv("AirQualityUCI.csv")

# Keep only the columns of interest
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
df = df[cols].copy()

# --- Step 3: Clean and preprocess ---
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert 'Time' (e.g., '18:00:00') to hour of day
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

# Drop rows with missing values
df = df.dropna(subset=['Date', 'Hour', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH'])

columns_to_clip = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
df[columns_to_clip] = df[columns_to_clip].clip(lower=0)

# --- Step 4: Correlation heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(df[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation between Pollutants and Climate Conditions")
plt.show()

# --- Step 5: Sort chronologically ---
df = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)

# --- Step 6: Split data into train (11 months) and test (1 month) ---
test_hours = 720  # ~30 days * 24 hours
train = df.iloc[:-test_hours]
test = df.iloc[-test_hours:]

# Feature inputs
X_train = train[['Hour', 'T', 'RH', 'AH']]
X_test = test[['Hour', 'T', 'RH', 'AH']]

# Targets (pollutant concentrations)
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)']

# Store results
results = {}

# --- Step 7: Train a separate Random Forest for each pollutant ---
for pollutant in pollutants:
    print(f"\nTraining model for {pollutant}...")
    
    y_train = train[pollutant]
    y_test = test[pollutant]
    
    rf = RandomForestRegressor(
        n_estimators=200,
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
    plt.figure(figsize=(6,4))
    sns.barplot(x=importance, y=X_train.columns, palette="viridis")
    plt.title(f"Feature Importance for {pollutant}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()

# --- Step 8: Display model evaluation metrics ---
print("\n--- Model Performance Summary ---")
for pollutant, metrics in results.items():
    print(f"{pollutant}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

# --- Step 9: Separate time-series plots for each pollutant ---
import matplotlib.dates as mdates

# Convert Date column to datetime if not already
test_dates = pd.to_datetime(test['Date'])

for pollutant, metrics in results.items():
    plt.figure(figsize=(12,5))
    
    # Plot actual values
    plt.plot(test_dates, metrics['y_test'].values, label='Actual', linewidth=2)
    
    # Plot predicted values
    plt.plot(test_dates, metrics['y_pred'], linestyle='--', label='Predicted')
    
    # Format x-axis to show days
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # every 2 days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.title(f"{pollutant}: Actual vs Predicted Concentration for Next Month")
    plt.xlabel("Date")
    plt.ylabel("Concentration")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
