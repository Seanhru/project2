# ==========================
# AIR QUALITY MODEL - RANDOM FOREST
# Validation (March 2005) + Future Prediction (May 2005)
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------
# Load and clean data
# --------------------------

df = pd.read_csv('AirQualityUCI.csv', sep=';')
df.columns = df.columns.str.strip() 

# Remove extra columns and handle missing data
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
df = df.dropna(subset=['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)'])

# Replace invalid (negative) values with zeros
df[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']] = (
    df[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']].clip(lower=0)
)

# Combine date and time into a datetime column
df['Datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'], 
    format='%d/%m/%Y %H.%M.%S', 
    errors='coerce'
)
df = df.dropna(subset=['Datetime'])
df = df.sort_values(by='Datetime')
df.set_index('Datetime', inplace=True)

# --------------------------
# Define pollutant columns and units
# --------------------------

pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

unit_conversions = {
    'CO(GT)': 'mg/m³',
    'NMHC(GT)': 'µg/m³',
    'C6H6(GT)': 'µg/m³',
    'NOx(GT)': 'ppb',
    'NO2(GT)': 'µg/m³'
}

df = df[pollutants]

# --------------------------
# Split data into training (before March) and validation (March)
# --------------------------

train_df = df[df.index < '2005-03-01']
val_df = df[(df.index >= '2005-03-01') & (df.index < '2005-04-01')]

X_train = np.arange(len(train_df)).reshape(-1, 1)
X_val = np.arange(len(train_df), len(train_df) + len(val_df)).reshape(-1, 1)

# --------------------------
# Train, validate, and plot
# --------------------------

models = {}
val_predictions = {}

for pollutant in pollutants:
    y_train = train_df[pollutant]
    y_val = val_df[pollutant]
    
    # Train random forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict validation set
    y_pred = model.predict(X_val)
    
    # Compute metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Store model and predictions
    models[pollutant] = model
    val_predictions[pollutant] = y_pred
    
    print(f"{pollutant} ({unit_conversions[pollutant]}): R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    # Plot actual vs predicted for March 2005
    plt.figure(figsize=(10, 5))
    plt.plot(val_df.index, y_val, label='Actual', linewidth=2)
    plt.plot(val_df.index, y_pred, label='Predicted', linestyle='--', linewidth=2)
    plt.title(f'Actual vs Predicted {pollutant} ({unit_conversions[pollutant]}) - March 2005', fontsize=13)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel(f'Concentration ({unit_conversions[pollutant]})', fontsize=11)
    plt.legend()
    plt.grid(True)
    
    # Display R² and RMSE on the plot
    plt.text(
        0.02, 0.92,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    plt.tight_layout()
    plt.show()

# --------------------------
# Predict FUTURE data (May 2005)
# --------------------------

# Create hourly timestamps for May 2005
may_dates = pd.date_range(start='2005-05-01', end='2005-05-31 23:00:00', freq='H')

# Define X values continuing from the end of training + validation
X_future = np.arange(len(train_df) + len(val_df), len(train_df) + len(val_df) + len(may_dates)).reshape(-1, 1)

# Predict future pollutant concentrations
future_predictions = {}
for pollutant, model in models.items():
    future_predictions[pollutant] = model.predict(X_future)

# Create DataFrame for May predictions
future_df = pd.DataFrame(future_predictions, index=may_dates)

# Save predictions to CSV file
future_df.to_csv('predicted_may_2005.csv')

print("Future predictions saved to 'predicted_may_2005.csv'")

# --------------------------
# Plot predicted future concentrations
# --------------------------

for pollutant in pollutants:
    plt.figure(figsize=(10, 5))
    plt.plot(future_df.index, future_df[pollutant], color='tab:blue', linewidth=2)
    plt.title(f'Predicted {pollutant} ({unit_conversions[pollutant]}) - May 2005', fontsize=13)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel(f'Concentration ({unit_conversions[pollutant]})', fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
