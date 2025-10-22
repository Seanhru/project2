# --- Step 1: Import libraries ---
# Import all necessary Python libraries for data manipulation, modeling, and visualization
import pandas as pd                     # For data manipulation and analysis
import numpy as np                      # For numerical operations
import matplotlib.pyplot as plt          # For plotting graphs
from sklearn.ensemble import RandomForestRegressor  # Machine learning model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Model evaluation metrics
import matplotlib.dates as mdates        # For formatting dates on time-series plots

# --- Step 2: Load data ---
# Load the Air Quality dataset from a CSV file
df = pd.read_csv("AirQualityUCI.csv")

# Select only the columns we’re interested in for this analysis
# Added NO2(GT) to include nitrogen dioxide as an additional pollutant
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
df = df[cols].copy()

# --- Step 3: Preprocess ---
# Convert 'Date' to datetime type (handle errors gracefully)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract hour from the 'Time' column for time-based feature creation
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

# Replace invalid or missing readings:
# Values below 0 are treated as invalid (negative pollutant concentrations are impossible)
for col in ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Interpolate missing values linearly across time to maintain continuity in the dataset
df = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
df[cols[2:]] = df[cols[2:]].interpolate(method='linear', limit_direction='both')

# --- Step 4: Add time features ---
# Create new columns for the day of the week (0=Monday, 6=Sunday)
# and for the month number (1=January, 12=December)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# --- Step 5: Add lag features (previous hour & previous day) ---
# Create lag features to help the model learn temporal dependencies
# For example, pollutant concentration at 1 hour ago (lag1)
# and at 24 hours ago (lag24) are often predictive of current concentration
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
for pollutant in pollutants:
    df[f'{pollutant}_lag1'] = df[pollutant].shift(1)     # 1-hour lag
    df[f'{pollutant}_lag24'] = df[pollutant].shift(24)   # 24-hour lag (previous day)

# Drop the first few rows that now contain NaN values due to lagging
df = df.dropna().reset_index(drop=True)

# --- Step 6: Save cleaned dataset ---
# Save the fully preprocessed and feature-augmented dataset to a new CSV file
df.to_csv("cleaned_air_quality_data.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_air_quality_data.csv'.")

# --- Step 7: Train/test split ---
# Split the data into training and testing sets.
# We'll reserve the last 720 hours (~1 month) of data as the test set.
test_hours = 720
train = df.iloc[:-test_hours]
test = df.iloc[-test_hours:]

# --- Step 8: Prepare features for model input ---
# Define which columns will be used as input (X) features.
# Includes meteorological variables, time features, and lagged pollutant values.
feature_cols = ['Hour', 'T', 'RH', 'AH', 'DayOfWeek', 'Month'] + \
               [f'{p}_lag1' for p in pollutants] + [f'{p}_lag24' for p in pollutants]
X_train = train[feature_cols]
X_test = test[feature_cols]

# Dictionary to store trained models and performance metrics for each pollutant
results = {}

# --- Step 9: Train Random Forest models ---
# Train a separate Random Forest Regressor for each pollutant
for pollutant in pollutants:
    print(f"\nTraining model for {pollutant}...")
    
    # Define target (y) variable for current pollutant
    y_train = train[pollutant]
    y_test = test[pollutant]
    
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=500,   # Number of trees
        max_depth=15,       # Limit tree depth to prevent overfitting
        random_state=42,    # For reproducibility
        n_jobs=-1           # Use all CPU cores for faster training
    )
    
    # Fit the model on the training data
    rf.fit(X_train, y_train)
    
    # Predict pollutant concentrations on the test set
    y_pred = rf.predict(X_test)
    
    # Calculate and store performance metrics
    results[pollutant] = {
        'model': rf,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# --- Step 10: Print model evaluation summary ---
# Display a performance summary for each pollutant model
print("\n--- Model Performance Summary ---")
for pollutant, metrics in results.items():
    print(f"{pollutant}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

# --- Step 11: Plot actual vs predicted concentration for each pollutant ---
# Create a separate time-series plot for each pollutant showing
# how well the model predicted concentrations over the last month
test_dates = pd.to_datetime(test['Date'])

# Define y-axis units for each pollutant to ensure plots are correctly labeled
pollutant_units = {
    'CO(GT)': 'Concentration (mg/m³)',
    'NMHC(GT)': 'Concentration (µg/m³)',
    'C6H6(GT)': 'Concentration (µg/m³)',
    'NOx(GT)': 'Concentration (ppb)',
    'NO2(GT)': 'Concentration (µg/m³)'
}

for pollutant, metrics in results.items():
    plt.figure(figsize=(12,5))
    
    # Plot the true measured concentrations
    plt.plot(test_dates, metrics['y_test'].values, label='Actual', linewidth=2)
    
    # Plot the model’s predicted concentrations
    plt.plot(test_dates, metrics['y_pred'], linestyle='--', label='Predicted')
    
    # Format the x-axis to show date labels every 2 days
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Add title and axis labels
    plt.title(f"{pollutant}: Actual vs Predicted Concentration for Next Month")
    plt.xlabel("Date")
    plt.ylabel(pollutant_units.get(pollutant, "Concentration"))
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
