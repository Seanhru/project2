import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# STEP 1 — Load and prepare data
df = pd.read_csv("AirQualityUCI.csv")

# Keep only the relevant columns
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
df = df[cols]

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert Time ("18:00:00") → numeric hour (0–23)
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Time'].dt.hour
df.drop(columns=['Time'], inplace=True)

# Combine Date + Hour into a single datetime
df['Datetime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
df = df.sort_values('Datetime')

# Interpolate missing values and drop any remaining NaNs
df.interpolate(inplace=True)
df.dropna(inplace=True)

# STEP 2 — Correlation analysis (for insight)
corr = df[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation between Air Pollutants and Climate Conditions")
plt.show()

# STEP 3 — Create lag features for each pollutant
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)']
for pollutant in pollutants:
    for lag in range(1, 25):  # last 24 hours
        df[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)

df.dropna(inplace=True)

# STEP 4 — Prepare features (X) and targets (y)
X = df.drop(columns=pollutants + ['Date', 'Datetime'])
y = df[pollutants]

# Verify all features are numeric
assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1], "Non-numeric data found!"

# STEP 5 — Split into training and test sets (no shuffle since it’s time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# STEP 6 — Train multi-output Random Forest
rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
rf_multi.fit(X_train, y_train)

# STEP 7 — Predict and evaluate
y_pred = rf_multi.predict(X_test)

# Compute metrics for each pollutant
results = {}
for i, col in enumerate(y.columns):
    r2 = r2_score(y_test[col], y_pred[:, i])
    rmse = mean_squared_error(y_test[col], y_pred[:, i], squared=False)
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    results[col] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}

results_df = pd.DataFrame(results).T
print("Model Performance per Pollutant:")
print(results_df)

# STEP 8 — Feature importance (average across all outputs)
importances = np.mean([est.feature_importances_ for est in rf_multi.estimators_], axis=0)
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,6))
feature_importance.head(10).plot(kind='barh')
plt.title("Top 10 Most Important Features for Predicting Air Pollutants")
plt.xlabel("Average Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()

# STEP 9 — Time Series Cross Validation (optional robustness check)
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42),
                            X, y['CO(GT)'], cv=tscv, scoring='r2')
print("Time Series CV R² scores for CO(GT):", cv_scores)
print("Average CV R²:", np.mean(cv_scores))

# STEP 10 — Plot example pollutant predictions
plt.figure(figsize=(10,6))
plt.plot(y_test['CO(GT)'].values, label='Actual CO(GT)')
plt.plot(y_pred[:, 0], label='Predicted CO(GT)')
plt.legend()
plt.title("Actual vs Predicted CO(GT) Concentration")
plt.xlabel("Time (test set)")
plt.ylabel("CO(GT)")
plt.show()
