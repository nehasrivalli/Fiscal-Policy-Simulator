import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('data/expanded_var_data.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Dataset shape: {df.shape}")

# Basic data preparation
if df['Exchange_Rates'].isna().sum() > 0:
    df['Exchange_Rates'].fillna(df['Exchange_Rates'].mean(), inplace=True)

df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter

# Define features (matching Random Forest model)
feature_columns = ['Government_Spendings', 'Exchange_Rates', 'Interest_Rates',
                  'Tax_Rates', 'Inflation_Rates', 'Unemployment_Rates', 
                  'National_Budget', 'Public_Debt', 'year', 'quarter']

# Clean data
df_clean = df.dropna(subset=feature_columns + ['GDP'])

# Load models
print("\nLoading models...")
rf_model = joblib.load('rf_gdp_prediction_model.pkl')
print("Random Forest model loaded successfully")

# You can create an XGBoost model with the same features
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Prepare data for XGBoost training
X = df_clean[feature_columns]
y = df_clean['GDP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("\nTraining XGBoost model with the same features as Random Forest...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Save the XGBoost model
joblib.dump(xgb_model, 'xgb_gdp_prediction_model.pkl')
print("XGBoost model saved successfully")

# Save the scaler
joblib.dump(scaler, 'rf_feature_scaler.pkl')
print("Feature scaler saved successfully")

# Evaluate both models
X_scaled = scaler.transform(df_clean[feature_columns])
rf_predictions = rf_model.predict(X_scaled)
xgb_predictions = xgb_model.predict(X_scaled)

# Calculate metrics
rf_mse = mean_squared_error(df_clean['GDP'], rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(df_clean['GDP'], rf_predictions)

xgb_mse = mean_squared_error(df_clean['GDP'], xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(df_clean['GDP'], xgb_predictions)

print("\nModel Evaluation:")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
print(f"XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")

# Create ensemble predictions
total_r2 = rf_r2 + xgb_r2
rf_weight = rf_r2 / total_r2
xgb_weight = xgb_r2 / total_r2

ensemble_predictions = (
    rf_weight * rf_predictions +
    xgb_weight * xgb_predictions
)

# Evaluate ensemble
ensemble_mse = mean_squared_error(df_clean['GDP'], ensemble_predictions)
ensemble_rmse = np.sqrt(ensemble_mse)
ensemble_r2 = r2_score(df_clean['GDP'], ensemble_predictions)

print("\nEnsemble Model Evaluation:")
print(f"RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.4f}")
print(f"Weights - RF: {rf_weight:.4f}, XGB: {xgb_weight:.4f}")

# Visualize predictions
plt.figure(figsize=(14, 8))
plt.scatter(df_clean['GDP'], rf_predictions, alpha=0.3, label='Random Forest')
plt.scatter(df_clean['GDP'], xgb_predictions, alpha=0.3, label='XGBoost')
plt.scatter(df_clean['GDP'], ensemble_predictions, alpha=0.5, label='Ensemble')
plt.plot([df_clean['GDP'].min(), df_clean['GDP'].max()], 
         [df_clean['GDP'].min(), df_clean['GDP'].max()], 'r--')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ensemble_comparison.png')

# GDP Forecasting for Next 4 Quarters
print("\nGenerating GDP forecast for next 4 quarters...")

# Get the most recent record
latest_data = df_clean[feature_columns + ['date']].sort_values('date').iloc[-1].copy()

# Prepare future feature set
future_quarters = []
for i in range(1, 5):
    future_record = latest_data.copy()
    new_quarter = (latest_data['quarter'] + i - 1) % 4 + 1
    new_year = latest_data['year'] + (latest_data['quarter'] + i - 1) // 4
    future_record['quarter'] = new_quarter
    future_record['year'] = new_year
    future_record['date'] = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
    future_quarters.append(future_record)

# Create DataFrame and prepare predictions
future_df = pd.DataFrame(future_quarters)
X_future = future_df[feature_columns]
X_future_scaled = scaler.transform(X_future)

# Get predictions
rf_future_preds = rf_model.predict(X_future_scaled)
xgb_future_preds = xgb_model.predict(X_future_scaled)
ensemble_future_preds = (
    rf_weight * rf_future_preds +
    xgb_weight * xgb_future_preds
)

# Assemble forecast DataFrame
forecast_df = future_df[['date']].copy()
forecast_df['RF_Predicted_GDP'] = rf_future_preds
forecast_df['XGB_Predicted_GDP'] = xgb_future_preds
forecast_df['Ensemble_Predicted_GDP'] = ensemble_future_preds

# Save forecast
forecast_df.to_csv('ensemble_gdp_forecast.csv', index=False)

# Plot GDP Forecast
plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['GDP'], label='Historical GDP')
plt.plot(forecast_df['date'], forecast_df['RF_Predicted_GDP'], marker='o', 
         linestyle='--', alpha=0.6, label='RF Forecast')
plt.plot(forecast_df['date'], forecast_df['XGB_Predicted_GDP'], marker='s', 
         linestyle='--', alpha=0.6, label='XGB Forecast')
plt.plot(forecast_df['date'], forecast_df['Ensemble_Predicted_GDP'], marker='D', 
         linestyle='-', linewidth=2, label='Ensemble Forecast')
plt.title('GDP Forecast for Next 4 Quarters')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ensemble_gdp_forecast.png')

print("\nGDP forecast complete. Results saved to 'ensemble_gdp_forecast.csv' and plot saved as 'ensemble_gdp_forecast.png'")