import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import joblib

# Load the data
df = pd.read_csv('data/expanded_var_data.csv')

# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isna().sum())

# Fill missing values in Exchange_Rates column with the mean
if df['Exchange_Rates'].isna().sum() > 0:
    mean_exchange_rate = df['Exchange_Rates'].mean()
    df['Exchange_Rates'].fillna(mean_exchange_rate, inplace=True)

# Convert date to datetime and extract useful features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter

# Create additional time series features as requested
# 1. Lag features (previous quarter values)
for col in ['GDP', 'Government_Spendings', 'Interest_Rates', 'Inflation_Rates', 'Unemployment_Rates']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# 2. Rolling statistics (mean over last year)
for col in ['GDP', 'Government_Spendings', 'Interest_Rates']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=4).mean()

# 3. Calculate additional economic indicators
# Debt-to-GDP ratio
df['Debt_to_GDP_ratio'] = df['Public_Debt'] / df['GDP']

# Budget deficit as % of GDP
df['Budget_deficit_to_GDP'] = df['National_Budget'] / df['GDP'] * 100

# Real interest rates (interest - inflation)
df['Real_Interest_Rate'] = df['Interest_Rates'] - df['Inflation_Rates']

# Unemployment gap (using 5% as a proxy for natural rate)
df['Unemployment_Gap'] = df['Unemployment_Rates'] - 5.0

# Select features for modeling (excluding GDP which is our target)
base_features = ['Government_Spendings', 'Exchange_Rates', 'Interest_Rates', 
               'Tax_Rates', 'Inflation_Rates', 'Unemployment_Rates', 
               'National_Budget', 'Public_Debt', 'year', 'quarter']

lag_features = [col for col in df.columns if '_lag' in col]
rolling_features = [col for col in df.columns if '_rolling_mean' in col]
derived_features = ['Debt_to_GDP_ratio', 'Budget_deficit_to_GDP', 'Real_Interest_Rate', 'Unemployment_Gap']

feature_columns = base_features + lag_features + rolling_features + derived_features

# Filter out rows with any remaining NaN values (due to lag and rolling features)
df_clean = df.dropna(subset=feature_columns + ['GDP'])

# Split the data into features and target
X = df_clean[feature_columns]
y = df_clean['GDP']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the XGBoost model
print("\nTraining XGBoost model...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Visualization of feature importance (top 15 features)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Features Importance in GDP Prediction (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')

# Visualize actual vs predicted GDP
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_actual_vs_predicted.png')

# Analyze residuals
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted GDP')
plt.ylabel('Residuals')
plt.title('Residual Analysis (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_residuals.png')

# Hyperparameter tuning with GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the model with best parameters
best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train_scaled, y_train)

# Evaluate the tuned model
y_pred_tuned = best_xgb_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\nTuned Model Evaluation:")
print(f"Mean Squared Error: {mse_tuned:.2f}")
print(f"Root Mean Squared Error: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.4f}")

# Save the model
joblib.dump(best_xgb_model, 'xgb_gdp_prediction_model.pkl')
print("\nModel saved as 'xgb_gdp_prediction_model.pkl'")

# Create the PDP plot for top features
top_features = feature_importance['Feature'].head(3).tolist()
top_feature_indices = [feature_columns.index(f) for f in top_features]

# Create the PDP plot
fig, ax = plt.subplots(figsize=(15, 10))
PartialDependenceDisplay.from_estimator(
    estimator=best_xgb_model,
    X=X_train_scaled,
    features=top_feature_indices,
    feature_names=feature_columns,
    ax=ax
)

plt.suptitle('Partial Dependence of GDP on Top Features (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_partial_dependence.png')

print("\nAnalysis and model building complete!")

# ---- GDP Forecasting for Next 4 Quarters ----

print("\nGenerating GDP forecast for next 4 quarters...")

# Get the most recent record from the cleaned dataset
latest_data = df_clean[feature_columns + ['date']].sort_values('date').iloc[-1].copy()

# Prepare future feature set
future_quarters = []
for i in range(1, 5):  # Forecast for next 4 quarters
    future_record = latest_data.copy()
    new_quarter = (latest_data['quarter'] + i - 1) % 4 + 1
    new_year = latest_data['year'] + (latest_data['quarter'] + i - 1) // 4
    future_record['quarter'] = new_quarter
    future_record['year'] = new_year
    future_record['date'] = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
    
    # For simplicity, we're keeping other features constant
    # In a more sophisticated model, you might want to predict these features as well
    future_quarters.append(future_record)

# Create DataFrame
future_df = pd.DataFrame(future_quarters)

# Drop date column for prediction
X_future = future_df[feature_columns]

# Scale features
X_future_scaled = scaler.transform(X_future)

# Predict future GDP
future_gdp_preds = best_xgb_model.predict(X_future_scaled)

# Assemble forecast DataFrame
forecast_df = future_df[['date']].copy()
forecast_df['Predicted_GDP'] = future_gdp_preds

# Save forecast to CSV
forecast_df.to_csv('xgboost_gdp_forecast.csv', index=False)

# Plot GDP Forecast
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['GDP'], label='Historical GDP')
plt.plot(forecast_df['date'], forecast_df['Predicted_GDP'], marker='o', linestyle='--', label='Forecasted GDP (XGBoost)')
plt.title('GDP Forecast for Next 4 Quarters (XGBoost)')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xgboost_gdp_forecast.png')

print("\nGDP forecast complete. Forecast saved to 'xgboost_gdp_forecast.csv' and plot saved as 'xgboost_gdp_forecast.png'")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")