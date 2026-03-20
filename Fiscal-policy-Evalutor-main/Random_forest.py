#random forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv('/content/expanded_var_data.csv')


print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())


print("\nMissing values per column:")
print(df.isna().sum())

if df['Exchange_Rates'].isna().sum() > 0:
    mean_exchange_rate = df['Exchange_Rates'].mean()
    df['Exchange_Rates'].fillna(mean_exchange_rate, inplace=True)


df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter


feature_columns = ['Government_Spendings', 'Exchange_Rates', 'Interest_Rates',
                   'Tax_Rates', 'Inflation_Rates', 'Unemployment_Rates',
                   'National_Budget', 'Public_Debt', 'year', 'quarter']

df_clean = df.dropna(subset=feature_columns + ['GDP'])

X = df_clean[feature_columns]
y = df_clean['GDP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")


feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in GDP Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')


plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')


residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted GDP')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.tight_layout()
plt.savefig('residuals.png')

correlation_matrix = df_clean[feature_columns + ['GDP']].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['GDP'])
plt.title('GDP Time Series')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.tight_layout()
plt.savefig('gdp_time_series.png')
print("\nAnalysis complete. Visualizations saved as PNG files.")

from sklearn.model_selection import GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='r2'
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)


y_pred_tuned = best_rf_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\nTuned Model Evaluation:")
print(f"Mean Squared Error: {mse_tuned:.2f}")
print(f"Root Mean Squared Error: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.4f}")


import joblib
joblib.dump(best_rf_model, 'rf_gdp_prediction_model.pkl')
print("\nModel saved as 'rf_gdp_prediction_model.pkl'")


top_features = feature_importance['Feature'].head(3).tolist()
top_feature_indices = [feature_columns.index(f) for f in top_features]


fig, ax = plt.subplots(figsize=(15, 10))
PartialDependenceDisplay.from_estimator(
    estimator=best_rf_model,
    X=X_train_scaled,
    features=top_feature_indices,
    feature_names=feature_columns,
    ax=ax
)

plt.suptitle('Partial Dependence of GDP on Top Features')
plt.tight_layout()
plt.savefig('partial_dependence.png')

print("\nAnalysis and model building complete!")



print("\nGenerating GDP forecast for next 4 quarters...")


latest_data = df_clean[feature_columns + ['date']].sort_values('date').iloc[-1].copy()


future_quarters = []
for i in range(1, 5):
    future_record = latest_data.copy()
    new_quarter = (latest_data['quarter'] + i - 1) % 4 + 1
    new_year = latest_data['year'] + (latest_data['quarter'] + i - 1) // 4
    future_record['quarter'] = new_quarter
    future_record['year'] = new_year
    future_record['date'] = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
    future_quarters.append(future_record)


future_df = pd.DataFrame(future_quarters)


X_future = future_df[feature_columns]


X_future_scaled = scaler.transform(X_future)


future_gdp_preds = best_rf_model.predict(X_future_scaled)

forecast_df = future_df[['date']].copy()
forecast_df['Predicted_GDP'] = future_gdp_preds

forecast_df.to_csv('gdp_forecast.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['GDP'], label='Historical GDP')
plt.plot(forecast_df['date'], forecast_df['Predicted_GDP'], marker='o', linestyle='--', label='Forecasted GDP')
plt.title('GDP Forecast for Next 4 Quarters')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('gdp_forecast.png')

print("\nGDP forecast complete. Forecast saved to 'gdp_forecast.csv' and plot saved as 'gdp_forecast.png'")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")