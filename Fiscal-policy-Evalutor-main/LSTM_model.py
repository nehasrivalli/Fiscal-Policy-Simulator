import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Create additional time series features
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

# Function to create sequences for LSTM
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# Split the data into features and target
X = df_clean[feature_columns].values
y = df_clean['GDP'].values

# Scale the features and target
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Define sequence length (number of time steps to look back)
seq_length = 4  # One year (4 quarters)

# Create sequences for LSTM
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# Build the LSTM model
print("\nBuilding LSTM model...")
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_training_history.png')

# Visualize actual vs predicted GDP
plt.figure(figsize=(12, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP (LSTM)')
plt.tight_layout()
plt.savefig('lstm_actual_vs_predicted.png')

# Analyze residuals
residuals = y_test_actual - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted GDP')
plt.ylabel('Residuals')
plt.title('Residual Analysis (LSTM)')
plt.tight_layout()
plt.savefig('lstm_residuals.png')

# Save the model and scalers
model.save('lstm_gdp_prediction_model.h5')
joblib.dump(feature_scaler, 'lstm_feature_scaler.pkl')
joblib.dump(target_scaler, 'lstm_target_scaler.pkl')

print("\nLSTM model saved as 'lstm_gdp_prediction_model.h5'")
print("Feature scaler saved as 'lstm_feature_scaler.pkl'")
print("Target scaler saved as 'lstm_target_scaler.pkl'")

# ---- GDP Forecasting for Next 4 Quarters ----

print("\nGenerating GDP forecast for next 4 quarters...")

# Get the most recent data points for sequence creation
latest_data = df_clean[feature_columns].values[-seq_length:]
latest_data_scaled = feature_scaler.transform(latest_data)

# Reshape for LSTM input
latest_sequence = latest_data_scaled.reshape(1, seq_length, len(feature_columns))

# Prepare future feature set
future_quarters = []
latest_date = df_clean['date'].iloc[-1]

for i in range(1, 5):  # Forecast for next 4 quarters
    new_quarter = (latest_date.quarter + i - 1) % 4 + 1
    new_year = latest_date.year + (latest_date.quarter + i - 1) // 4
    future_date = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
    future_quarters.append(future_date)

# Predict future GDP (one step at a time for multi-step forecasting)
future_gdp_preds = []
current_sequence = latest_sequence.copy()

for _ in range(4):
    # Predict the next value
    next_pred_scaled = model.predict(current_sequence)[0][0]
    future_gdp_preds.append(next_pred_scaled)
    
    # Update the sequence for the next prediction
    # For simplicity, we're assuming the features remain constant
    # In a more sophisticated model, you would predict these features as well
    next_features = current_sequence[0, -1, :].copy()
    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, :] = next_features

# Convert predictions back to original scale
future_gdp_preds = target_scaler.inverse_transform(np.array(future_gdp_preds).reshape(-1, 1)).flatten()

# Assemble forecast DataFrame
forecast_df = pd.DataFrame({
    'date': future_quarters,
    'Predicted_GDP': future_gdp_preds
})

# Save forecast to CSV
forecast_df.to_csv('lstm_gdp_forecast.csv', index=False)

# Plot GDP Forecast
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['GDP'], label='Historical GDP')
plt.plot(forecast_df['date'], forecast_df['Predicted_GDP'], marker='o', linestyle='--', label='Forecasted GDP (LSTM)')
plt.title('GDP Forecast for Next 4 Quarters (LSTM)')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_gdp_forecast.png')

print("\nGDP forecast complete. Forecast saved to 'lstm_gdp_forecast.csv' and plot saved as 'lstm_gdp_forecast.png'")
print(f"R² Score: {r2:.4f}")