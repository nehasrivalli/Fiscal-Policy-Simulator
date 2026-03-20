import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

economic_indicators = [
    'GDP',
    'Government_Spendings', 
    'Exchange_Rates', 
    'Interest_Rates',
    'Tax_Rates', 
    'Inflation_Rates', 
    'Unemployment_Rates'
]

def load_and_preprocess_data(file_path='"C:\project\data\expanded_var_data.csv"'):
    """Load and preprocess the economic data"""
    print("\nLoading and preprocessing data...")
    
    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Sort by date to ensure time order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Define feature columns (excluding GDP as it's now an indicator)
    feature_columns = [
        'Government_Spendings', 'Exchange_Rates', 'Interest_Rates',
        'Tax_Rates', 'Inflation_Rates', 'Unemployment_Rates',
        'year', 'quarter'
    ]
    
    # Clean data by removing rows with NaN values
    df_clean = df.dropna(subset=feature_columns + ['GDP'])
    
    return df_clean, feature_columns

def train_and_evaluate_model(indicator, df_clean, feature_columns, n_folds=3):
    print(f"\n{'='*50}")
    print(f"Processing {indicator}")
    print(f"{'='*50}")
    
    if indicator == 'GDP':
        X = df_clean[feature_columns]
        y = df_clean[indicator]
    else:
        X = df_clean[[indicator, 'year', 'quarter']]
        y = df_clean[indicator].shift(-1)
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    # TIME-BASED SPLIT TO PREVENT DATA LEAKAGE
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if indicator == 'GDP':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=6,
            max_features='sqrt',
            random_state=42,
            bootstrap=True,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

    # Cross-validation: no shuffle for time series!
    kf = KFold(n_splits=n_folds, shuffle=False)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')

    print(f"\n{indicator} {n_folds}-Fold Cross-Validation Results:")
    print(f"R² Scores: {cv_scores}")
    print(f"Mean R² Score: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{indicator} Test Set Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    joblib.dump(model, f'models/{indicator}_model.pkl')
    joblib.dump(scaler, f'models/{indicator}_scaler.pkl')
    print(f"Model and scaler saved to 'models/{indicator}_model.pkl' and 'models/{indicator}_scaler.pkl'")

    if indicator == 'GDP':
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Feature Importance in {indicator} Prediction')
        plt.tight_layout()
        plt.savefig(f'plots/{indicator}_feature_importance.png')
        print(f"Feature importance plot saved to 'plots/{indicator}_feature_importance.png'")

    return model, scaler

def generate_future_predictions(df_clean, feature_columns, num_quarters=4):
    """Generate predictions for future quarters for all indicators"""
    print("\nGenerating future predictions...")
    
    # Get latest data
    latest_data = df_clean[feature_columns].iloc[-1]
    latest_quarter = df_clean['quarter'].iloc[-1]
    latest_year = df_clean['year'].iloc[-1]
    
    # Initialize forecast DataFrame
    forecast_df = pd.DataFrame({
        'Quarter': range(1, num_quarters + 1),
        'Year': [(latest_year + (latest_quarter + i - 1) // 4) for i in range(1, num_quarters + 1)],
        'Quarter_Value': [((latest_quarter + i - 1) % 4 + 1) for i in range(1, num_quarters + 1)]
    })
    
    # Generate future quarters data
    future_quarters = []
    for i in range(1, num_quarters + 1):
        future_record = latest_data.copy()
        new_quarter = (latest_quarter + i - 1) % 4 + 1
        new_year = latest_year + (latest_quarter + i - 1) // 4
        future_record['quarter'] = new_quarter
        future_record['year'] = new_year
        future_quarters.append(future_record)
    
    future_df = pd.DataFrame(future_quarters)
    
    # Make predictions for each indicator
    for indicator in economic_indicators:
        # Load model and scaler
        model = joblib.load(f'models/{indicator}_model.pkl')
        scaler = joblib.load(f'models/{indicator}_scaler.pkl')
        
        if indicator == 'GDP':
            # For GDP, use all features
            X_future = future_df
            X_future_scaled = scaler.transform(X_future)
        else:
            # For other indicators, use indicator-specific features
            X_future = future_df[[indicator, 'year', 'quarter']]
            X_future_scaled = scaler.transform(X_future)
        
        # Make predictions
        predictions = model.predict(X_future_scaled)
        forecast_df[f'Predicted_{indicator}'] = predictions
    
    # Save forecast
    forecast_df.to_csv('data/economic_forecast.csv', index=False)
    print("Complete forecast saved to 'data/economic_forecast.csv'")
    
    # Visualize predictions
    plt.figure(figsize=(15, 10))
    for i, indicator in enumerate(economic_indicators, 1):
        plt.subplot((len(economic_indicators) + 1) // 2, 2, i)
        plt.plot(forecast_df['Quarter'], forecast_df[f'Predicted_{indicator}'], marker='o')
        plt.title(f'Predicted {indicator}')
        plt.xlabel('Quarter')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('plots/all_predictions.png')
    print("Prediction plots saved to 'plots/all_predictions.png'")
    
    return forecast_df

def perform_window_analysis(df_clean, feature_columns, min_periods=12, window_size=12):
    """Perform expanding and rolling window analysis"""
    print("\nPerforming window analysis...")
    
    # Expanding window analysis
    print("Performing expanding window analysis...")
    expanding_scores = []
    for i in range(min_periods, len(df_clean)):
        X_window = df_clean[feature_columns].iloc[:i]
        y_window = df_clean['GDP'].iloc[:i]
        # Use all but last as train, last as test
        X_train_window = X_window.iloc[:-1]
        y_train_window = y_window.iloc[:-1]
        X_test_window = X_window.iloc[-1:]
        y_test_window = y_window.iloc[-1:]
        scaler_window = StandardScaler()
        X_train_window_scaled = scaler_window.fit_transform(X_train_window)
        X_test_window_scaled = scaler_window.transform(X_test_window)
        window_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        window_model.fit(X_train_window_scaled, y_train_window)
        y_pred_window = window_model.predict(X_test_window_scaled)
        # Only compute R2 if more than one sample in test, else use np.nan
        score = r2_score(y_test_window, y_pred_window) if len(y_test_window) > 1 else np.nan
        expanding_scores.append(score)
    
    # Rolling window analysis
    print("Performing rolling window analysis...")
    rolling_scores = []
    for i in range(window_size, len(df_clean)):
        X_window = df_clean[feature_columns].iloc[i-window_size:i]
        y_window = df_clean['GDP'].iloc[i-window_size:i]
        X_train_window = X_window.iloc[:-1]
        y_train_window = y_window.iloc[:-1]
        X_test_window = X_window.iloc[-1:]
        y_test_window = y_window.iloc[-1:]
        scaler_window = StandardScaler()
        X_train_window_scaled = scaler_window.fit_transform(X_train_window)
        X_test_window_scaled = scaler_window.transform(X_test_window)
        window_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        window_model.fit(X_train_window_scaled, y_train_window)
        y_pred_window = window_model.predict(X_test_window_scaled)
        score = r2_score(y_test_window, y_pred_window) if len(y_test_window) > 1 else np.nan
        rolling_scores.append(score)
    
    # Visualize window analysis results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(expanding_scores)
    plt.title('Expanding Window Analysis\nModel Performance Over Time')
    plt.xlabel('Number of Periods')
    plt.ylabel('R² Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(rolling_scores)
    plt.title('Rolling Window Analysis\nModel Performance Over Time')
    plt.xlabel('Window Position')
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig('plots/window_analysis.png')
    print("Window analysis plots saved to 'plots/window_analysis.png'")

def main():
    """Main function to run the economic forecasting pipeline"""
    print("\n" + "="*70)
    print("ECONOMIC INDICATORS FORECASTING PIPELINE")
    print("="*70)
    
    # Load and preprocess data
    df_clean, feature_columns = load_and_preprocess_data()
    
    # Train and evaluate models for all indicators
    for indicator in economic_indicators:
        train_and_evaluate_model(indicator, df_clean, feature_columns)
    
    # Generate future predictions
    generate_future_predictions(df_clean, feature_columns)
    
    # Perform window analysis
    perform_window_analysis(df_clean, feature_columns)
    
    print("\n" + "="*70)
    print("FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()