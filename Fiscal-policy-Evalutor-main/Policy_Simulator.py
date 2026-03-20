import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import stats

class PolicySimulator:
    """
    A class for simulating the effects of policy changes on economic indicators
    using the ensemble model and sensitivity analysis.
    
    This simulator uses the trained ensemble model to predict how changes in policy
    variables (government spending, tax rates, interest rates) would affect
    economic indicators like GDP, inflation, and unemployment.
    """
    
    def __init__(self):
        """
        Initialize the PolicySimulator by loading the necessary models and data.
        """
        # Load the data
        self.df = pd.read_csv('data/expanded_var_data.csv')
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create time-based features and economic indicators
        self._prepare_data()
        
        # Load the saved models
        print("\nLoading saved models...")
        try:
            # Load Random Forest model
            self.rf_model = joblib.load('rf_gdp_prediction_model.pkl')
            print("Random Forest model loaded successfully")
            
            # Load XGBoost model
            self.xgb_model = joblib.load('xgb_gdp_prediction_model.pkl')
            print("XGBoost model loaded successfully")
            
            # Load LSTM model and scalers
            self.lstm_model = load_model('lstm_gdp_prediction_model.h5')
            self.lstm_feature_scaler = joblib.load('lstm_feature_scaler.pkl')
            self.lstm_target_scaler = joblib.load('lstm_target_scaler.pkl')
            print("LSTM model and scalers loaded successfully")
            
            # Load standard scaler for RF and XGBoost models
            self.scaler = joblib.load('rf_feature_scaler.pkl')
            print("Feature scaler loaded successfully")
            
            # Load ensemble weights
            self.ensemble_weights = joblib.load('ensemble_weights.pkl')
            print("Ensemble weights loaded successfully")
            
            self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run the Ensemble_model.py script first to train and save the models.")
            self.models_loaded = False
            return
        
        # Define feature columns
        self.base_features = ['Government_Spendings', 'Exchange_Rates', 'Interest_Rates', 
                           'Tax_Rates', 'Inflation_Rates', 'Unemployment_Rates', 
                           'National_Budget', 'Public_Debt', 'year', 'quarter']

        self.lag_features = [col for col in self.df.columns if '_lag' in col]
        self.rolling_features = [col for col in self.df.columns if '_rolling_mean' in col]
        self.derived_features = ['Debt_to_GDP_ratio', 'Budget_deficit_to_GDP', 'Real_Interest_Rate', 'Unemployment_Gap']

        self.feature_columns = self.base_features + self.lag_features + self.rolling_features + self.derived_features
        
        # Define policy variables (variables that can be directly changed by policy)
        self.policy_variables = {
            'Government_Spendings': {'name': 'Government Spending', 'unit': 'billion $'},
            'Tax_Rates': {'name': 'Tax Rate', 'unit': '%'},
            'Interest_Rates': {'name': 'Interest Rate', 'unit': '%'}
        }
        
        # Define target variables (economic indicators we want to predict)
        self.target_variables = {
            'GDP': {'name': 'GDP', 'unit': 'billion $'},
            'Inflation_Rates': {'name': 'Inflation Rate', 'unit': '%'},
            'Unemployment_Rates': {'name': 'Unemployment Rate', 'unit': '%'},
            'Public_Debt': {'name': 'Public Debt', 'unit': 'billion $'}
        }
        
        # Get the latest economic data
        self.latest_data = self.df_clean[self.feature_columns + ['date', 'GDP']].sort_values('date').iloc[-1].copy()
        
        print("\nPolicy Simulator initialized successfully!")
    
    def _prepare_data(self):
        """
        Prepare the data by creating time-based features and economic indicators.
        """
        # Extract useful features
        self.df['year'] = self.df['date'].dt.year
        self.df['quarter'] = self.df['date'].dt.quarter

        # Create additional time series features
        # 1. Lag features (previous quarter values)
        for col in ['GDP', 'Government_Spendings', 'Interest_Rates', 'Inflation_Rates', 'Unemployment_Rates']:
            self.df[f'{col}_lag1'] = self.df[col].shift(1)
            self.df[f'{col}_lag2'] = self.df[col].shift(2)

        # 2. Rolling statistics (mean over last year)
        for col in ['GDP', 'Government_Spendings', 'Interest_Rates']:
            self.df[f'{col}_rolling_mean'] = self.df[col].rolling(window=4).mean()

        # 3. Calculate additional economic indicators
        # Debt-to-GDP ratio
        self.df['Debt_to_GDP_ratio'] = self.df['Public_Debt'] / self.df['GDP']

        # Budget deficit as % of GDP
        self.df['Budget_deficit_to_GDP'] = self.df['National_Budget'] / self.df['GDP'] * 100

        # Real interest rates (interest - inflation)
        self.df['Real_Interest_Rate'] = self.df['Interest_Rates'] - self.df['Inflation_Rates']

        # Unemployment gap (using 5% as a proxy for natural rate)
        self.df['Unemployment_Gap'] = self.df['Unemployment_Rates'] - 5.0
        
        # Filter out rows with any remaining NaN values
        self.df_clean = self.df.dropna()
    
    def _create_sequences(self, data, seq_length):
        """
        Create sequences for LSTM model.
        """
        X = []
        for i in range(len(data) - seq_length + 1):
            X.append(data[i:i + seq_length])
        return np.array(X)
    
    def simulate_policy(self, policy_changes, num_quarters=4):
        """
        Simulate the effects of policy changes over a specified number of quarters.
        
        Parameters:
        -----------
        policy_changes : dict
            Dictionary with policy variables as keys and their changes as values.
            Example: {'Tax_Rates': 2.0, 'Government_Spendings': 100}
        num_quarters : int
            Number of quarters to forecast (default: 4)
            
        Returns:
        --------
        dict
            Dictionary with baseline and policy scenarios for each target variable.
        """
        if not self.models_loaded:
            print("Models not loaded. Please run the Ensemble_model.py script first.")
            return None
        
        # Create baseline scenario (no policy changes)
        baseline_scenario = self._forecast_baseline(num_quarters)
        
        # Create policy scenario
        policy_scenario = self._forecast_policy(policy_changes, num_quarters)
        
        # Calculate differences
        differences = {}
        for target in self.target_variables.keys():
            differences[target] = []
            for i in range(num_quarters):
                diff = policy_scenario[target][i] - baseline_scenario[target][i]
                differences[target].append(diff)
        
        return {
            'baseline': baseline_scenario,
            'policy': policy_scenario,
            'differences': differences,
            'quarters': [i+1 for i in range(num_quarters)],
            'dates': baseline_scenario['dates']
        }
    
    def _forecast_baseline(self, num_quarters):
        """
        Forecast economic indicators for the baseline scenario (no policy changes).
        """
        # Start with the latest data
        current_data = self.latest_data.copy()
        
        # Initialize results dictionary
        results = {target: [] for target in self.target_variables.keys()}
        results['dates'] = []
        
        # For LSTM, we need the last sequence from the data
        X = self.df_clean[self.feature_columns].values
        X_lstm = self.lstm_feature_scaler.transform(X)
        seq_length = 4  # Same as used in LSTM_model.py
        last_sequence = X_lstm[-seq_length:].reshape(1, seq_length, len(self.feature_columns))
        current_sequence = last_sequence.copy()
        
        # Forecast for each quarter
        for i in range(num_quarters):
            # Update date and time features
            new_quarter = (current_data['quarter'] + 1) % 4
            if new_quarter == 0:
                new_quarter = 4
            new_year = current_data['year'] + (1 if new_quarter == 1 else 0)
            
            current_data['quarter'] = new_quarter
            current_data['year'] = new_year
            current_data['date'] = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
            
            # Store the date
            results['dates'].append(current_data['date'])
            
            # Prepare features for prediction
            X_future = current_data[self.feature_columns].values.reshape(1, -1)
            X_future_scaled = self.scaler.transform(X_future)
            
            # Make predictions with each model
            rf_pred = self.rf_model.predict(X_future_scaled)[0]
            xgb_pred = self.xgb_model.predict(X_future_scaled)[0]
            
            # LSTM prediction
            lstm_pred_scaled = self.lstm_model.predict(current_sequence)[0][0]
            lstm_pred = self.lstm_target_scaler.inverse_transform(
                np.array([[lstm_pred_scaled]])
            )[0][0]
            
            # Create ensemble prediction for GDP
            rf_weight = self.ensemble_weights['random_forest_weight']
            xgb_weight = self.ensemble_weights['xgboost_weight']
            lstm_weight = self.ensemble_weights['lstm_weight']
            
            gdp_pred = rf_weight * rf_pred + xgb_weight * xgb_pred + lstm_weight * lstm_pred
            
            # Store GDP prediction
            results['GDP'].append(gdp_pred)
            
            # Update GDP in current data for next quarter prediction
            current_data['GDP'] = gdp_pred
            current_data['GDP_lag1'] = self.latest_data['GDP']
            current_data['GDP_lag2'] = self.latest_data['GDP_lag1']
            current_data['GDP_rolling_mean'] = (gdp_pred + self.latest_data['GDP'] + 
                                              self.latest_data['GDP_lag1'] + 
                                              self.latest_data['GDP_lag2']) / 4
            
            # Simple forecasts for other target variables (in a real model, these would be more sophisticated)
            # For simplicity, we're using historical relationships with GDP
            # Inflation tends to increase with GDP growth
            gdp_growth = (gdp_pred / self.latest_data['GDP'] - 1) * 100
            inflation_pred = self.latest_data['Inflation_Rates'] + 0.2 * gdp_growth
            results['Inflation_Rates'].append(inflation_pred)
            
            # Unemployment tends to decrease with GDP growth (Okun's Law)
            unemployment_pred = self.latest_data['Unemployment_Rates'] - 0.3 * gdp_growth
            unemployment_pred = max(2.0, unemployment_pred)  # Set a floor of 2%
            results['Unemployment_Rates'].append(unemployment_pred)
            
            # Public debt increases with budget deficits
            debt_pred = self.latest_data['Public_Debt'] * (1 + 0.01)  # Simple 1% growth
            results['Public_Debt'].append(debt_pred)
            
            # Update LSTM sequence for next prediction
            next_features = current_data[self.feature_columns].values.reshape(1, -1)
            next_features_scaled = self.lstm_feature_scaler.transform(next_features)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features_scaled
        
        return results
    
    def _forecast_policy(self, policy_changes, num_quarters):
        """
        Forecast economic indicators for the policy scenario.
        """
        # Start with the latest data
        current_data = self.latest_data.copy()
        
        # Apply policy changes
        for var, change in policy_changes.items():
            if var in self.policy_variables:
                current_data[var] += change
        
        # Initialize results dictionary
        results = {target: [] for target in self.target_variables.keys()}
        results['dates'] = []
        
        # For LSTM, we need the last sequence from the data
        X = self.df_clean[self.feature_columns].values
        X_lstm = self.lstm_feature_scaler.transform(X)
        seq_length = 4  # Same as used in LSTM_model.py
        last_sequence = X_lstm[-seq_length:].reshape(1, seq_length, len(self.feature_columns))
        current_sequence = last_sequence.copy()
        
        # Apply policy changes to the last sequence
        for var, change in policy_changes.items():
            if var in self.policy_variables:
                # Find the index of the variable in feature_columns
                if var in self.feature_columns:
                    idx = self.feature_columns.index(var)
                    # Apply the change to all time steps in the sequence
                    for t in range(seq_length):
                        # Scale the change
                        var_data = np.array([[current_data[var]]])
                        var_data_scaled = self.lstm_feature_scaler.transform(var_data)
                        var_change_scaled = var_data_scaled[0][0] - current_sequence[0, t, idx]
                        current_sequence[0, t, idx] += var_change_scaled
        
        # Forecast for each quarter
        for i in range(num_quarters):
            # Update date and time features
            new_quarter = (current_data['quarter'] + 1) % 4
            if new_quarter == 0:
                new_quarter = 4
            new_year = current_data['year'] + (1 if new_quarter == 1 else 0)
            
            current_data['quarter'] = new_quarter
            current_data['year'] = new_year
            current_data['date'] = pd.Timestamp(f"{new_year}-{3 * new_quarter}-01")
            
            # Store the date
            results['dates'].append(current_data['date'])
            
            # Prepare features for prediction
            X_future = current_data[self.feature_columns].values.reshape(1, -1)
            X_future_scaled = self.scaler.transform(X_future)
            
            # Make predictions with each model
            rf_pred = self.rf_model.predict(X_future_scaled)[0]
            xgb_pred = self.xgb_model.predict(X_future_scaled)[0]
            
            # LSTM prediction
            lstm_pred_scaled = self.lstm_model.predict(current_sequence)[0][0]
            lstm_pred = self.lstm_target_scaler.inverse_transform(
                np.array([[lstm_pred_scaled]])
            )[0][0]
            
            # Create ensemble prediction for GDP
            rf_weight = self.ensemble_weights['random_forest_weight']
            xgb_weight = self.ensemble_weights['xgboost_weight']
            lstm_weight = self.ensemble_weights['lstm_weight']
            
            gdp_pred = rf_weight * rf_pred + xgb_weight * xgb_pred + lstm_weight * lstm_pred
            
            # Store GDP prediction
            results['GDP'].append(gdp_pred)
            
            # Update GDP in current data for next quarter prediction
            current_data['GDP'] = gdp_pred
            current_data['GDP_lag1'] = self.latest_data['GDP']
            current_data['GDP_lag2'] = self.latest_data['GDP_lag1']
            current_data['GDP_rolling_mean'] = (gdp_pred + self.latest_data['GDP'] + 
                                              self.latest_data['GDP_lag1'] + 
                                              self.latest_data['GDP_lag2']) / 4
            
            # Policy effects on other variables
            # Inflation: affected by government spending, interest rates
            gdp_growth = (gdp_pred / self.latest_data['GDP'] - 1) * 100
            gov_spending_effect = 0.1 * (policy_changes.get('Government_Spendings', 0) / self.latest_data['Government_Spendings'])
            interest_rate_effect = -0.2 * policy_changes.get('Interest_Rates', 0)
            inflation_pred = self.latest_data['Inflation_Rates'] + 0.2 * gdp_growth + gov_spending_effect + interest_rate_effect
            results['Inflation_Rates'].append(inflation_pred)
            
            # Unemployment: affected by GDP growth, government spending
            gov_employment_effect = -0.1 * (policy_changes.get('Government_Spendings', 0) / self.latest_data['Government_Spendings'])
            unemployment_pred = self.latest_data['Unemployment_Rates'] - 0.3 * gdp_growth + gov_employment_effect
            unemployment_pred = max(2.0, unemployment_pred)  # Set a floor of 2%
            results['Unemployment_Rates'].append(unemployment_pred)
            
            # Public debt: affected by government spending, tax rates
            spending_effect = policy_changes.get('Government_Spendings', 0) * 4  # Annual effect
            tax_effect = -policy_changes.get('Tax_Rates', 0) * self.latest_data['GDP'] * 0.01  # Approximate tax revenue change
            debt_pred = self.latest_data['Public_Debt'] * (1 + 0.01) + spending_effect - tax_effect
            results['Public_Debt'].append(debt_pred)
            
            # Update LSTM sequence for next prediction
            next_features = current_data[self.feature_columns].values.reshape(1, -1)
            next_features_scaled = self.lstm_feature_scaler.transform(next_features)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features_scaled
        
        return results
    
    def visualize_simulation(self, policy_changes, num_quarters=4):
        """
        Visualize the simulated effects of policy changes.
        
        Parameters:
        -----------
        policy_changes : dict
            Dictionary with policy variables as keys and their changes as values.
        num_quarters : int
            Number of quarters to forecast (default: 4)
        """
        # Run the simulation
        results = self.simulate_policy(policy_changes, num_quarters)
        
        if results is None:
            return
        
        # Create a figure with subplots for each target variable
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Policy Simulation: Effects of Policy Changes', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot each target variable
        for i, (target, info) in enumerate(self.target_variables.items()):
            ax = axes[i]
            
            # Get the historical data for context
            historical_dates = self.df['date']
            historical_values = self.df[target]
            
            # Plot historical data
            ax.plot(historical_dates, historical_values, 'b-', alpha=0.3, label='Historical')
            
            # Plot baseline and policy scenarios
            ax.plot(results['dates'], results['baseline'][target], 'b--', marker='o', label='Baseline Forecast')
            ax.plot(results['dates'], results['policy'][target], 'r-', marker='s', label='Policy Forecast')
            
            # Add labels and title
            ax.set_title(f'{info["name"]} ({info["unit"]})')
            ax.set_xlabel('Date')
            ax.set_ylabel(info['name'])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Focus on the forecast period
            ax.set_xlim(historical_dates.iloc[-8], results['dates'][-1])
            
            # Add text showing the difference at the end
            final_diff = results['differences'][target][-1]
            final_diff_pct = (final_diff / results['baseline'][target][-1]) * 100
            color = 'green' if (target == 'GDP' and final_diff > 0) or \
                              (target in ['Inflation_Rates', 'Unemployment_Rates', 'Public_Debt'] and final_diff < 0) \
                    else 'red'
            
            ax.text(0.05, 0.95, 
                   f'Final change: {final_diff:.2f} {info["unit"]} ({final_diff_pct:.2f}%)',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add policy changes as text in the figure
        policy_text = "Policy Changes:\n"
        for var, change in policy_changes.items():
            if var in self.policy_variables:
                info = self.policy_variables[var]
                policy_text += f"• {info['name']}: {'+' if change > 0 else ''}{change} {info['unit']}\n"
        
        fig.text(0.5, 0.02, policy_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('policy_simulation.png')
        plt.show()
        
        # Create a summary table
        summary_data = []
        for target, info in self.target_variables.items():
            baseline_final = results['baseline'][target][-1]
            policy_final = results['policy'][target][-1]
            diff_final = results['differences'][target][-1]
            diff_pct = (diff_final / baseline_final) * 100
            
            summary_data.append({
                'Indicator': info['name'],
                'Unit': info['unit'],
                'Baseline': f"{baseline_final:.2f}",
                'Policy': f"{policy_final:.2f}",
                'Difference': f"{diff_final:.2f}",
                'Percent Change': f"{diff_pct:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nPolicy Simulation Summary:")
        print(summary_df.to_string(index=False))
        
        # Provide an overall assessment
        gdp_change = float(summary_data[0]['Difference'])
        inflation_change = float(summary_data[1]['Difference'])
        unemployment_change = float(summary_data[2]['Difference'])
        debt_change = float(summary_data[3]['Difference'])
        
        print("\nPolicy Assessment:")
        if gdp_change > 0:
            print(f"✓ GDP is projected to increase by {gdp_change:.2f} {self.target_variables['GDP']['unit']}")
        else:
            print(f"✗ GDP is projected to decrease by {-gdp_change:.2f} {self.target_variables['GDP']['unit']}")
            
        if inflation_change < 0:
            print(f"✓ Inflation is projected to decrease by {-inflation_change:.2f} {self.target_variables['Inflation_Rates']['unit']}")
        else:
            print(f"✗ Inflation is projected to increase by {inflation_change:.2f} {self.target_variables['Inflation_Rates']['unit']}")
            
        if unemployment_change < 0:
            print(f"✓ Unemployment is projected to decrease by {-unemployment_change:.2f} {self.target_variables['Unemployment_Rates']['unit']}")
        else:
            print(f"✗ Unemployment is projected to increase by {unemployment_change:.2f} {self.target_variables['Unemployment_Rates']['unit']}")
            
        if debt_change < 0:
            print(f"✓ Public debt is projected to decrease by {-debt_change:.2f} {self.target_variables['Public_Debt']['unit']}")
        else:
            print(f"✗ Public debt is projected to increase by {debt_change:.2f} {self.target_variables['Public_Debt']['unit']}")
        
        # Overall recommendation
        positive_count = sum(1 for change in [gdp_change, -inflation_change, -unemployment_change, -debt_change] if change > 0)
        
        print("\nOverall Recommendation:")
        if positive_count >= 3:
            print("This policy change is RECOMMENDED as it positively affects most economic indicators.")
        elif positive_count == 2:
            print("This policy change has MIXED EFFECTS with equal positive and negative outcomes.")
        else:
            print("This policy change is NOT RECOMMENDED as it negatively affects most economic indicators.")
        
        return summary_df
    
    def sensitivity_analysis(self, policy_variable, range_min, range_max, steps=10, num_quarters=4):
        """
        Perform sensitivity analysis for a single policy variable.
        
        Parameters:
        -----------
        policy_variable : str
            The policy variable to analyze (e.g., 'Tax_Rates')
        range_min : float
            Minimum value of the change to test
        range_max : float
            Maximum value of the change to test
        steps : int
            Number of steps between min and max (default: 10)
        num_quarters : int
            Number of quarters to forecast (default: 4)
            
        Returns:
        --------
        dict
            Dictionary with sensitivity analysis results
        """
        if not self.models_loaded:
            print("Models not loaded. Please run the Ensemble_model.py script first.")
            return None
        
        if policy_variable not in self.policy_variables:
            print(f"Invalid policy variable: {policy_variable}")
            print(f"Available policy variables: {list(self.policy_variables.keys())}")
            return None
        
        # Generate the range of values to test
        changes = np.linspace(range_min, range_max, steps)
        
        # Initialize results dictionary
        results = {target: [] for target in self.target_variables.keys()}
        results['changes'] = changes
        
        # Run simulations for each change value
        for change in changes:
            policy_changes = {policy_variable: change}
            sim_results = self.simulate_policy(policy_changes, num_quarters)
            
            # Store the final quarter results
            for target in self.target_variables.keys():
                results[target].append(sim_results['differences'][target][-1])
        
        # Visualize the sensitivity analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sensitivity Analysis: Impact of {self.policy_variables[policy_variable]["name"]} Changes', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot each target variable
        for i, (target, info) in enumerate(self.target_variables.items()):
            ax = axes[i]
            
            # Plot the sensitivity curve
            ax.plot(changes, results[target], 'b-', marker='o')
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # Add labels and title
            ax.set_title(f'Impact on {info["name"]}')
            ax.set_xlabel(f'Change in {self.policy_variables[policy_variable]["name"]} ({self.policy_variables[policy_variable]["unit"]})')
            ax.set_ylabel(f'Change in {info["name"]} ({info["unit"]})')
            ax.grid(True, alpha=0.3)
            
            # Find the optimal point (where the curve crosses zero or reaches its maximum/minimum)
            if target == 'GDP':
                # For GDP, we want to maximize it
                optimal_idx = np.argmax(results[target])
                optimal_change = changes[optimal_idx]
                optimal_effect = results[target][optimal_idx]
                ax.plot(optimal_change, optimal_effect, 'go', markersize=10)
                ax.text(optimal_change, optimal_effect, f'  Optimal: {optimal_change:.2f}')

# Add this code at the end of the file
if __name__ == "__main__":
    # Initialize the policy simulator
    simulator = PolicySimulator()
    
    # Define policy changes to simulate
    policy_changes = {
        'Government_Spendings': 100.0,  # Increase government spending by 100 billion
        'Tax_Rates': -1.0,              # Decrease tax rates by 1 percentage point
        'Interest_Rates': -0.5          # Decrease interest rates by 0.5 percentage points
    }
    
    # Visualize the simulation results
    simulator.visualize_simulation(policy_changes, num_quarters=8)
    
    # Perform sensitivity analysis on government spending
    print("\nPerforming sensitivity analysis on Government Spending...")
    simulator.sensitivity_analysis('Government_Spendings', -200, 200, steps=10, num_quarters=4)