#policy evalutor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference

class PolicyEvaluator:
    def __init__(self, data_path='/content/expanded_var_data.csv'):
        """
        Initialize the Policy Evaluator with the dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset containing economic indicators
        """
        # Load the data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create time-based features
        self.df['year'] = self.df['date'].dt.year
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Create additional economic indicators
        self._create_economic_indicators()
        
        # Define target variables and treatment variables
        self.target_variables = ['GDP', 'Inflation_Rates', 'Unemployment_Rates', 'Public_Debt']
        self.treatment_variables = ['Government_Spendings', 'Tax_Rates', 'Interest_Rates']
        
        # Define control variables (features that affect outcomes but aren't policy levers)
        self.control_variables = [
            'Exchange_Rates', 'year', 'quarter', 'GDP_lag1', 'GDP_lag2',
            'Government_Spendings_lag1', 'Interest_Rates_lag1', 'Inflation_Rates_lag1',
            'Unemployment_Rates_lag1', 'GDP_rolling_mean', 'Government_Spendings_rolling_mean',
            'Interest_Rates_rolling_mean', 'Debt_to_GDP_ratio', 'Budget_deficit_to_GDP',
            'Real_Interest_Rate', 'Unemployment_Gap'
        ]
        
        # Initialize models dictionary
        self.models = {}
        
    def _create_economic_indicators(self):
        """
        Create additional economic indicators and time series features.
        """
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
        
        # Drop rows with NaN values
        self.df = self.df.dropna()
    
    def train_causal_models(self):
        """
        Train causal forest models for each target variable and treatment variable combination.
        """
        print("Training causal forest models...")
        
        for target in self.target_variables:
            self.models[target] = {}
            
            for treatment in self.treatment_variables:
                print(f"Training model for target: {target}, treatment: {treatment}")
                
                # Prepare data
                Y = self.df[target].values
                T = self.df[treatment].values
                X = self.df[self.control_variables].values
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train causal forest model with explicit models for Y and T
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.linear_model import LassoCV
                
                model = CausalForestDML(
                    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                    model_t=LassoCV(cv=5, random_state=42),
                    n_estimators=100,
                    min_samples_leaf=10,
                    max_depth=10,
                    verbose=0,
                    random_state=42
                )
                
                # Fit the model
                model.fit(Y, T, X=X_scaled)
                
                # Store model and scaler
                self.models[target][treatment] = {
                    'model': model,
                    'scaler': scaler
                }
        
        print("All causal models trained successfully!")
    
    def save_models(self, path='causal_models.pkl'):
        """
        Save trained causal models to disk.
        """
        joblib.dump(self.models, path)
        print(f"Causal models saved to {path}")
    
    def load_models(self, path='causal_models.pkl'):
        """
        Load trained causal models from disk.
        """
        self.models = joblib.load(path)
        print(f"Causal models loaded from {path}")
    
    def evaluate_policy(self, policy_changes):
        """
        Evaluate the impact of policy changes on economic indicators.
        
        Parameters:
        -----------
        policy_changes : dict
            Dictionary with treatment variables as keys and their changes as values.
            Example: {'Tax_Rates': 2.0, 'Government_Spendings': 100}
        
        Returns:
        --------
        dict
            Dictionary with predicted impacts on each target variable.
        """
        if not self.models:
            raise ValueError("Models not trained or loaded. Call train_causal_models() or load_models() first.")
        
        # Get the latest economic data as baseline
        latest_data = self.df.sort_values('date').iloc[-1:]
        
        # Create baseline and policy scenarios
        baseline = {}
        policy = {}
        
        for target in self.target_variables:
            baseline[target] = latest_data[target].values[0]
            
            # Initialize policy with baseline value
            policy[target] = baseline[target]
            
            # For each treatment that's changing, estimate its effect on this target
            for treatment, change in policy_changes.items():
                if treatment in self.treatment_variables:
                    # Get the model for this target-treatment pair
                    model_dict = self.models[target][treatment]
                    model = model_dict['model']
                    scaler = model_dict['scaler']
                    
                    # Prepare control variables
                    X = latest_data[self.control_variables].values
                    X_scaled = scaler.transform(X)
                    
                    # Current treatment value
                    current_T = latest_data[treatment].values[0]
                    
                    # New treatment value after policy change
                    new_T = current_T + change
                    
                    # Estimate treatment effect
                    effect = model.effect(X_scaled, T0=current_T, T1=new_T)[0]
                    
                    # Update policy prediction
                    policy[target] += effect
        
        # Calculate differences
        differences = {}
        for target in self.target_variables:
            differences[target] = policy[target] - baseline[target]
            differences[f"{target}_percent"] = (differences[target] / baseline[target]) * 100
        
        return {
            'baseline': baseline,
            'policy': policy,
            'differences': differences
        }
    
    def visualize_policy_impact(self, policy_changes, title="Policy Impact Analysis"):
        """
        Visualize the impact of policy changes on economic indicators.
        
        Parameters:
        -----------
        policy_changes : dict
            Dictionary with treatment variables as keys and their changes as values.
        title : str
            Title for the visualization.
        """
        results = self.evaluate_policy(policy_changes)
        
        # Create a DataFrame for visualization
        impact_data = []
        for target in self.target_variables:
            impact_data.append({
                'Indicator': target,
                'Baseline': results['baseline'][target],
                'Policy': results['policy'][target],
                'Absolute Change': results['differences'][target],
                'Percent Change': results['differences'][f"{target}_percent"]
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot each indicator
        for i, target in enumerate(self.target_variables):
            ax = axes[i]
            target_data = impact_df[impact_df['Indicator'] == target]
            
            # Create bar chart
            x = ['Baseline', 'Policy']
            y = [target_data['Baseline'].values[0], target_data['Policy'].values[0]]
            bars = ax.bar(x, y, color=['blue', 'green'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Add percent change as text
            percent_change = target_data['Percent Change'].values[0]
            color = 'green' if percent_change >= 0 else 'red'
            ax.text(0.5, 0.9, f"Change: {percent_change:.2f}%", 
                    transform=ax.transAxes, ha='center', color=color,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title(target)
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add policy changes as text in the figure
        policy_text = "Policy Changes:\n"
        for treatment, change in policy_changes.items():
            policy_text += f"• {treatment}: {'+' if change > 0 else ''}{change}\n"
        
        fig.text(0.5, 0.02, policy_text, ha='center', bbox=dict(facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('policy_impact_analysis.png')
        plt.show()
        
        return impact_df
    
    def recommend_policy(self, target_improvements, constraints=None):
        """
        Recommend policy changes to achieve target improvements in economic indicators.
        
        Parameters:
        -----------
        target_improvements : dict
            Dictionary with target variables as keys and desired improvements as values.
            Example: {'GDP': 200, 'Unemployment_Rates': -0.5}
        constraints : dict, optional
            Dictionary with constraints on treatment variables.
            Example: {'Tax_Rates': {'min': -2, 'max': 2}}
        
        Returns:
        --------
        dict
            Dictionary with recommended policy changes.
        """
        if not self.models:
            raise ValueError("Models not trained or loaded. Call train_causal_models() or load_models() first.")
        
        # Get the latest economic data
        latest_data = self.df.sort_values('date').iloc[-1:]
        
        # Initialize recommended policy changes
        recommended_changes = {treatment: 0 for treatment in self.treatment_variables}
        
        # Set default constraints if not provided
        if constraints is None:
            constraints = {}
            for treatment in self.treatment_variables:
                current_value = latest_data[treatment].values[0]
                constraints[treatment] = {
                    'min': -0.2 * current_value,  # Default: allow 20% decrease
                    'max': 0.2 * current_value    # Default: allow 20% increase
                }
        
        # Simple grid search for policy recommendations
        # This is a simplified approach - in a real system, you might use optimization algorithms
        best_score = float('inf')
        best_policy = None
        
        # Define grid search ranges for each treatment
        grid_points = 5  # Number of points to try for each treatment
        
        # Generate all combinations of policy changes
        from itertools import product
        
        grid_values = {}
        for treatment in self.treatment_variables:
            if treatment in constraints:
                min_val = constraints[treatment].get('min', -0.2 * latest_data[treatment].values[0])
                max_val = constraints[treatment].get('max', 0.2 * latest_data[treatment].values[0])
            else:
                current_value = latest_data[treatment].values[0]
                min_val = -0.2 * current_value
                max_val = 0.2 * current_value
                
            grid_values[treatment] = np.linspace(min_val, max_val, grid_points)
        
        # Generate all combinations
        treatment_names = list(grid_values.keys())
        combinations = list(product(*[grid_values[t] for t in treatment_names]))
        
        # Evaluate each combination
        for combo in combinations:
            policy_changes = {treatment_names[i]: combo[i] for i in range(len(treatment_names))}
            
            # Evaluate this policy
            results = self.evaluate_policy(policy_changes)
            
            # Calculate score based on how well it meets target improvements
            score = 0
            for target, desired_change in target_improvements.items():
                if target in results['differences']:
                    actual_change = results['differences'][target]
                    score += (actual_change - desired_change) ** 2
            
            # Update best policy if this one is better
            if score < best_score:
                best_score = score
                best_policy = policy_changes
        
        return {
            'recommended_changes': best_policy,
            'expected_outcomes': self.evaluate_policy(best_policy)
        }

# Example usage
if __name__ == "__main__":
    # Initialize the policy evaluator
    evaluator = PolicyEvaluator()
    
    # Train causal models
    evaluator.train_causal_models()
    
    # Save models
    evaluator.save_models()
    
    # Example: Evaluate a policy change
    policy_changes = {
        'Tax_Rates': 2.0,  # Increase tax rates by 2 percentage points
        'Government_Spendings': 100.0,  # Increase government spending by 100 units
        'Interest_Rates': -0.5  # Decrease interest rates by 0.5 percentage points
    }
    
    # Visualize the impact
    impact_df = evaluator.visualize_policy_impact(policy_changes, 
                                               title="Impact of Tax Increase, Spending Increase, and Interest Rate Cut")
    print(impact_df)
    
    # Get policy recommendations
    target_improvements = {
        'GDP': 200,  # Aim to increase GDP by 200 units
        'Unemployment_Rates': -0.5  # Aim to decrease unemployment by 0.5 percentage points
    }
    
    recommendations = evaluator.recommend_policy(target_improvements)
    print("\nPolicy Recommendations:")
    for treatment, change in recommendations['recommended_changes'].items():
        print(f"{treatment}: {change:.2f}")
    
    print("\nExpected Outcomes:")
    for target in evaluator.target_variables:
        baseline = recommendations['expected_outcomes']['baseline'][target]
        policy = recommendations['expected_outcomes']['policy'][target]
        diff = recommendations['expected_outcomes']['differences'][target]
        diff_percent = recommendations['expected_outcomes']['differences'][f"{target}_percent"]
        print(f"{target}: {baseline:.2f} -> {policy:.2f} (Change: {diff:.2f}, {diff_percent:.2f}%)")