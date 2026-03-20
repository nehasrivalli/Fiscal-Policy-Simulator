
# Fiscal Forecaster and Policy Evaluator

An end-to-end fiscal forecasting and simulation engine designed to predict key macroeconomic indicators and evaluate the potential impact of fiscal policy changes using machine learning and causal inference techniques.

## Project Overview

This project consists of two main phases:

### Phase 1: Economic Forecasting

Predicts key macroeconomic indicators (GDP, Inflation, Unemployment, Public Debt) using time-series enhanced machine learning models:

- Random Forest model
- XGBoost model
- LSTM model
- Ensemble model (weighted combination of the above)

The models incorporate advanced time series features:
- Lag features (previous quarter values)
- Rolling statistics (mean over last year)
- Derived economic indicators (Debt-to-GDP ratio, Budget deficit as % of GDP, Real interest rates, Unemployment gap)

### Phase 2: Policy Evaluation

Evaluates the potential impact of fiscal policy changes using two complementary approaches:

1. **Causal Forest DML**: Estimates the causal effect of policy changes on economic indicators
2. **Policy Simulation**: Simulates the effects of policy changes over time using the ensemble model

## Project Structure

- `Random_forest.py`: Implements and trains the Random Forest model for GDP prediction
- `XGBoost_model.py`: Implements and trains the XGBoost model for GDP prediction
- `LSTM_model.py`: Implements and trains the LSTM neural network model for GDP prediction
- `Ensemble_model.py`: Combines the three models using weighted averaging based on performance
- `Policy_Evaluator.py`: Implements causal inference using CausalForestDML to evaluate policy impacts
- `Policy_Simulator.py`: Simulates the effects of policy changes over time using the ensemble model
- `data/expanded_var_data.csv`: Dataset containing economic indicators

## Usage Instructions

### Phase 1: Economic Forecasting

1. Run the individual model scripts to train and save the models:

```bash
python Random_forest.py
python XGBoost_model.py
python LSTM_model.py
```

2. Run the ensemble model to combine predictions and generate forecasts:

```bash
python Ensemble_model.py
```

This will generate:
- Model evaluation metrics
- GDP forecasts for the next 4 quarters
- Visualizations comparing model performance

### Phase 2: Policy Evaluation

1. For causal inference-based policy evaluation:

```bash
python Policy_Evaluator.py
```

This will:
- Train causal forest models for each target-treatment pair
- Evaluate the impact of example policy changes
- Generate policy recommendations based on target improvements

2. For simulation-based policy evaluation:

```bash
python Policy_Simulator.py
```

This will:
- Simulate the effects of policy changes over time
- Visualize the projected impact on economic indicators
- Provide policy recommendations based on the simulation results

## Example Policy Evaluation

You can evaluate custom policy changes by modifying the policy parameters in the scripts. For example:

```python
# In Policy_Evaluator.py or Policy_Simulator.py
policy_changes = {
    'Tax_Rates': 2.0,  # Increase tax rates by 2 percentage points
    'Government_Spendings': 100.0,  # Increase government spending by 100 units
    'Interest_Rates': -0.5  # Decrease interest rates by 0.5 percentage points
}

# Evaluate and visualize the impact
evaluator.visualize_policy_impact(policy_changes)
# or
simulator.visualize_simulation(policy_changes)
```

## Methodology

### Economic Forecasting

- **Random Forest**: Ensemble of decision trees that captures non-linear relationships
- **XGBoost**: Gradient boosting algorithm that builds trees sequentially
- **LSTM**: Deep learning model that captures temporal dependencies in time series data
- **Ensemble**: Weighted average of the three models based on their R² scores

### Policy Evaluation

- **Causal Forest DML**: Double Machine Learning approach that estimates heterogeneous treatment effects
- **Policy Simulation**: Uses the trained ensemble model to simulate the effects of policy changes over time

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow
- matplotlib
- seaborn
- econml (for causal inference)

## Future Improvements

- Incorporate Bayesian structural time series for causal inference
- Add Prophet or ARIMA models for time series forecasting
- Implement sensitivity analysis framework to test policy robustness
- Develop a web interface for interactive policy evaluation
