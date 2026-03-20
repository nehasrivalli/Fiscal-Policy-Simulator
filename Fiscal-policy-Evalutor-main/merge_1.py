import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Load the Existing Datasets
# -----------------------

# Load quarterly datasets: GDP and Government Spendings
df_gdp = pd.read_csv('data/gdp_fred.csv', parse_dates=['date'])
df_gov = pd.read_csv('data/Government_spendings_Fred.csv', parse_dates=['date'])

# Load daily Exchange Rates
df_exch = pd.read_csv('data/exchange_rates_Fred.csv', parse_dates=['date'])

# Load monthly Interest Rates
df_interest = pd.read_csv('data/Interest_rates_Fred.csv', parse_dates=['date'])

# Load annual Tax Rates
years = pd.date_range(start='1970-01-01', end='2020-12-31', freq='YS')
tax_values = np.random.uniform(70, 80, len(years))  # simulate tax rates
df_tax = pd.DataFrame({'date': years, 'Tax_Rates': tax_values})

# -----------------------
# 2. Load New Datasets
# -----------------------

# Load monthly Inflation Rates
df_inflation = pd.read_csv('data/inflation_rates_Fred.csv', parse_dates=['date'])
# If you don't have this file yet, you would need to create or obtain it

# Load monthly Unemployment Rates
df_unemployment = pd.read_csv('data/unemployment_rate_Fred.csv', parse_dates=['date'])
# If you don't have this file yet, you would need to create or obtain it

# Load quarterly National Budget
df_budget = pd.read_csv('data/Nation_budget_Fred.csv', parse_dates=['date'])
# If you don't have this file yet, you would need to create or obtain it

# Load quarterly Public Debt
df_debt = pd.read_csv('data/public_debt_Fred.csv', parse_dates=['date'])
# If you don't have this file yet, you would need to create or obtain it

# -----------------------
# 3. Establish Quarterly Dates
# -----------------------

# Use the quarterly dates from the GDP data as the reference quarterly dates.
df_gdp.sort_values('date', inplace=True)
quarterly_dates = pd.DataFrame({'date': df_gdp['date'].unique()})
quarterly_dates.sort_values('date', inplace=True)

df_exch['date'] = pd.to_datetime(df_exch['date'], format='%d-%m-%Y')
quarterly_dates['date'] = pd.to_datetime(quarterly_dates['date'])

# -----------------------
# 4. Resample/Align Data to Quarterly Dates
# -----------------------

# --- Exchange Rates (Daily -> Quarterly) ---
df_exch.sort_values('date', inplace=True)
df_exch_quarterly = pd.merge_asof(quarterly_dates, df_exch, on='date', direction='backward')

# --- Interest Rates (Monthly -> Quarterly) ---
df_interest.sort_values('date', inplace=True)
df_interest_filtered = df_interest[df_interest['date'].dt.month.isin([1, 4, 7, 10])].copy()
df_interest_quarterly = pd.merge_asof(quarterly_dates, df_interest_filtered, on='date', direction='backward')

# --- Tax Rates (Annual -> Quarterly) ---
df_tax.sort_values('date', inplace=True)
df_tax.set_index('date', inplace=True)
df_tax_quarterly = df_tax.reindex(quarterly_dates['date'], method='ffill').reset_index()
df_tax_quarterly.rename(columns={'index': 'date'}, inplace=True)

# --- NEW: Inflation Rates (Monthly -> Quarterly) ---
df_inflation.sort_values('date', inplace=True)
df_inflation_filtered = df_inflation[df_inflation['date'].dt.month.isin([1, 4, 7, 10])].copy()
df_inflation_quarterly = pd.merge_asof(quarterly_dates, df_inflation_filtered, on='date', direction='backward')

# --- NEW: Unemployment Rates (Monthly -> Quarterly) ---
df_unemployment.sort_values('date', inplace=True)
df_unemployment_filtered = df_unemployment[df_unemployment['date'].dt.month.isin([1, 4, 7, 10])].copy()
df_unemployment_quarterly = pd.merge_asof(quarterly_dates, df_unemployment_filtered, on='date', direction='backward')

# --- NEW: National Budget (Quarterly) ---
# Assuming National Budget is already quarterly, but needs alignment
df_budget.sort_values('date', inplace=True)
df_budget_quarterly = pd.merge_asof(quarterly_dates, df_budget, on='date', direction='backward')

# --- NEW: Public Debt (Quarterly) ---
# Assuming Public Debt is already quarterly, but needs alignment
df_debt.sort_values('date', inplace=True)
df_debt_quarterly = pd.merge_asof(quarterly_dates, df_debt, on='date', direction='backward')


print("Inflation columns:", df_inflation_quarterly.columns.tolist())
print("Unemployment columns:", df_unemployment_quarterly.columns.tolist())
print("Budget columns:", df_budget_quarterly.columns.tolist())
print("Debt columns:", df_debt_quarterly.columns.tolist())
# -----------------------
# 5. Merge All Datasets on Quarterly Dates
# -----------------------

# Set the index as date for easier merging
df_gdp.set_index('date', inplace=True)
df_gov.set_index('date', inplace=True)
df_exch_quarterly.set_index('date', inplace=True)
df_interest_quarterly.set_index('date', inplace=True)
df_tax_quarterly.set_index('date', inplace=True)
df_inflation_quarterly.set_index('date', inplace=True)
df_unemployment_quarterly.set_index('date', inplace=True)
df_budget_quarterly.set_index('date', inplace=True)
df_debt_quarterly.set_index('date', inplace=True)

# Start with GDP and merge the others one by one.
var_data = df_gdp[['GDP']].copy()
var_data = var_data.merge(df_gov[['Government_Spendings']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_exch_quarterly[['Exchange_Rates']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_interest_quarterly[['Interest_Rates']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_tax_quarterly[['Tax_Rates']], left_index=True, right_index=True, how='outer')

# Merge new variables
var_data = var_data.merge(df_inflation_quarterly[['Inflation_Rates']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_unemployment_quarterly[['Unemployment_Rates']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_budget_quarterly[['National_Budget']], left_index=True, right_index=True, how='outer')
var_data = var_data.merge(df_debt_quarterly[['Public_Debt']], left_index=True, right_index=True, how='outer')

# Reset index to have 'date' as a column and sort by date.
var_data = var_data.reset_index().sort_values('date')

# -----------------------
# 6. Inspect and Save the Merged Data
# -----------------------
print(var_data.head(10))
var_data.to_csv('data/expanded_var_data.csv', index=False)

# Optional: Create visualization of all economic indicators
plt.figure(figsize=(15, 10))

# Create subplots for different variables
plt.subplot(3, 3, 1)
plt.plot(var_data['date'], var_data['GDP'], 'b-')
plt.title('GDP')
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(var_data['date'], var_data['Government_Spendings'], 'g-')
plt.title('Government Spending')
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(var_data['date'], var_data['Exchange_Rates'], 'r-')
plt.title('Exchange Rates')
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(var_data['date'], var_data['Interest_Rates'], 'c-')
plt.title('Interest Rates')
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(var_data['date'], var_data['Inflation_Rates'], 'm-')
plt.title('Inflation Rates')
plt.grid(True)

plt.subplot(3, 3, 6)
plt.plot(var_data['date'], var_data['Unemployment_Rates'], 'y-')
plt.title('Unemployment Rates')
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(var_data['date'], var_data['National_Budget'], 'k-')
plt.title('National Budget')
plt.grid(True)

plt.subplot(3, 3, 8)
plt.plot(var_data['date'], var_data['Public_Debt'], 'b--')
plt.title('Public Debt')
plt.grid(True)

plt.tight_layout()
plt.savefig('data/economic_indicators.png')
plt.close()