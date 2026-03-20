import requests
import pandas as pd

def fetch_fred_data(series_id, api_key, start_date='1970-01-01', end_date='2020-12-31'):
    """
    Fetches data from FRED for a given series_id between start_date and end_date.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert the list of observations to a DataFrame
    observations = data.get('observations', [])
    df = pd.DataFrame(observations)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

# Replace with your actual FRED API key
api_key = '5bbfc9d2ec68ce738b1a98bff55517a7'
# Example: Fetching US GDP (the FRED series for GDP is 'GDP')
gdp_series_id = 'DEXUSUK'
df_gdp = fetch_fred_data(gdp_series_id, api_key)
print("Sample")
print(df_gdp.head())

# Save the data to a CSV file for later use
df_gdp.to_csv('data/exchange_rates_Fred.csv', index=False)
