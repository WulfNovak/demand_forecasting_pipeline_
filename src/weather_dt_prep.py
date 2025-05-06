'''
Prepare weather data for feature engineering.
'''

import pandas as pd
import pickle
import numpy as np

def country_level_variables(df, load_only=False):
    '''
    Description: Takes dataframe of variables and seperates them in groups 
    according to the first two letters of the variable.

    Inputs: Energy Dataframe
        load_only: To ONLY include the load_actual_entsoe_transparency variable 
        for a country instead of all variables. 
    Output: Dictionary of countries and their associated variables
    '''
    # Get all unique two-letter prefixes
    countries = set(col[:2] for col in df.columns if col not in ['utc_timestamp', 'cet_cest_timestamp'])
    
    # Dictionary to store results
    group_dfs = {}
    
    if load_only:
        for country in countries:
            # Load actual variable
            load_only_var = country + '_load_actual_entsoe_transparency'

            # Variables to subset to
            all_cols = ['utc_timestamp', load_only_var]
            
            # Create dataframe for country and assign to dictionary
            group_df = df[all_cols].copy() 
            group_dfs[country] = pd.DataFrame(group_df)
            
            # Create global variable for each group
            globals()[country] = group_df

    else:
        # Create dataframe for each prefix
        for country in countries:
            # Get columns that start with the prefix
            country_cols = [col for col in df.columns if col.startswith(country)]
            
            # Add time variables
            time_col = ['utc_timestamp']
            all_cols = time_col + country_cols
            
            # Create dataframe for country and assign to dictionary
            group_df = df[all_cols].copy() 
            group_dfs[country] = pd.DataFrame(group_df)
            
            # Create global variable for each group
            globals()[country] = group_df
    
    return group_dfs


weather_dt = pd.read_csv('./data/preprocessed/weather_data.csv')

all_country_wthr = country_level_variables(weather_dt)

# subset to countries with 15 minute interval data
country_wthr = {k: all_country_wthr[k] for k in ['LU', 'NL', 'BE', 'HU', 'DE', 'AT']}

# Select variables and pivot by country
pivot_country_weather = pd.DataFrame({})

for abbreviation, data in country_wthr.items():
    # Country level variables
    temp = abbreviation + '_temperature'
    radi_direct = abbreviation + '_radiation_direct_horizontal'
    radi_diffuse = abbreviation + '_radiation_diffuse_horizontal'

    # New dataframe with timestamp, country, and load_actual
    data = (data.assign(utc_timestamp = pd.to_datetime(data.utc_timestamp),
                        country = abbreviation)
                .rename(columns = {temp: 'temp',
                                   radi_direct: 'radiation_direct_horizontal',
                                   radi_diffuse: 'radiation_diffuse_horizontal'})
                [['utc_timestamp', 'country', 'temp', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal']]) 
    # Concat
    pivot_country_weather = pd.concat([pivot_country_weather, data])

# Aggregate for daily stats
pivot_country_weather = pivot_country_weather.reset_index(drop=True)
pivot_country_weather['day'] = pivot_country_weather['utc_timestamp'].dt.date
day_mean_temp = (pivot_country_weather.groupby('day', as_index=False)
                                      .agg(mean_temp = ('temp', 'mean'),
                                           max_temp = ('temp', 'max'),
                                           min_temp = ('temp', 'min')))

# Create hdd and cdd variables
pivot_country_weather_2 = (pivot_country_weather.merge(day_mean_temp, how='left', on='day')
                                                .assign(hdd = lambda x: np.where(15.5 - x.temp > 0, 1, 0),
                                                        cdd = lambda x: np.where(x.temp - 15.5 > 0, 1, 0)))

# Save prepared weather data
wthr_file_path_1 = './data/pivot_country_weather.pickle'
with open(wthr_file_path_1, 'wb') as file:
    pickle.dump(pivot_country_weather_2, file, protocol=pickle.HIGHEST_PROTOCOL)