import pandas as pd
import os
import pickle
import numpy as np

print("Running energy_dt_prep.py")

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
    
    return group_dfs #pd.DataFrame(group_dfs)

def create_hour_minute(df):
    '''
    This function is supplemental to the following 'add_time_features' function,
    creating an hour_minute variable
    '''
    # Initialize empty variable
    df['hour_minute'] = None

    # Conditions
    conditions = [df['utc_timestamp'].dt.minute == 15,
                  df['utc_timestamp'].dt.minute == 30,
                  df['utc_timestamp'].dt.minute == 45]
    
    transform = [df['utc_timestamp'].dt.hour + .25,
                 df['utc_timestamp'].dt.hour + .5,
                 df['utc_timestamp'].dt.hour + .75]
    
    # given conditions, transform 
    converted_times = np.select(conditions, transform, df['utc_timestamp'].dt.hour)

    return converted_times

def country_holiday_indicator(df):
    '''
    This function is supplemental to the following 'add_time_features' function,
    creating an indicator for each holiday in a given country.
    '''
    import holidays 

    NL_holidays = holidays.NL()
    LU_holidays = holidays.LU()
    AT_holidays = holidays.AT()
    HU_holidays = holidays.HU()
    BE_holidays = holidays.BE()
    DE_holidays = holidays.DE()

    # Select countries
    conditions = [
        df['country'] == 'NL',
        df['country'] == 'LU',
        df['country'] == 'AT',
        df['country'] == 'HU',
        df['country'] == 'BE',
        df['country'] == 'DE',
    ]
    # Holidays for that country
    holidays = [
        df['utc_timestamp'].dt.date.isin(NL_holidays),
        df['utc_timestamp'].dt.date.isin(LU_holidays),
        df['utc_timestamp'].dt.date.isin(AT_holidays),
        df['utc_timestamp'].dt.date.isin(HU_holidays),
        df['utc_timestamp'].dt.date.isin(BE_holidays),
        df['utc_timestamp'].dt.date.isin(DE_holidays)
    ]

    # Create indicator 
    indicator = np.select(conditions, holidays, None).astype(int)
    return indicator

def add_time_features(df): # requires country_actuals dataframe
    '''
    Description: Given country_actuals dataset derived from 
    country_level_variables method + .reset_index(), adds
    several simple time features ('day', 'hour', 'month'
    'is_weekend' indicator, and 'is_holiday' indicator)
    '''
    # day
    df['day'] = df['utc_timestamp'].dt.date
    # day ordinal
    df['day_ordinal'] = pd.to_datetime(df['utc_timestamp']).apply(lambda x: x.toordinal()) # toordinal
    # hour
    df['hour'] = df['utc_timestamp'].dt.hour
    # hour_minute? # consider adding hour_minute here for Xu dataset
    df['hour_minute'] = create_hour_minute(df)
    # week of year
    df['week_of_year'] = df['utc_timestamp'].dt.isocalendar().week
    # month
    df['month'] = df['utc_timestamp'].dt.month
    # year
    df['year'] = df['utc_timestamp'].dt.year
    # weekend indicator variable
    df['is_weekend'] = df['utc_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    # holiday indicator variable
    df['is_holiday'] = country_holiday_indicator(df)

    return df

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read in and transform data --- Ideally functionalized in the future

# Read in Data

print("Reading in raw country level electricy data")
energy_15_raw = pd.read_csv(str(os.getcwd()) + "/data/preprocessed/time_series_15min_singleindex.csv")

# Country level variables
print("Getting country level variables")
country_dfs = country_level_variables(energy_15_raw)

# Select starting point of series based on load_actuals
print("Selecting starting point of series based on load actuals")
lower_bounds = []
for abbreviation, data in country_dfs.items():
    var = abbreviation + '_load_actual_entsoe_transparency'
    # "First Data Available (fdw)" data
    fdw_data = country_dfs[abbreviation][[var, 'utc_timestamp']].dropna()
    data_start = min(fdw_data.utc_timestamp)
    # Eariest datetime where data is available
    lower_bound_row = fdw_data[fdw_data['utc_timestamp']==data_start].index.values.astype(int)[0]
    lower_bounds.append(lower_bound_row)

print(f"First row of available data: {lower_bounds}")

# Max row number is the minimum utc_timestamp with data available
print("Creating bounded dataset")
energy_15_bounded = energy_15_raw.iloc[max(lower_bounds):]

dfs = country_level_variables(energy_15_bounded, load_only=True)

country_actuals = pd.DataFrame({})
for abbreviation, data in dfs.items():
    # Consolidate
    name = abbreviation + '_load_actual_entsoe_transparency'
    # New dataframe with timestamp, country, and load_actual
    data = (data.assign(utc_timestamp = pd.to_datetime(data.utc_timestamp),
                        country = abbreviation)
                .rename(columns = {name: 'load_actual_entsoe_transparency'})
                [['utc_timestamp', 'country', 'load_actual_entsoe_transparency']]) 
    country_actuals = pd.concat([country_actuals, data])

country_actuals = country_actuals.drop_duplicates().reset_index(drop=True)

# Add time features (in place)
add_time_features(country_actuals)


with open('./data/country_actuals.pickle', 'wb') as f:
    pickle.dump(country_actuals, f)

# Delete unecessary dataframes
del energy_15_raw
del country_dfs
del energy_15_bounded
del dfs
del country_actuals