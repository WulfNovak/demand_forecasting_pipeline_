'''
Combine weather_dt and energy_dt, then create features for model pipline.
'''

import pandas as pd
import pickle
import numpy as np
from hidden.TimeSeriesImpute import * 
import time
import lightgbm
import warnings

def country_na_counts(data, variable):
    '''
    Get NA counts by country for a given variable.
    '''
    for country in set(data.country):
        print(f"{country} NA count: {sum(data.loc[lambda x: x.country == country, variable].isna())}")

def dd_hourly_pivot(df): # requires country actuals
    '''
    Description: pivots table long-ways, with country and aggregated hourly 
    actuals load as columns
    '''
    return (df.pivot_table(index=['day', 'country', 'hdd', 'cdd'], 
                           values='load_actual', 
                           columns = 'hour', 
                           aggfunc='sum')
                            # remove first day with NaNs
              .dropna()
              .reset_index(level=1))

def lags_and_rolling_means(df, 
                           list_vars=None, 
                           group='country', 
                           lag_num=1,
                           drop_na=True
                           ):
    '''
    Description: With data frame, list of variables, and hours to lag/roll, 
    adds lags and rolling mean to a copy of the dataframe. 
    *Requires a group argument
    '''

    df_copy = df.copy(deep=True)

    # iterate over variables
    for var in list_vars:
        # iterate over hours
        for lag in lag_num:
            # lag columns
            lag_col = f"{var}_lag_{int(lag)}" 
            df_copy[lag_col] = df_copy.groupby(group, as_index=False)[var].shift(lag)
            # rolling columns
            roll_col = f"{var}_{lag}_rollmean_{3}" 
            df_copy[roll_col] = (df_copy.groupby(group, as_index=False)[var]
                                 .shift(lag+1) # prevent data leakage
                                 .rolling(3) # rolling 3, shift is then the lag
                                 .mean()
                                 )
    if drop_na: # remove NAs from created lagged variables
        return df_copy.dropna()
    else:
        return df_copy    
    
def trainval_test_split(df):
    '''
    Input: Dataframe from features prep
    Output: Train and final test sets with a 7 day window
    '''
    train_validate = df.loc[df['day'] < (df['day'].max() - pd.to_timedelta('7day'))]
    final_test = df.loc[df['day'] > (df['day'].max() - pd.to_timedelta('8day'))]

    return train_validate, final_test

# train validate
def load_train_test(df, drop_index=True):
    ''' 
    Input: Dataframe from features prep
    Output: Data prepared for hyperparameter search
    '''
    X = (df.drop(columns=['load_actual', 'country', 'day']) 
           .dropna()
           .reset_index()
           )
    if drop_index:
        X = X.drop(columns='utc_timestamp')
     
    y = (df.reset_index(drop=True)[['load_actual']])
    
    return X, y

# Read in, then bound weather and load data
load_wthr_dt = "./data/pivot_country_weather.pickle"
load_energy_dt = "./data/country_actuals.pickle"

print("Reading weather and electricity load data")
with open(load_wthr_dt, 'rb') as f:
    wthr_dt = (pickle.load(f).loc[lambda x: x.utc_timestamp.between('2015-01-01', '2019-04-30')]
                             # Note: setting utc_timestamp as index causes drop_duplicates to remove needed observations
                             .set_index(['utc_timestamp']) 
                             # simplify var names
                             .rename(columns={'radiation_direct_horizontal': 'radi_direct',
                                              'radiation_diffuse_horizontal': 'radi_diffuse',
                                              'temp': 'temperature'}))
with open(load_energy_dt, 'rb') as f:
    load_dt = (pickle.load(f).loc[lambda x: x.utc_timestamp.between('2015-01-01', '2019-04-30')]
                             .set_index(['utc_timestamp']) 
                             # simplify var names
                             .rename(columns={'load_actual_entsoe_transparency': 'load_actual'}))
print(f"\nData Loaded Successfully")

# Upsample weather data to match load data frequency
ffill_vars = ['country', 'day', 'mean_temp', 'max_temp', 'min_temp', 'hdd', 'cdd']
interp_vars = ['temperature', 'radi_direct', 'radi_diffuse']
wthr_upsample = pd.DataFrame({})

for country in wthr_dt['country'].unique():
    country_upsample = wthr_dt.loc[lambda x: x.country == country].resample('15min').asfreq()
    # interpolate ts variables, and ffill categorical variables
    interped = country_upsample[interp_vars].interpolate()
    ffilled = country_upsample[ffill_vars].ffill()

    # merge
    upsampled = interped.merge(ffilled, how='left', on='utc_timestamp')

    wthr_upsample = pd.concat([wthr_upsample, upsampled])

load_dt = load_dt[['day', 'day_ordinal', 'year',  'week_of_year', 'month', 'hour', # arrange variables
                   'hour_minute', 'country', 'load_actual', 'is_weekend', 'is_holiday']]
load_wthr_up = load_dt.merge(wthr_upsample, how='left', on=['utc_timestamp', 'country', 'day'])

print('NAs by Country prior to imputation:')
country_na_counts(load_wthr_up, 'load_actual')

print(f"\nBeginning Imputation")

print("\nNote: Warnings are being ignored.\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    start = time.time()
    ts_init = TimeSeriesImpute()

    load_wthr_up_2 = pd.DataFrame({})
    for country in set(load_wthr_up.country):
        input_df = ts_init.create_impute_data(load_wthr_up.reset_index(), country=country)
        imputed_df = ts_init.impute_timeseries_gaps(input_df, impute_var='load_actual') # --- function should be updated to impute in place

        load_wthr_up_2 = pd.concat([load_wthr_up_2, imputed_df])

    # Combine
    load_wthr_up = (load_wthr_up.reset_index()
                                .merge(load_wthr_up_2[['utc_timestamp', 'country', 'load_actual']], 
                                    how='left', 
                                    on=['utc_timestamp', 'country']
                                    #left_index=True, right_index=True
                                    )
                                .drop(columns='load_actual_x')
                                .rename(columns={'load_actual_y': 'load_actual'})
                                .set_index('utc_timestamp')
    )
if len(load_wthr_up_2) == len(load_wthr_up):
    print(f"Imputation complete in {(time.time() - start):2f} seconds.")

print(f'\nNAs remaining after imputation:')
country_na_counts(load_wthr_up, 'load_actual')

### ------------------------------------------------------------------------------------------- ###
#                                     Feature engineering                                         #
### ------------------------------------------------------------------------------------------- ###

# Add day_or_night indicator
load_wthr_up['day_or_night'] = np.where((load_wthr_up['hour'] > 6) & (load_wthr_up['hour'] < 20), 1, 0)


# def add_agg_vars(data, variables, aggregators) --- Consider functionalizing the following
# Daily stats
day_agg = (load_wthr_up.groupby(['year', 'day'], as_index=False)
            .agg(
                d_mean_load_actual = ('load_actual', 'mean'),
                d_std_load_actual = ('load_actual', 'std'),
                d_mean_temperature = ('temperature', 'mean'),
                d_std_temperature = ('temperature', 'std')
            ))

# Weekly stats
week_agg = (load_wthr_up.groupby(['year', 'week_of_year'], as_index=False)
            .agg(
                w_mean_load_actual = ('load_actual', 'mean'),
                w_std_load_actual = ('load_actual', 'std'),
                w_mean_temperature = ('temperature', 'mean'),
                w_std_temperature = ('temperature', 'std')
            ))

# Monthly stats
month_agg = (load_wthr_up.groupby(['year', 'month'], as_index=False)
            .agg(
                m_mean_load_actual = ('load_actual', 'mean'),
                m_std_load_actual = ('load_actual', 'std'),
                m_mean_temperature = ('temperature', 'mean'),
                m_std_temperature = ('temperature', 'std')
            ))

# Combine into load_wthr_up df
load_wthr_up = (load_wthr_up.reset_index()
                  .merge(day_agg, how='left', on=['year', 'day'])
                  .merge(week_agg, how='left', on=['year', 'week_of_year'])
                  .merge(month_agg, how='left', on=['year', 'month'])
                  .set_index('utc_timestamp'))

print("\nNote: Warnings are being ignored.\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # 15min interval lags
    horizon = 96 * 7 # 1 week horizon
    n = 5
    int_15min = [*np.arange(horizon, horizon+n), # 15min 
                *np.arange(horizon+96, horizon+96+n), # day
                *np.arange(horizon+96*7, horizon+96*7+n), # week
                *np.arange(horizon+96*365, horizon+96*365+n),] # year 
    vars = ['load_actual', 'temperature']
    Xu = (lags_and_rolling_means(load_wthr_up, 
                                list_vars=vars, 
                                group='country',
                                lag_num=int_15min,
                                drop_na=False)
                                .drop(columns=['hdd', 'cdd']))

    # day interval vars
    horizon = 7
    n = 3
    int_day = [*np.arange(horizon, horizon+n), # days
            *np.arange(horizon+7, horizon+7+n), # week
            *np.arange(horizon+365, horizon+365+n),] # year
    vars = ['d_mean_load_actual', 'd_std_load_actual', 'd_mean_temperature', 'd_std_temperature']
    Xu = lags_and_rolling_means(Xu, 
                                list_vars=vars, 
                                group='country',
                                lag_num=int_day,
                                drop_na=False)
    # week interval vars
    horizon = 1
    n = 3
    int_week = [*np.arange(horizon, horizon+n), # weeks
                *np.arange(horizon+52, horizon+52+n),] # year
    vars = ['w_mean_load_actual', 'w_std_load_actual', 'w_mean_temperature', 'w_std_temperature']
    Xu = lags_and_rolling_means(Xu, 
                                list_vars=vars, 
                                group='country',
                                lag_num=int_week,
                                drop_na=False)
    # month interval vars
    horizon = 1
    n = 3
    int_month = [*np.arange(horizon, horizon+n), # months
                *np.arange(horizon+12)] # year
    vars = ['m_mean_load_actual', 'm_std_load_actual', 'm_mean_temperature', 'm_std_temperature']
    Xu = lags_and_rolling_means(Xu, 
                                list_vars=vars, 
                                group='country',
                                lag_num=int_month,
                                drop_na=True)

correct_dtypes = {'day': 'object',
                  'day_ordinal': 'int32',
                  'year': 'int32',
                  'week_of_year': 'int32',
                  'month': 'int32',
                  'hour': 'int32',
                  'hour_minute': 'float64',
                  'country': 'object',
                  'load_actual': 'float64',
                  'is_weekend': 'int8',
                  'is_holiday': 'int8',
                  'temperature': 'float64',
                  'radi_direct': 'float64',
                  'radi_diffuse': 'float64',
                  'mean_temp': 'float64',
                  'max_temp': 'float64',
                  'min_temp': 'float64'}

def change_dtypes(data, dtype_dict):
    for col, dtype in dtype_dict.items():
        data[col] = data[col].astype(dtype)

#change_dtypes(Xd, correct_dtypes)
change_dtypes(Xu, correct_dtypes)

# Drop variables to prevent data leakage
Xu = Xu.drop(
    columns = [
        'temperature', 'mean_temp', 'max_temp', 'min_temp',
        'radi_direct', 'radi_diffuse',
        'd_mean_load_actual', 'd_std_load_actual',
        'd_mean_temperature', 'd_std_temperature', 
        'w_mean_load_actual', 'w_std_load_actual',
        'w_mean_temperature', 'w_std_temperature',
        'm_mean_load_actual', 'm_std_load_actual',
        'm_mean_temperature', 'm_std_temperature'] 
                  )

Xu_train, Xu_test = trainval_test_split(Xu)
X_train, y_train = load_train_test(Xu_train)

# Train model
early_stopping = lightgbm.early_stopping(5)
model = lightgbm.LGBMRegressor(
                random_state=21, 
                n_jobs=-1,
                callbacks=[early_stopping],
                verbosity=-1) # Simple model for feature selection
model.fit(X_train, y_train)

importances = model.feature_importances_
imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)
imp_df['perc'] = imp_df['Importance'] / sum(imp_df['Importance'])

### --- Threshold selected based on importance plot --- ###
threshold = 0.2 

selected_vars = list(imp_df.loc[imp_df['perc'] >= threshold]['Feature'])
selected_vars.append('load_actual') # y
# selected_vars.append('diff_actual') 
selected_vars.append('country') # used within optimization function
selected_vars.append('day') # used to create 7 day horizon

model_dt = Xu[selected_vars].round(8)

# Save data
load_wthr = './data/load_wthr_model_ready.pickle'
with open(load_wthr, 'wb') as f:
    pickle.dump(model_dt, f)

