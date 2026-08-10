'''
"Top Secret" - (not really) data preparation, to be read into jupyter notebook
'''
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
def load_train_test(df, drop_index=True, y='load_actual'):
    ''' 
    Input: Dataframe from features prep
    Output: Data prepared for hyperparameter search
    '''
    X = (df.drop(columns=[y, 'country', 'day']) 
           .dropna()
           .reset_index()
           )
    if drop_index:
        X = X.drop(columns='utc_timestamp')
     
    y = (df.reset_index(drop=True)[[y]])
    
    return X, y

import pandas as pd
import numpy as np

load_wthr_up['day_or_night'] = np.where((load_wthr_up['hour'] > 6) & (load_wthr_up['hour'] < 20), 1, 0)
load_wthr_up['diff_actual'] = load_wthr_up['load_actual'].diff()
# def add_agg_vars(data, variables, aggregators)

# Daily stats
day_agg = (load_wthr_up.groupby(['year', 'day'], as_index=False)
            .agg(
                d_mean_load_actual = ('load_actual', 'mean'),
                d_std_load_actual = ('load_actual', 'std'),
                d_mean_diff_actual = ('diff_actual', 'mean'),
                d_std_diff_actual = ('diff_actual', 'std'),
                d_mean_temperature = ('temperature', 'mean'),
                d_std_temperature = ('temperature', 'std')
            ))

# Weekly stats
week_agg = (load_wthr_up.groupby(['year', 'week_of_year'], as_index=False)
            .agg(
                w_mean_load_actual = ('load_actual', 'mean'),
                w_std_load_actual = ('load_actual', 'std'),
                w_mean_diff_actual = ('diff_actual', 'mean'),
                w_std_diff_actual = ('diff_actual', 'std'),
                w_mean_temperature = ('temperature', 'mean'),
                w_std_temperature = ('temperature', 'std')
            ))

# Monthly stats
month_agg = (load_wthr_up.groupby(['year', 'month'], as_index=False)
            .agg(
                m_mean_load_actual = ('load_actual', 'mean'),
                m_std_load_actual = ('load_actual', 'std'),
                m_mean_diff_actual = ('diff_actual', 'mean'),
                m_std_diff_actual = ('diff_actual', 'std'),
                m_mean_temperature = ('temperature', 'mean'),
                m_std_temperature = ('temperature', 'std')
            ))

# Combine into load_wthr_up df
load_wthr_up = (load_wthr_up.reset_index()
                  .merge(day_agg, how='left', on=['year', 'day'])
                  .merge(week_agg, how='left', on=['year', 'week_of_year'])
                  .merge(month_agg, how='left', on=['year', 'month'])
                  .set_index('utc_timestamp'))

# 15min interval lags
horizon = 96 * 7 # 1 week horizon
n = 5
int_15min = [*np.arange(horizon, horizon+n), # 15min 
             *np.arange(horizon+96, horizon+96+n), # day
             *np.arange(horizon+96*7, horizon+96*7+n), # week
             *np.arange(horizon+96*365, horizon+96*365+n),] # year 
vars = ['load_actual', 'diff_actual', 'temperature']
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
vars = ['d_mean_load_actual', 'd_std_load_actual', 
        'd_mean_diff_actual', 'd_std_diff_actual', 
        'd_mean_temperature', 'd_std_temperature']
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
vars = ['w_mean_load_actual', 'w_std_load_actual',
        'w_mean_diff_actual', 'w_std_diff_actual',
        'w_mean_temperature', 'w_std_temperature']
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
vars = ['m_mean_load_actual', 'm_std_load_actual', 
        'm_mean_diff_actual', 'm_std_diff_actual',
        'm_mean_temperature', 'm_std_temperature']
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
        'd_mean_diff_actual', 'd_std_diff_actual',
        'd_mean_temperature', 'd_std_temperature', 
        'w_mean_load_actual', 'w_std_load_actual',
        'w_mean_diff_actual', 'w_std_diff_actual',
        'w_mean_temperature', 'w_std_temperature',
        'm_mean_load_actual', 'm_std_load_actual',
        'm_mean_diff_actual', 'm_std_diff_actual',
        'm_mean_temperature', 'm_std_temperature'] 
                  )

import lightgbm

Xu_train, Xu_test = trainval_test_split(Xu)
X_train, y_train = load_train_test(Xu_train, y='load_actual')
X_train = X_train.drop(columns='diff_actual')
# X_train, y_train = load_train_test(Xu_train, y='diff_actual')

# Train model
early_stopping = lightgbm.early_stopping(5)
model = lightgbm.LGBMRegressor(
                random_state=21, 
                n_jobs=-1,
                callbacks=[early_stopping],
                verbosity=-1) # Simple model for feature selection
model.fit(X_train, y_train)