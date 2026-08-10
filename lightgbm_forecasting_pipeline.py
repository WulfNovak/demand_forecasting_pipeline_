'''
Author: Wulf Novak
Last edit: 4/25/2025

This forecast pipeline pairs with MLflow and is somewhat bespoke to the 
load_forecasting_analysis project. 

MLflow records: 
    - optimal model
    - variable importance plot
    - plot of predicted vs actuals on the test dataset
    - metrics including: training time, training mape, test mape and test mae

The optimized forecasting algorithm utilizes: 
    - lightGBM
    - bayesian hyperparameter optimization
    - nested windows for cross validation
    - code to generate the outputs given to MLflow
'''

import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgbm
import time
import matplotlib.pyplot as plt
import seaborn as sbn
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit 

RANDOM_STATE = 221

def num_rows_horizon(df, timeframe='7day'):
    ''' 
    Input: Dataframe, Timeframe (in to_timedelta format)
    Output: Outputs number of rows in dataframe to reach timeframe
    '''
    return len(df.loc[df['day'] > (df['day'].max() - pd.to_timedelta(timeframe))])

# validate test
def trainval_test_split(df):
    '''
    Input: Dataframe from features prep
    Output: Train and final test sets with a 7 day window
    '''
    train_validate = df.loc[df['day'] < (df['day'].max() - pd.to_timedelta('7day'))]
    final_test = df.loc[df['day'] > (df['day'].max() - pd.to_timedelta('8day'))]
    # final_test = df.copy(deep=True)

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

# Function for logging parameters from Hyperparameter Optimization function
def optimization_logs(opt_model, 
                      importance_plot, 
                      test_plot, 
                      bayes_opt, 
                      final_mape, 
                      final_mae, 
                      end, 
                      country, 
                      imp_df):

    # Log params, metrics, model, and plots
    mlflow.log_params(opt_model.best_params_)
    mlflow.log_table(opt_model.best_params_, f'{country}_best_params.json')
    mlflow.log_table(imp_df, f'{country}_importance_vars.json')
    mlflow.log_figure(importance_plot.figure, f'{country}_variable_importance.png')
    mlflow.log_figure(test_plot, f'{country}_predicted_vs_actuals.png') 
    mlflow.log_metrics({
                    'optimization_training_time': end,
                    'best_training_mape': abs(bayes_opt.best_score_) * 100,
                    'final_test_mape': final_mape * 100,
                    'final_test_mae': final_mae,
                })
    mlflow.sklearn.log_model(sk_model=opt_model.best_estimator_, 
                              artifact_path=f"{country}_best_model"
                              )
    
def hyperparam_opt(dataset, country, iterations=25, nested_windows=10):
    with mlflow.start_run(run_name=f'{country}_model_optimization', nested=True): 
        # Split into training/validation/test
        trainval, test = trainval_test_split(dataset) 
    #---# further split training test into training and validation with load_train_test #---#
        nrow = num_rows_horizon(dataset, timeframe='7day')
        X_train, y_train = load_train_test(trainval)

        # Nested timeseries split for cross validation
        ts_cv = TimeSeriesSplit(n_splits=nested_windows, test_size=nrow)

        # Model and callbacks
        early_stopping = lgbm.early_stopping(5)
        model = lgbm.LGBMRegressor(
                        random_state=RANDOM_STATE, 
                        n_jobs=-1,
                        callbacks=[early_stopping],
                        verbosity=-1)
        
        # Parameter set / ranges
        params = {
            'learning_rate': Real(0.001, 1, 'uniform'),
            'reg_alpha': Real(0, 1, 'uniform'),
            'num_leaves': Integer(20, 800, 'uniform'), 
            'max_depth': Integer(5, 12, 'uniform'), 
            'subsample': Real(0.5, 1, 'uniform'), 
            'colsample_bytree': Real(0.7, 1, 'uniform'),
            'min_data_in_leaf': Integer(20, 100, 'uniform'),
        } 

        # BayesSearch of optimal params
        bayes_opt = BayesSearchCV(
            model,
            params,
            cv=ts_cv,
            n_iter=iterations, # 50, 100
            scoring='neg_mean_absolute_percentage_error', # 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'
            n_jobs=-1,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )

        # Run and time bayesian optimization of params
        start = time.time()
        opt_model = bayes_opt.fit(X_train, y_train)
        #opt_model = bayes_opt.best_estimator.fit(X_train, y_train)
        end = round(time.time() - start, 2)

        # Variable Importance Plot
        importance_plot = lgbm.plot_importance(bayes_opt.best_estimator_, 
                                                figsize=(8,5), 
                                                title=f'{country} Variable Importance',
                                                max_num_features=20)
        

        importances = bayes_opt.best_estimator_.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        imp_df['perc'] = imp_df['Importance'] / sum(imp_df['Importance'])
        imp_df['perc'] = (imp_df['perc'] * 100).round(2)
        imp_df['cum_imp'] = imp_df['Importance'].cumsum()
        imp_df['cum_perc'] = ((imp_df['Importance'].cumsum() / sum(imp_df['Importance'])).round(4) * 100).round(2)
        
        # Get predictions of optimal set of params for final test set
        X_train, _ = load_train_test(trainval, drop_index=False)
        X_test, y_test = load_train_test(test, drop_index=False)
        y_pred = opt_model.best_estimator_.predict(X_test.drop(columns='utc_timestamp'),
                                    num_iteration = opt_model.best_estimator_.best_iteration_)
        y_test = np.reshape(y_test, (len(y_test),))
        final_mape = mean_absolute_percentage_error(y_test, y_pred)  
        final_mae = mean_absolute_error(y_test, y_pred)

        # Time Series plot of Predicted Vs Actuals
        plot_dt = pd.DataFrame({'Day': X_test['utc_timestamp'],
                                'actual': np.reshape(y_test, (len(y_test),)), 
                                'predicted': y_pred})

        preds_vs_actual = plt.figure(figsize=(10,5))
        _ = sbn.lineplot(data=plot_dt, 
                        x='Day', 
                        y='actual', 
                        color='red', 
                        label='Actuals')
        _ = sbn.lineplot(data=plot_dt, 
                        x='Day', 
                        y='predicted', 
                        color='grey', 
                        linestyle='--',
                        label='Predicted')
        _ = plt.ylabel('Load (MW)')
        _ = plt.title(f'Predicted Vs. Actual Load for {country}')

        # for listing the importance plot variables

        # # Log params, result metrics, and visualizations
        optimization_logs(opt_model, 
                          importance_plot, 
                          preds_vs_actual, 
                          bayes_opt, 
                          final_mape, 
                          final_mae,
                          end, 
                          country,
                          imp_df)

                          

