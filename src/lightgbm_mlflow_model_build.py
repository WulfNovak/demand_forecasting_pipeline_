'''
LightGBM Model optimization pipeline. 
'''

import subprocess
import mlflow
import pickle
import lightgbm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit 
import time
import matplotlib.pyplot as plt
import seaborn as sbn
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from hidden.lightgbm_forecasting_pipeline import *

RANDOM_STATE = 221

def automated_forecast(
          datasets, 
          dataset_names, 
          iter_per_model=25, 
          nested_windows=10, 
          experiment_name='No experiment name given'):

    if __name__ == "__main__":
        # Initialize server
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)

        for name, data in zip(dataset_names, datasets):
            mlflow.start_run(run_name=f'Automated Load Forecasting Pipeline')

            # Group data, begin process for each group in data
            country_data = data.groupby('country')

            for country, data in country_data: # --- This portion can be done in parallel
                #with mlflow.start_run(nested=True, run_name=f"Country: {country}"):
                hyperparam_opt(data, country, iterations=iter_per_model, nested_windows=nested_windows)  
    
    # end mlflow run (precautionary)
    mlflow.end_run()

# Load model data
model_dt_path = './data/load_wthr_model_ready.pickle'

with open(model_dt_path, 'rb') as f:
    model_dt = pickle.load(f)

if len(model_dt) > 1:
    print(f"Model Data successfully loaded with shape: {model_dt.shape}")

### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
#          Decide on what outputs to restrict from the timeseries forecast        #
### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###

# Begin mlflow 
# run_mlflow = subprocess.run("mlflow ui", shell=True)
# time.sleep(15)

auto_run = model_dt.loc[model_dt['country'] == 'DE']

automated_forecast(datasets=[auto_run], # Xu
                   dataset_names='15min Intervals',
                   iter_per_model=2, # 20
                   experiment_name=f'Automated pipeline tests',
                   nested_windows=2) # 10

#run_mlflow.terminate()
