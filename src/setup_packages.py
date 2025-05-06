'''
Setup file with necessary packages and to complete the model pipeline
'''
print("Beginning packages imports ---")

# Data loading
import os
from pathlib import Path 
import pickle
print("Data loading packages successfully imported")

# Data 
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
print("Data manipulation and visualization packages successfully imported")

# Modeling and optimization
print("importing modeling and optimization packages")
from skopt import BayesSearchCV
from skopt.space import Real, Integer
# from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import lightgbm as lgbm
import time

print("--- Package imports complete!")


# This is not the ideal way to annotate a script - seek better ways to do so

