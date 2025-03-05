# Demand Forecasting MLOps

### Description
With energy and weather data across 6 european countries (Austria, Netherlands, Luxembourg, Hungary, Belgium, Germany), the following predicts energy usage 7 days into the future. -- conducts some EDA, engineers features, unites the data, then performs automated model optimization, logging training and result metrics per country with MLflow.

### Script order
1. energy_eda.ipynb
2. weather_prep.ipynb
3. feature_prep.ipynb
4. lgbm_forecast.ipynb

Lightgbm was selected as the first model due to there being variable importance plots, assisting variable selection. Looking forward to integrating other model types once a set of predictive features has been refined.  

## MLflow
The result of the above scripts are stored on an mlflow server. This server containers the parameters of the best model per country during training. Currently, results metrics, variable importance plots, and forecasting visualizations are logged during the automated training and testing.

Here are some example screenshots of this ongoing experiment:
![example_param_chart](https://github.com/user-attachments/assets/a97fbac2-62d6-4699-a198-53df8727a484)
![example_result_metrics](https://github.com/user-attachments/assets/ea0059ec-54f4-49a0-9960-d5eed933f14d)
![image](https://github.com/user-attachments/assets/13a39f8d-7093-4fa3-bba3-ae6630d811d7)

## Data Sources
### Energy Data 
Origin: https://data.open-power-system-data.org/time_series/2019-06-05      

Details on how the Data was processed:
https://github.com/Open-Power-System-Data/time_series/blob/885c0946fe57d1a2f44f7bc57306e87811e4e2e8//processing.ipynb   

 Citation(s): 
- Baur, L.; Chandramouli, V.; Sauer, A.: Publicly Available Datasets For Electric Load Forecasting – An Overview. In: Herberger, D.; Hübner, M. (Eds.): Proceedings of the CPSL 2024. Hannover : publish-Ing., 2024, S. 1-12. DOI: https://doi.org/10.15488/17659

- Open Power System Data. 2020. Data Package Time series. Version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06. (Primary data from various sources, for a complete list see URL).

Details on how the Data was processed:
https://github.com/Open-Power-System-Data/time_series/blob/885c0946fe57d1a2f44f7bc57306e87811e4e2e8//processing.ipynb   

___

### Weather Data 
Origin: https://data.open-power-system-data.org/weather_data/    

Citation:
- Open Power System Data. 2020. Data Package Weather Data. Version 2020-09-16. https://doi.org/10.25832/weather_data/2020-09-16. (Primary data from various sources, for a complete list see URL).
___

## Closing Thoughts
There many opportunities to further generalize and automate this demand forecasting setup. Additionally, other model types and optimizations can easily be integrated, then compared with the existing lightgbm forecasting models through MLflow. There is great potential for further productionalizing ML models. 
