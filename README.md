# Demand Forecasting MLOps

### Description
With energy and weather data, predicted energy usage with a 7-day forecast horizon across 6 european countries (Austria, Netherlands, Luxembourg, Hungary, Belgium, Germany). 

### Script order
1. energy_eda.ipynb 
2. weather_prep.ipynb
3. feature_prep.ipynb
4. lgbm_forecast.ipynb

Lightgbm selected as first model due to there being variable importance plots, assisting variable selection. 

## MLflow
Forecasting results are stored on a mlflow server. Currently, results metrics, variable importance plots, and forecasting visualizations are logged during the training and testing.

Example screenshots:
![image](https://github.com/user-attachments/assets/f478347f-6476-4db2-b3b9-8a0b01707f52)
![image](https://github.com/user-attachments/assets/814b338d-54f8-4d27-b455-0ec1c34555d1)
![image](https://github.com/user-attachments/assets/13a39f8d-7093-4fa3-bba3-ae6630d811d7)

## Data Sources
### Energy Data 
Origin: https://data.open-power-system-data.org/time_series/2019-06-05      

Details on how the Data was processed:
https://github.com/Open-Power-System-Data/time_series/blob/885c0946fe57d1a2f44f7bc57306e87811e4e2e8//processing.ipynb   

 Citation(s): 
- Baur, L.; Chandramouli, V.; Sauer, A.: Publicly Available Datasets For Electric Load Forecasting – An Overview. In: Herberger, D.; Hübner, M. (Eds.): Proceedings of the CPSL 2024. Hannover : publish-Ing., 2024, S. 1-12. DOI: https://doi.org/10.15488/17659

- Open Power System Data. 2020. Data Package Time series. Version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06. (Primary data from various sources, for a complete list see URL).

___

### Weather Data 
Origin: https://data.open-power-system-data.org/weather_data/    

Citation:
- Open Power System Data. 2020. Data Package Weather Data. Version 2020-09-16. https://doi.org/10.25832/weather_data/2020-09-16. (Primary data from various sources, for a complete list see URL).
___
