import joblib
import numpy as np
import pandas as pd

import rasterio as rio
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
import ale_py


model_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/model/'

rf = joblib.load(model_dir + 'ranFor_model_500m_grid_mm_yr_12_02_2025.joblib')
data_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/data/test_data/'
X_test_data = pd.read_csv(data_dir + 'X_test_data.csv' )
data_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/data/test_data/'
X_test_data = pd.read_csv(data_dir + 'X_test_data.csv' )
# X_test_data_50k = X_test_data.sample(n=50000, random_state=42)
X_test_data_50k = X_test_data.sample(n=1000, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values_2 = explainer.shap_values(X_test_data_50k)



hru_scale = pd.read('D:\topic_2\ml_input_data\hru_scale/hru_scale_SWAT_gwflow_input_output_data_2000_2015.csv')