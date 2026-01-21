import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from python_scripts.ml_analysis_plots import burn_raster_from_shapefile
from python_scripts.create_df_for_ml_input import create_df_for_ml_input


def calculate_model_performance(rf, x_train_data, x_test_data, y_train_data, y_test_data, error_metrics_dir):
    """
    This function calculates error metrics using train and test data.(MSE,RMSE,R² and MAE) 
    Parameters
    ----------
    rf : Trained Random Forest model
    x_train_data : Training data  containing predictor features used to fit a model
    x_test_data : Testing data  contain predictor features to test a model
    y_train_data : Training data containing target variable used to fit a model
    y_test_data : Testing data containing target variable used to test a model
    error_metrics_dir :  Directory path to save the calculated error metrics
    Returns
    -------
    None.

    """
    
    y_train_pred = rf.predict(x_train_data)
    y_test_pred = rf.predict(x_test_data)


    train_mse = mean_squared_error(y_train_data, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_data, y_train_pred)
    train_mae = mean_absolute_error(y_train_data, y_train_pred)

    test_mse = mean_squared_error(y_test_data, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_data, y_test_pred)
    test_mae = mean_absolute_error(y_test_data, y_test_pred)

    print("\nTraining Performance:\n")
    print(f"  MSE:  {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.2f}")
    print(f"  MAE:   {train_mae:.2f}")

    print("\nTesting Performance:\n")
    print(f"  MSE:  {test_mse:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.2f}")
    print(f"  MAE:  {test_mae:.2f}")

    score_metric_dic =  {
     'Model': '',
     'Train R²':[],
     'Train MSE(mm)' : [],
     'Train RMSE(mm)' : [],
     'Train  MAE(mm)' : [], 
     
     'Test R²' : [],
     'Test MSE(mm)' : [],
     'Test RMSE(mm)' : [],
     'Test MAE(mm)' : [],
     } 

    score_metric_dic['Train R²'].append(train_r2)
    score_metric_dic['Train MSE(mm)'].append(train_mse)
    score_metric_dic['Train RMSE(mm)'].append(train_rmse)
    score_metric_dic['Train  MAE(mm)'].append(train_mae)

    score_metric_dic['Test R²' ].append(test_r2)
    score_metric_dic['Test MSE(mm)'].append(test_mse)
    score_metric_dic['Test RMSE(mm)'].append(test_rmse)
    score_metric_dic['Test MAE(mm)'].append(test_mae)
    
    score_metric_dic['Model'] = 'RandomForestRegressor'

    randomForest_model_evaluation = pd.DataFrame(score_metric_dic)
    randomForest_model_evaluation.to_csv( error_metrics_dir + 'rf_model_error_metrics.csv', index= False)

def test_model_using_case_study_data(rf, main_data_dir,ref_raster_file, grid_500m_shapefile, ml_analysis_output_dir):
    """
    This function evaluates model's ability to generalize in when tested on new places
    Parameters
    ----------
    rf: trained and tested model
    ref_raster_file: Reference raster file used to create prediction raster file
    grid_500m_shapefile: Shapefile contain 500m polygon for area of interest
    main_data_dir : Directory path  containing csv files all the predictor variables and target variable
    ml_analysis_output_dir : Directory path to save prediction csv, raster and error metrics
    Returns
    -------
    None.

    """
    
    ml_input_all_df = create_df_for_ml_input(main_data_dir)
    print(ml_input_all_df.columns)
    renaming_dic = { 
                    'pct_developed_lowIntensity':'developedlowIntensity',
                    'pct_developed_mediumIntensity':'developedMediumIntensity',
                    'pct_evergreenForest':'evergreenForest',
                    'pct_grasslandOrHerbaceous':'grasslandOrHerbaceous', 
                    'pct_cultivatedCrops':'cultivatedCrops',
                    'pct_woodyWetlands':'woodyWetlands', 
                    'pct_emergentHerbaceousWetlands':'emergentHerbaceousWetlands',
                    'awcPcnt_':'awcPcnt',
                    'awsPcnt_':'awsPcnt',
                    'clayPcnt_':'clayPcnt',
                    'dTs_':'dTs',
                    'drainage_density_':'drainage_density',
                    'floodfclPcnt_':'floodfclPcnt',
                    'ksatPcnt_':'ksatPcnt',
                    'pondfclPcnt_':'pondfclPcnt',
                    'recharge':'recharge [m3/day]',
                    'sandPcnt_':'sandPcnt',
                    'siltPcnt_':'siltPcnt',
                    'SPI_':'SPI',
                    'surftext_':'surftextPcnt',
                    'TRI_':'TRI',
                    'TWI':'TWI',
                    'depth2Water_tbl':'depth2Water_tbl',
                    'k_mday_':'k_mday',
                    'sy_':'sy',
                    'thick_m_':'thick_m',
                    'openET': 'ET',
                    'ppt':'Precip',
                    'tmin':'Tmin',
                    'tmax':'Tmax',
                    'slope_': 'slope'
                                            }
    
    
    ml_input_all_df = ml_input_all_df.rename(columns=renaming_dic)
    ml_input_all_df =   ml_input_all_df[ ml_input_all_df['year']>=2008]
    print(ml_input_all_df.columns)
    grid_shapfile = gpd.read_file(grid_500m_shapefile)
    ref_raster = rio.open(ref_raster_file)
    
    x_data = ml_input_all_df.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]','recharg_frac'], axis=1)

    y_data = ml_input_all_df[['gridid', 'year' ,'recharge [mm/yr]']]
    

    predictors_data = x_data.drop(columns= ['gridid', 'year'])
    target_data = y_data['recharge [mm/yr]']
    
    case_study_pred = rf.predict(predictors_data)
    

    case_study_mse = mean_squared_error(target_data, case_study_pred)
    case_study_rmse = np.sqrt(case_study_mse)
    case_study_r2 = r2_score(target_data, case_study_pred)
    case_study_mae =mean_absolute_error(target_data, case_study_pred)


    print("\nTesting Performance:\n")
    print(f"  MSE:  {case_study_mse:.2f}")
    print(f"  RMSE: {case_study_rmse:.2f}")
    print(f"  R²:   {case_study_r2:.2f}")
    print(f"  MAE:  {case_study_mae:.2f}")

    score_metric_dic =  {
     'Model': '',
     
     'Case study R²' : [],
     'Case study MSE(mm)' : [],
     'Case study RMSE(mm)' : [],
     'Case study MAE(mm)' : [],
     } 

    score_metric_dic['Case study R²' ].append(case_study_r2)
    score_metric_dic['Case study MSE(mm)'].append(case_study_mse)
    score_metric_dic['Case study RMSE(mm)'].append(case_study_rmse)
    score_metric_dic['Case study MAE(mm)'].append(case_study_mae)
    
    score_metric_dic['Model'] = 'RandomForestRegressor'
    
    error_metrics_dir = ml_analysis_output_dir + 'error_metrics/'
    randomForest_model_evaluation = pd.DataFrame(score_metric_dic)
    randomForest_model_evaluation.to_csv( error_metrics_dir + 'rf_model_error_metrics_frac.csv', index= False)
    
    case_study_prediction_df = ml_input_all_df[['gridid', 'year','recharge [mm/yr]']]
    case_study_prediction_df['Predicted_recharge [mm/yr]'] = case_study_pred 
    case_study_prediction_df.to_csv(ml_analysis_output_dir + 'predictions/grid_500m/tif/ml_mm/case_study_prediction_frac_df.csv',index= False)
    spatial_500m_plots_dir = ml_analysis_output_dir + 'predictions/grid_500m/tif/ml_mm/'
    for year in case_study_prediction_df['year'].unique():
        print(year)
        predicted_df = case_study_prediction_df[case_study_prediction_df['year']==year]
        merged_gdf = grid_shapfile.merge(predicted_df, on='gridid')
        fname = f'{spatial_500m_plots_dir}predicted_recharge_frac_{year}.tif'
        shp =  merged_gdf
        rast = ref_raster
        colname = 'Predicted_recharge [mm/yr]'
        burn_raster_from_shapefile(shp,rast,fname,colname,dtype='float32',nodata=-9999)

