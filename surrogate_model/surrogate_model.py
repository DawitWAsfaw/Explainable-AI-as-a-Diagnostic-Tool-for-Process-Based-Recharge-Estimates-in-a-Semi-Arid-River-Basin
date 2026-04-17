import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib



def surrogate_ml_model(swat_df_path, model_dir, plots_dir):
    """
    Parameters
    ----------
    swat_df_path : Directory path for SWAT+ data
    model_dir : Directory path where model results will be saved
    plots_dir : Directory path where model results plots will be saved
    Returns
    -------
    None.

    """
    ml_input_all_df_swat = pd.read_csv(swat_df_path)
    
    drop_columns = ['id', 'name', 'area', 'lat', 'lon', 'elev', 'hru', 'wst',
                    'cst', 'ovfl', 'rule', 'out_tot', 'jday', 'mon', 'day',  'unit', 'wet_stor [mm]']
    
    
    ml_input_all_df_swat = ml_input_all_df_swat.drop(columns = drop_columns, axis=1)
    ml_input_all_df_swat = ml_input_all_df_swat[ml_input_all_df_swat['year']>=2002]
    
    
    x_data =ml_input_all_df_swat.drop(['perc [mm]'], axis=1)
    
    x_data.columns
    
    y_data = ml_input_all_df_swat[['gis_id', 'year','perc [mm]']]
    
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data , 
                                                        random_state=42,
                                                        test_size=0.30, 
                                                        shuffle=True)
    
    
    
    x_train_df = x_train.drop(['gis_id', 'year'], axis =1) 
    x_test_df = x_test.drop(['gis_id', 'year'], axis =1) 
    y_train_df = y_train['perc [mm]']
    y_test_df = y_test['perc [mm]']
    
    
    
    surrogate_model =RandomForestRegressor()
    
    surrogate_model.fit(x_train_df, y_train_df)
    
    with open(model_dir + 'model/surrogate_model.joblib','wb') as f:
        joblib.dump(surrogate_model, f)
    
    y_train_pred = surrogate_model.predict(x_train_df)
    y_test_pred = surrogate_model.predict(x_test_df)
      
      
    train_mse = mean_squared_error(y_train_df, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_df, y_train_pred)
    train_mae = mean_absolute_error(y_train_df, y_train_pred)
    train_me = np.mean(y_train_df-y_train_pred)
      
    test_mse = mean_squared_error(y_test_df, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_df, y_test_pred)
    test_mae = mean_absolute_error(y_test_df, y_test_pred)
    test_me = np.mean(y_test_df-y_test_pred)
      
    print("\nTraining Performance:\n")
    print(f"  MSE:  {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.2f}") # The formula used for R² in sklearn is same as NSE
    print(f"  MAE:   {train_mae:.2f}")
    print(f"  ME:   {train_me:.2f}")
      
    print("\nTesting Performance:\n")
    print(f"  MSE:  {test_mse:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.2f}")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  ME:  {test_me:.2f}")
      
    score_metric_dic =  {
     'Model': '',
     'Train R²':[],
     'Train MSE(mm)' : [],
     'Train RMSE(mm)' : [],
     'Train  MAE(mm)' : [], 
     'Train  ME(mm)' : [],
     
     'Test R²' : [],
     'Test MSE(mm)' : [],
     'Test RMSE(mm)' : [],
     'Test MAE(mm)' : [],
     'Test  ME(mm)' : [],
     } 
      
    score_metric_dic['Train R²'].append(train_r2)
    score_metric_dic['Train MSE(mm)'].append(train_mse)
    score_metric_dic['Train RMSE(mm)'].append(train_rmse)
    score_metric_dic['Train  MAE(mm)'].append(train_mae)
    score_metric_dic['Train  ME(mm)'].append(train_me)
      
    score_metric_dic['Test R²' ].append(test_r2)
    score_metric_dic['Test MSE(mm)'].append(test_mse)
    score_metric_dic['Test RMSE(mm)'].append(test_rmse)
    score_metric_dic['Test MAE(mm)'].append(test_mae)
    score_metric_dic['Test  ME(mm)'].append(test_me)
      
    score_metric_dic['Model'] = 'RandomForestRegressor'
    round_places = 2    
      
    score_metric_df = pd.DataFrame(score_metric_dic).round(round_places) 
    score_metric_df.to_csv(model_dir + 'surrogate_model_error_metrices.csv',index=False)
    
  
    swat_rech_df =  ml_input_all_df_swat[['gis_id', 'year','perc [mm]']]
    ml_rech_df =  ml_input_all_df_swat[['gis_id', 'year']]
    
    y_train_pred_df = pd.DataFrame(y_train_pred, columns=['ml_recharge [mm/yr'] )
    y_test_pred_df  = pd.DataFrame(y_test_pred, columns= ['ml_recharge [mm/yr'])
    ml_rech_concat_df = pd.concat([y_train_pred_df ,y_test_pred_df ],axis=0)
    ml_rech_df['ml_recharge [mm/yr'] = ml_rech_concat_df['ml_recharge [mm/yr'].to_list()
    swat_ml_predicted_df = pd.merge(swat_rech_df,ml_rech_df, how='inner', on =['gridid', 'year'] )
    swat_ml_predicted_df['residuals'] = swat_ml_predicted_df['recharge [mm/yr]'] - swat_ml_predicted_df['ml_recharge [mm/yr]']
    swat_ml_predicted_df.to_csv( model_dir + 'surrogate_model_swat_ml_predicted.csv', index=False)
    
