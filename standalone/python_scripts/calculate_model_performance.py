import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    train_me = np.mean(y_train_data-y_train_pred)

    test_mse = mean_squared_error(y_test_data, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_data, y_test_pred)
    test_mae = mean_absolute_error(y_test_data, y_test_pred)
    test_me = np.mean(y_test_data-y_test_pred)

    print("\nTraining Performance:\n")
    print(f"  MSE:  {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.2f}")
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

    randomForest_model_evaluation = pd.DataFrame(score_metric_dic)
    randomForest_model_evaluation.to_csv( error_metrics_dir + 'standalone_model_error_metrics.csv', index= False)


