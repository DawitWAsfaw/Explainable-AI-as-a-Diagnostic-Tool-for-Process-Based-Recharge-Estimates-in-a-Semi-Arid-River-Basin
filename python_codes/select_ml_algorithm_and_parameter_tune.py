import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb




#====================================================================================
def select_ml_algorithm(ml_input_all_df, model_csv_path, numOfsamples=100000):
    """
    This function evaluate different machine learning algorithm using lazypredict
    Parameters
    ----------
    ml_input_all_df : Machine learning input dataframe containing predictor and target varibles
    model_csv_path : Directory path where ranking of models saved as csv file
    numOfsamples: Total number of observation data used to train and test different algorithms

    Returns
    -------
    None.

    """
    
    ml_input_all_df_100k = ml_input_all_df.sample(n=numOfsamples, random_state=42)

    x_data =  ml_input_all_df_100k.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]', 'recharg_frac'], axis=1)
    print(x_data.columns)
    y_data =  ml_input_all_df_100k[['gridid', 'year', 'recharge [mm/yr]']]
    print(y_data.columns)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    X_train_data = X_train.drop(columns= ['gridid', 'year']).astype(np.float32)
    X_test_data = X_test.drop(columns= ['gridid', 'year']).astype(np.float32)
      
    y_train_data = y_train['recharge [mm/yr]']
    y_test_data = y_test['recharge [mm/yr]']
    
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "KNeighbors Regressor": KNeighborsRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor()
    }

    # Evaluate models
    results = []
    for name, model in models.items():
        print('Running: ', name)
        model.fit( X_train_data, y_train_data)
        y_pred = model.predict(X_test_data)
        print(y_pred.shape)
        mae = mean_absolute_error(y_test_data, y_pred)
        mse = mean_squared_error(y_test_data, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_data, y_pred)
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })
    
   

    # Convert results to DataFrame and sort by R²
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    print(results_df)
    results_df.to_csv(model_csv_path + f'models_error_metrics_numOfsamples_{numOfsamples}.csv',index=False)

def parameter_tuning_rf_model(ml_input_all_df, parameter_tune_dir, folds=5, numOfsamples=100000):
    """
    Random Forest model parameter Tuning  using GridSearchCV.
    Parameters:
    ml_input_all_df : A dataframe contains predictor and target 
    folds : Number of folds in K Fold CV. Default set to 5. 
    numOfsamples: Total number of observation data used for model training for parameter value determination
    Returns : Best parameter values.
    """

    
    param_grid_values = {
        'n_estimators': [100, 200,500],
        'max_depth': [None, 6, 10],
        'min_samples_split': [3,5, 10],
        'min_samples_leaf': [2, 4,6],
        'max_features': [5,10,15,20]
    }
    
    ml_input_all_df_100k = ml_input_all_df.sample(n=numOfsamples, random_state=42)

    x_data =  ml_input_all_df_100k.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]', 'recharg_frac' ], axis=1)

    y_data =  ml_input_all_df_100k[['gridid', 'year', 'recharge [mm/yr]']]

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    
    X_train_data = X_train.drop(columns= ['gridid', 'year']).astype(np.float32)
    y_train_data = y_train['recharge [mm/yr]']
    
    rf = RandomForestRegressor(random_state=0,n_jobs=-1)
    
    kfold = KFold(n_splits=folds,shuffle=True, random_state=0)
    
    
    
    CV = GridSearchCV(estimator=rf , 
                            param_grid =param_grid_values, 
                            cv=kfold, 
                            verbose=1,  
                            n_jobs=-1,

    scoring={'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2'},
    refit='r2', 
    return_train_score=True)
        
    
    CV.fit(X_train_data, y_train_data)
    df_cv_results = pd.DataFrame(CV.cv_results_)
    df_cv_results.to_csv(parameter_tune_dir + 'cv_results_rf2.csv', index=False)

    print('\n')
    print(CV.best_params_)
    print('\n')
    
    print('mean_train_r2', round(CV.cv_results_['mean_train_r2'][CV.best_index_], 2))
    print('mean_test_r2', round(CV.cv_results_['mean_test_r2'][CV.best_index_], 2))
 
    optimized_param_dict = {'n_estimators': CV.best_params_['n_estimators'],
                            'max_depth': CV.best_params_['max_depth'],
                            'min_samples_split':CV.best_params_['min_samples_split'],
                            'min_samples_leaf': CV.best_params_['min_samples_leaf'],
                            'max_features': CV.best_params_['max_features']}
           
      
        
    return optimized_param_dict  