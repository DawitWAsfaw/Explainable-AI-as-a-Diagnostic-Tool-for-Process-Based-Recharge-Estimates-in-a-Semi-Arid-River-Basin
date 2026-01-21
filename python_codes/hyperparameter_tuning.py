import pprint

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb



def parameter_tuning(x_train_data,y_train_data,target_variable,parameter_tune_dir, folds=5):
    """
    Random Forest model parameter Tuning  using GridSearchCV.
    Parameters:
    x_train_data, y_train_data : x_train_data (predictor) and y_train_data (target) arrays from split_train_test_ratio 

    folds : Number of folds in K Fold CV. Default set to 5. 
    Returns : Best parameter values.
    """

    
    param_grid_values = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 6, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8]
    }

  
    rf = RandomForestRegressor(random_state=0,n_jobs=-1)
    
    kfold = KFold(n_splits=folds,shuffle=True, random_state=0)
    
    
    
    CV = GridSearchCV(estimator=rf , 
                            param_grid =param_grid_values, 
                            cv=kfold, 
                            verbose=1, 
                            random_state=0, 
                            n_jobs=-1,

    scoring={'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2'},
    refit='r2', 
    return_train_score=True)
        
    
    CV.fit(x_train_data, y_train_data)
    df_cv_results = pd.DataFrame(CV.cv_results_)
    df_cv_results.to_csv(parameter_tune_dir + 'cv_results_{}.csv'.format(target_variable), index=False)

    print('\n')
    print('r2', '\n')
    print(CV.best_params_)
    print('\n')
    print('r2', round(CV.cv_results_['r2'][CV.best_index_], 2))
    print('r2', round(CV.cv_results_['r2'][CV.best_index_], 2))
    
 
    optimized_param_dict = {'n_estimators': CV.best_params_['n_estimators'],
                            'max_depth': CV.best_params_['max_depth'],
                            'min_samples_split':CV.best_params_['min_samples_split'],
                            'min_samples_leaf': CV.best_params_['min_samples_leaf'],
                            'max_features': CV.best_params_['max_features']}
           
      
        
    return optimized_param_dict
    
    
   