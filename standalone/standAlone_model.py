
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib



from python_scripts.create_df_for_ml_input import create_df_for_ml_input
from python_scripts.calculate_model_performance import calculate_model_performance


def standAlone_model(csv_all_years_dir, data_dir, model_dir,error_metrics_dir, ml_plots):
    """
    This function trains and test model
    Parameters
    ----------
    csv_all_years_dir: csv directory path for predictor and target variables 
    data_dir: Directory path to save train and test data
    model_dir: Directory path to save trained model
    ml_plots: Directory path to save ml analysis plots plotted using different variables using different conditionss
    Returns
    -------
     Trained model 

    """
    
    ml_input_all_df = create_df_for_ml_input(csv_all_years_dir)
   

    x_data = ml_input_all_df.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]', 'recharg_frac'], axis=1)

    y_data = ml_input_all_df[['gridid', 'year', 'recharge [mm/yr]']]
    
    print(x_data.columns)
    print(y_data.columns)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    x_train_data = x_train.drop(columns= ['gridid', 'year'])
    x_test_data = x_test.drop(columns= ['gridid', 'year'])
    y_train_data = y_train['recharge [mm/yr]']
    y_test_data = y_test['recharge [mm/yr]']
    
    
    x_train_data.to_csv(data_dir  + 'train_data/x_train_data.csv', index=False)
    y_train_data.to_csv(data_dir  + 'train_data/y_train_data.csv', index=False)
    x_test_data.to_csv(data_dir  + 'test_data/x_test_data.csv', index=False)
    y_test_data.to_csv(data_dir  + 'test_data/y_test_data.csv', index=False)

   
    rf = RandomForestRegressor(n_estimators=500,  max_depth= None,
                                 min_samples_split= 3,
                                 min_samples_leaf= 2,
                                 max_features= 15, random_state=42)
    import time
    start_time = time.time()

    rf.fit(x_train_data, y_train_data)
    
    with open(model_dir + 'ranFor_model.joblib','wb') as f:
        joblib.dump(rf,f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Fitting the RandomForestRegressor model took: ', elapsed_time)
    
    calculate_model_performance(rf, x_train_data, x_test_data, y_train_data, y_test_data, error_metrics_dir)
    print('Machine learning training and evaluation completed! ')
    
    
    y_train_pred = rf.predict(x_train_data)
    y_test_pred = rf.predict(x_test_data)

    swat_rech_df = ml_input_all_df[['gridid', 'year', 'recharge [mm/yr]','Precip']]
    ml_rech_df = ml_input_all_df[['gridid', 'year']]
    
    y_train_pred_df = pd.DataFrame(y_train_pred, columns=['ml_recharge [mm/yr'] )
    y_test_pred_df  = pd.DataFrame(y_test_pred, columns= ['ml_recharge [mm/yr'])
    ml_rech_concat_df = pd.concat([y_train_pred_df ,y_test_pred_df ],axis=0)
    ml_rech_df['ml_recharge [mm/yr'] = ml_rech_concat_df['ml_recharge [mm/yr'].to_list()
    swat_ml_merged_df = pd.merge(swat_rech_df,ml_rech_df, how='inner', on =['gridid', 'year'] )
    swat_ml_merged_df['residuals'] = swat_ml_merged_df['recharge [mm/yr]'] - swat_ml_merged_df['ml_recharge [mm/yr]']
    swat_ml_merged_df.to_csv(data_dir + 'swat_ml_predicted_all_years_recharge.csv', index=False)
        
    return rf
        