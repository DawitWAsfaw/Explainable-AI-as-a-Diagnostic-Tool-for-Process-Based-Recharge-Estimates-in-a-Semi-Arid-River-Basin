import joblib
import pandas as pd

import rasterio as rio
from rasterio import features
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
import ale_py
from sklearn.model_selection import train_test_split
from python_scripts.create_df_for_ml_input import create_df_for_ml_input


csv_all_years_dir='D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_input_data/all_years_csv/'
ml_input_all_df = create_df_for_ml_input(csv_all_years_dir)

ml_input_all_df[ml_input_all_df['Precip']>=500]

condition_based_df_list = [ml_input_all_df[ml_input_all_df['Precip'] >  500],
                  ml_input_all_df[ml_input_all_df['cultivatedCrops']>0.70],
                  ml_input_all_df[ml_input_all_df['sandPcnt']> 65],
                  ml_input_all_df[ml_input_all_df['grasslandOrHerbaceous']>0.7],
                  ml_input_all_df[ml_input_all_df['slope']>12],
                  ml_input_all_df[ml_input_all_df['evergreenForest']>0.75],
                  ml_input_all_df[ml_input_all_df['ET']>760],
                  ml_input_all_df[ml_input_all_df['emergentHerbaceousWetlands']>0.2]]

#==============================================================================================================



def burn_raster_from_shapefile(shp,rast,fname,colname,dtype='int16',nodata=0):
    meta = rast.meta
    meta['dtype'] = dtype
    meta['nodata'] = nodata
    
    with rio.open(fname, 'w+', **meta) as out:
        out_arr = out.read(1)    
        shapes = ((geom,value) for geom, value in zip(shp.geometry, shp[colname]))
        
        burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write(burned,1)
        
    return(burned)

larb_grid_shapfile = gpd.read_file('D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/grid/larb_grid.shp')
larb_ref_raster = rio.open('D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/ref_raster/larb_ref_raster.tif')

output_rast_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/high_recharge_conditon_rast_plots/'


col_names_list = ['Precip','cultivatedCrops','sandPcnt','grasslandOrHerbaceous','slope','evergreenForest','ET','emergentHerbaceousWetlands']

for df,col_name in zip(condition_based_df_list,col_names_list): 
    print(col_name)
    for year in df['year'].unique():
        print(year)
        yrly_df = df[df['year']==year]
        merged_gdf = larb_grid_shapfile.merge(yrly_df, on='gridid')
        fname =output_rast_dir +  f'{col_name}_{year}.tif'
        shp =  merged_gdf
        rast = larb_ref_raster
        colname = col_name
        burn_raster_from_shapefile(shp,rast,fname,colname,dtype='int16',nodata=-9999)
        


#==============================================================================================================
x_data =ml_input_all_df.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]', 'recharg_frac'], axis=1)

y_data = ml_input_all_df[['gridid', 'year', 'recharge [mm/yr]']]


X_train, X_test, y_train, y_test = train_test_split(
    x_data ,y_data , test_size=0.3, random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    x_data ,y_data , test_size=0.3, random_state=42
)
x_train_df = X_train.drop(['gridid', 'year'], axis =1) 
x_test_df = X_test.drop(['gridid', 'year'], axis =1) 
y_train_df = y_train['recharge [mm/yr]']
y_test_df = y_test['recharge [mm/yr]']

model_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/model/'
rf = joblib.load(model_dir + 'ranFor_model_500m_grid_mm_yr_12_02_2025.joblib')

y_train_pred = rf.predict(x_train_df)
y_test_pred = rf.predict(x_test_df)
y_train_pred_df = X_train[['gridid', 'year']]
y_train_pred_df['Train Predicted_recharge [mm/yr]'] = y_train_pred
y_test_pred_df = y_test[['gridid', 'year']]
y_test_pred_df['Test Predicted_recharge [mm/yr]'] = y_test_pred 

y_train_pred_df = y_train_pred_df.rename(columns ={'Train Predicted_recharge [mm/yr]':'Predicted_recharge [mm/yr]' })
y_test_pred_df = y_test_pred_df .rename(columns ={'Test Predicted_recharge [mm/yr]':'Predicted_recharge [mm/yr]' })

Predicted_recharge_df = pd.concat([y_train_pred_df,y_test_pred_df])  