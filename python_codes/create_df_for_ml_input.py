import numpy as np
import pandas as pd
from glob import glob
from functools import reduce
# from ml_analysis_plots import create_pd_for_different_predictor_variables_with_condition

# import polars as pl
import  matplotlib.pyplot as plt

def create_df_for_ml_input(csv_all_years_dir):
    """
    This function merges  variable csv files into one main Dataframe
    Parameters
    ----------
    csv_all_years_dir : Directory path for csv files for all the input variables
    Returns
    -------
     Dataframe
    """
    pattern = '*.csv'
    hydrology_factor_var_names = ['distanceTo_stream','drainage_density','TRI','TWI','SPI','slope'] 
    hydrologeology_var_names = ['k_mday','sy','depth2Water_tbl','thick_m']
    soil_phy_property_var_names = ['aws', 'clay','ksat','sand','silt','awc','FloodFCls', 'PondFCls'] # 'floodfcl','pondfcl',
    climate_data_var_names = ['openET','ppt', 'tmax','tmin']
    lulc_name = 'Annual_NLCD_LndCov'
    
    hydrology_factor_df  = pd.DataFrame()
    for var in hydrology_factor_var_names:
        pattern = f'{var}*.csv'
        for file in glob(csv_all_years_dir + pattern):
            print('Processing file: ', file,'\n')
            df = pd.read_csv(file)
            if  hydrology_factor_df .empty:
                hydrology_factor_df = df
            else:
                hydrology_factor_df = pd.merge(hydrology_factor_df, df, how='inner', on=['gridid','year'])
    print(hydrology_factor_df.shape)
    
    #===============================================================================================================
    hydrologeology_df = pd.DataFrame()
    for var in hydrologeology_var_names:
        pattern = f'{var}*.csv'
        for file in glob(csv_all_years_dir + pattern):
            print('Processing file: ', file, '\n')
            df = pd.read_csv(file)
            if  hydrologeology_df.empty:
                hydrologeology_df = df
            else:
                hydrologeology_df = pd.merge(hydrologeology_df, df, how='inner', on=['gridid','year'])
    print(hydrologeology_df.shape)     
    #===============================================================================================================
    soil_phy_propert_df  = pd.DataFrame()
    for var in soil_phy_property_var_names :
        pattern = f'{var}*.csv'
        for file in glob(csv_all_years_dir + pattern):
            print('Processing file: ', file, '\n')
            df = pd.read_csv(file)
            if  soil_phy_propert_df.empty:
                soil_phy_propert_df = df
            else:
                soil_phy_propert_df = pd.merge(soil_phy_propert_df , df, how='inner', on=['gridid','year'])

    print(soil_phy_propert_df.shape)
    #===============================================================================================================
    climate_data_df  = pd.DataFrame()
    for var in climate_data_var_names :
        pattern = f'{var}*.csv'
        for file in glob(csv_all_years_dir + pattern):
            print('Processing file: ', file, '\n')
            df = pd.read_csv(file)
            if  climate_data_df.empty:
                climate_data_df  = df
            else:
                climate_data_df  = pd.merge(climate_data_df, df, how='inner', on=['gridid','year'])
    print(climate_data_df.shape)
    #===============================================================================================================
    print('Processing file: ',csv_all_years_dir + f'{lulc_name}_all.csv', '\n')
    lulc_data_df = pd.read_csv(csv_all_years_dir + f'{lulc_name}_all.csv', low_memory=False)
    columns_to_remove_from_lulc = ['majority','unique', 'pct_openWater','pct_perennialIceOrSnow', 'pct_developed_openSpace',
           'pct_developed_highIntensity', 'pct_barrenLand', 'pct_deciduousForest', 'pct_mixedForest', 'pct_dwarfScrub',
           'pct_shrubOrScrub', 'pct_sedgeOrHerbaceous', 'pct_lichens', 'pct_moss', 'pct_pastureOrHay']

    lulc_data_df_colm_removed = lulc_data_df.drop(columns=columns_to_remove_from_lulc)
    print(lulc_data_df_colm_removed.shape) 

    colms = lulc_data_df_colm_removed.columns[2:]
    invalid_values = ['--']
    lulc_data_df_colm_removed[colms] = lulc_data_df[colms].replace(invalid_values, np.nan)
    lulc_data_df_colm_removed[colms]= lulc_data_df_colm_removed[colms].astype('float32')

    #===============================================================================================================
    print('Processing file: ', csv_all_years_dir + 'recharge_all.csv' , '\n')
    recharge_data_df = pd.read_csv(csv_all_years_dir + 'recharge_all.csv', low_memory=False)
    print(recharge_data_df.shape )
    #===============================================================================================================
    # hydrology_factor_df_2015 = hydrology_factor_df[hydrology_factor_df['year']<=2015]
    # soil_phy_propert_df_2015  = soil_phy_propert_df[soil_phy_propert_df['year']<=2015]
    # climate_data_df_2015  = climate_data_df[climate_data_df['year']<=2015]
    # lulc_data_df_2015  = lulc_data_df_colm_removed[lulc_data_df_colm_removed['year']<=2015]
    # hydrologeology_df_2015 = hydrologeology_df[hydrologeology_df['year']<=2015]

    #===============================================================================================================
    dfs = [hydrology_factor_df , soil_phy_propert_df , climate_data_df ,lulc_data_df_colm_removed ,hydrologeology_df , recharge_data_df] 
    # dfs = [hydrology_factor_df_2015 , soil_phy_propert_df_2015 , climate_data_df_2015 ,lulc_data_df_2015 ,hydrologeology_df_2015 , recharge_data_df] 
    # Merge on common keys
    ml_input_df = reduce(lambda left, right: pd.merge(left, right, on=['gridid','year'], how='inner'), dfs)
    ml_input_df.shape
    ml_input_df.info()

    cols = ml_input_df.columns[2:]
    ml_input_df[cols] = (ml_input_df[cols].apply(pd.to_numeric, errors='coerce').where(lambda x: x > -3e38, np.nan))

    #===============================================================================================================
    # Function to summarize invalid values
    def summarize_invalid_values(df):
        summary = {}
        for col in df.columns:
            zeros = (df[col] == 0).sum()
            inf_count = np.isinf(df[col]).sum()
            neg_inf_count = (df[col] == -np.inf).sum()
            nan_count = df[col].isna().sum()
            big_num = (df[col] ==-3.4028235E+38 ).sum()
            summary[col] = {
                'inf': inf_count,
                '-inf': neg_inf_count,
                'NaN': nan_count,
                'big_num':big_num,
                'zeros':zeros 
            }
        return pd.DataFrame(summary).T

    summary_df = summarize_invalid_values(ml_input_df)
    print(summary_df,'\n')

    ml_input_df_cleaned = ml_input_df.dropna()
    summary_df = summarize_invalid_values(ml_input_df_cleaned)
    print(summary_df,'\n')
    ml_input_df_cleaned.shape
    #====================================================================================
    renaming_dic = { 
                  'pct_developed_lowIntensity':'developedlowIntensity',
                  'pct_developed_mediumIntensity':'developedMediumIntensity',
                  'pct_evergreenForest':'evergreenForest',
                  'pct_grasslandOrHerbaceous':'grasslandOrHerbaceous', 
                  'pct_cultivatedCrops':'cultivatedCrops',
                  'pct_woodyWetlands':'woodyWetlands', 
                  'pct_emergentHerbaceousWetlands':'emergentHerbaceousWetlands',
                  'awcPcnt_':'awcPcnt',
                  'awsPcnt_':'awsPcnt_',
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
                  'TWI_':'TWI',
                  'depth2Water_tbl_':'depth2Water_tbl',
                  'k_mday_':'k_mday',
                  'sy_':'sy',
                  'thick_m_':'thick_m',
                  'openET': 'ET',
                  'ppt':'Precip',
                  'tmin':'Tmin',
                  'tmax':'Tmax',
                  'slope_': 'slope'
                                          }
    ml_input_all_df = ml_input_df_cleaned.rename(columns=renaming_dic)
    ml_input_all_df.columns
    summary_df = summarize_invalid_values(ml_input_all_df)
    print('\n', summary_df,'\n')
    ml_input_all_df['depth2Water_tbl'] = ml_input_all_df['depth2Water_tbl'] * 0.3048
    min(ml_input_all_df['depth2Water_tbl'])
    max(ml_input_all_df['depth2Water_tbl'])
    ml_input_all_df['area_sqrtm'] = 500*500
    ml_input_all_df['mTomm'] = 1000
    ml_input_all_df['dayInayr'] = 365.25   
    ml_input_all_df['recharge [mm/yr]'] = (ml_input_all_df['recharge [m3/day]']/ml_input_all_df['area_sqrtm']) *(ml_input_all_df['mTomm'] *ml_input_all_df['dayInayr'])
    ml_input_all_df['recharg_frac'] = ml_input_all_df['recharge [mm/yr]']/ml_input_all_df['Precip']
    
    return ml_input_all_df



# csv_all_years_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_input_data/all_years_csv/'

# ml_input_all_df = create_df_for_ml_input(csv_all_years_dir)


# ml_plots = 'D:/deep_percolotation_ml_project/ml_analysis_results/plots/ml_plots/'

# create_pd_for_different_predictor_variables_with_condition(ml_input_all_df,ml_plots , numOfsample_data =50000)

def eda_hist_plot_of_variables(ml_input_all_df, eda_plots_path):
    """
    This functions creates histogram plots of predictor variables
    Parameters
    ----------
    ml_input_all_df : Dataframe contains observation values of predictor variables
    eda_plots_path : Directory path where the plots are saved. Datatype: String
    Returns
    -------
        None.

    """
    
    ml_input_var_group_dict = {
        'climate': ['ET', 'Precip', 'Tmax', 'Tmin'],
        'soil_phy_prop': ['aws', 'clay', 'floodfcl', 'ksat', 'pondfcl', 'sand', 'silt','awc'],
        'hydro_factors': ['dTs', 'drainage_density', 'TRI', 'TWI', 'SPI', 'slope'],
        'lulc_classes':['evergreenForest', 'grasslandOrHerbaceous', 'cultivatedCrops','emergentHerbaceousWetlands'],
        'hydrogeol_var':['k_mday', 'sy', 'depth2Water_tbl', 'thick_m']
        }

    for features_key in ml_input_var_group_dict.keys():
        print(features_key)
        df = ml_input_all_df[ml_input_var_group_dict[features_key]]
        if features_key == 'climate'or features_key == 'lulc_classes' or features_key == 'hydrogeol_var':
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)
            for idx, col_name  in enumerate(ml_input_var_group_dict[features_key]):
                ax = axes.ravel()[idx]
                df[col_name]=  df[col_name].round(3)
                ax.hist( df[col_name], bins=10,  color='lightblue', edgecolor='black', linewidth=2)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title(col_name,fontsize=16)
                
            fig.savefig(( eda_plots_path +  f'{features_key}_eda_histplots.png'), dpi=600)   
            plt.show()
                
        elif  features_key == 'soil_phy_prop':
              df = ml_input_all_df[ml_input_var_group_dict[features_key]]
              df_group1 = df[df.columns[0:4]]
              fig, axes = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)

              for idx, col_name in enumerate(df_group1.columns):
                  ax = axes.ravel()[idx]
                  df[col_name]=  df[col_name].round(3)
                  ax.hist( df[col_name], bins=10,  color='lightblue', edgecolor='black', linewidth=2)
                  ax.set_ylabel('Count', fontsize=12)
                  ax.set_title(col_name,fontsize=16)
                  
              fig.savefig(( eda_plots_path +  f'{features_key}_group1_eda_histplots.png'), dpi=600)   
              plt.show()
              
              df_group2 = df[df.columns[4:8]]
              fig, axes = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)
              for idx, col_name in enumerate(df_group2.columns):
                  ax = axes.ravel()[idx]
                  df[col_name]=  df[col_name].round(3)
                  ax.hist( df[col_name], bins=10,  color='lightblue', edgecolor='black', linewidth=2)
                  ax.set_ylabel('Count', fontsize=12)
                  ax.set_title(col_name,fontsize=16)
                  
              fig.savefig(( eda_plots_path +  f'{features_key}_group2_eda_histplots.png'), dpi=600)   
              plt.show()   
             
        elif  features_key == 'hydro_factors':
            df = ml_input_all_df[ml_input_var_group_dict[features_key]]
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), tight_layout=True)
            for idx, col_name in enumerate(df.columns):
                ax = axes.ravel()[idx]
                df[col_name]=  df[col_name].round(3)
                ax.hist( df[col_name], bins=10,  color='lightblue', edgecolor='black', linewidth=2)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title(col_name,fontsize=16)
                
            fig.savefig(( eda_plots_path +  f'{features_key}_eda_histplots.png'), dpi=600)   
            plt.show()

        else:
            print('Please, provide or check if the feature key is in the df columns')  
    
    















