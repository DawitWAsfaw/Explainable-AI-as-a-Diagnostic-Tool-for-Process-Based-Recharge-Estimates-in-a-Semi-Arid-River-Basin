import numpy as np
# import polars as pl
import  matplotlib.pyplot as plt
import seaborn as sns

def eda_hist_plot_of_open_source_variables(ml_input_all_df, eda_plots_path):
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
    renaming_dic = { 
                  'dTs':'Distance to stream', 
                  'drainage_density': 'Drainage Density',  
                  'TRI': 'Topographic Roughness Index',  
                  'TWI':'Topographic Wetness Index',  
                  'SPI':'Stream Power Index',  
                  'slope': 'Slope',  
                  'awsPcnt': 'Available Water Storage',  
                  'clayPcnt':'Percent Clay', 
                  'sandPcnt':'Percent Sand',  
                  'siltPcnt':'Percent Silt',  
                  'awcPcnt':'Available Water Capacity', 
                  'ksatPcnt':'Saturated Hydraulic Conductivity (Ksat)',
                  'FloodFCls': 'Flooding Frequency Class',  
                  'PondFCls': 'Ponding Frequency Class', 
                  'ET':'Actual Evapotranspiration',  
                  'Precip':'Precipitation',  
                  'Tmax':'Maximum Air Temperature',  
                  'Tmin':'Minimum Air Temperature ',  
                  'developedlowIntensity':'Developed Low Intensity', 
                  'developedMediumIntensity':'Developed Medium Intensity',  
                  'evergreenForest':'Evergreen Forest',  
                  'grasslandOrHerbaceous':'Grassland Or Herbaceous', 
                  'cultivatedCrops':'Cultivated Crops',  
                  'woodyWetlands':'Woody Wet Lands',  
                  'emergentHerbaceousWetlands': 'Emergent Herbaceous Wet lands', 
                  'k_mday':'Hydraulic Conductivity',  
                  'sy':'Specific Yield',  
                  'depth2Water_tbl':'Depth to Water Table',  
                  'thick_m' : 'Aquifer Thickness',
                  'recharge [mm/yr]':'Recharge [mm/yr]'}

    ml_input_all_df = ml_input_all_df.rename(columns=renaming_dic)
        
    
    ml_input_var_group_dict = {
        'climate': ['Actual Evapotranspiration', 'Precipitation',  'Maximum Air Temperature', 'Minimum Air Temperature '],
        
        'soil_phy_prop': ['Available Water Storage', 'Percent Clay','Saturated Hydraulic Conductivity (Ksat)', 
        'Percent Sand', 'Percent Silt', 'Available Water Capacity','Flooding Frequency Class','Ponding Frequency Class'],
        
        'hydro_factors': ['Distance to stream', 'Drainage Density', 'Topographic Roughness Index', 'Topographic Wetness Index', 
                          'Stream Power Index', 'Slope'],
        
        'lulc_classes':['Evergreen Forest', 'Grassland Or Herbaceous','Cultivated Crops','Emergent Herbaceous Wet lands'],
        
        'hydrogeol_var':['Hydraulic Conductivity',  'Specific Yield', 'Depth to Water Table',  'Aquifer Thickness']
        }
    
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    for features_key in ml_input_var_group_dict.keys():
        print(features_key)
        df = ml_input_all_df[ml_input_var_group_dict[features_key]]
        if features_key == 'climate'or features_key == 'lulc_classes' or features_key == 'hydrogeol_var':
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)
            for idx, col_name  in enumerate(ml_input_var_group_dict[features_key]):
                ax = axes.ravel()[idx]
                df[col_name]=  df[col_name].round(3)
                ax.hist( df[col_name], bins=10,  color='lightblue', edgecolor='black', linewidth=2)
                ax.set_ylabel('Count', fontsize=14)
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
            print('Please, provide chekc if the feature key in the df columns')
            
            

def create_corr_heatmap_of__open_source_predictor_variables(ml_input_all_df,eda_plots_path):
    """
    This function creates correlation heat for predictor variables
    Parameters
    ----------
    ml_input_all_df : Dataframe contains observation values of predictor variables
    eda_plots_path : Directory path where the plots are saved. Datatype: String
    Returns
    -------
        None.

    """
 
    corr_col_rename ={
                    'Distance to stream':1, 'Drainage Density':2, 'Topographic Roughness Index': 3, 
                    'Topographic Wetness Index':4,'Stream Power Index':5, 'Slope':6,
                    
                    'Available Water Storage':7, 'Percent Clay':8, 'Saturated Hydraulic Conductivity (Ksat)':9,
                    'Percent Sand':10, 'Percent Silt':11, 'Available Water Capacity':12, 
                    'Flooding Frequency Class':13,'Ponding Frequency Class':14,
                    
                    'Actual Evapotranspiration':15, 'Precipitation':16,  
                    'Maximum Air Temperature':17,  'Minimum Air Temperature ':18,
                    
                     'Developed Low Intensity':19, 'Developed Medium Intensity':20, 'Evergreen Forest':21,
                     'Grassland Or Herbaceous':22,'Cultivated Crops':23,'Emergent Herbaceous Wet lands':24,
              
                     'Hydraulic Conductivity':25,  'Specific Yield':26, 'Depth to Water Table':27,  
                     'Aquifer Thickness':28,  'Recharge [mm/yr]':29
                     }
    text_label_list = []
    
    for idx, col_key in zip(range(1,30), corr_col_rename.keys()):
        label = f'{idx} - {col_key}'
        
        text_label_list.append(label)
        
    renaming_dic = { 
                  'dTs':'Distance to stream', 
                  'drainage_density': 'Drainage Density',  
                  'TRI': 'Topographic Roughness Index',  
                  'TWI':'Topographic Wetness Index',  
                  'SPI':'Stream Power Index',  
                  'slope': 'Slope',  
                  'awsPcnt': 'Available Water Storage',  
                  'clayPcnt':'Percent Clay', 
                  'sandPcnt':'Percent Sand',  
                  'siltPcnt':'Percent Silt',  
                  'awcPcnt':'Available Water Capacity', 
                  'ksatPcnt':'Saturated Hydraulic Conductivity (Ksat)',
                  'FloodFCls': 'Flooding Frequency Class',  
                  'PondFCls': 'Ponding Frequency Class', 
                  'ET':'Actual Evapotranspiration',  
                  'Precip':'Precipitation',  
                  'Tmax':'Maximum Air Temperature',  
                  'Tmin':'Minimum Air Temperature ',  
                  'developedlowIntensity':'Developed Low Intensity', 
                  'developedMediumIntensity':'Developed Medium Intensity',  
                  'evergreenForest':'Evergreen Forest',  
                  'grasslandOrHerbaceous':'Grassland Or Herbaceous', 
                  'cultivatedCrops':'Cultivated Crops',  
                  'woodyWetlands':'Woody Wet Lands',  
                  'emergentHerbaceousWetlands': 'Emergent Herbaceous Wet lands', 
                  'k_mday':'Hydraulic Conductivity',  
                  'sy':'Specific Yield',  
                  'depth2Water_tbl':'Depth to Water Table',  
                  'thick_m' : 'Aquifer Thickness',
                  'recharge [mm/yr]':'Recharge [mm/yr]'}

    ml_input_all_df = ml_input_all_df.rename(columns=renaming_dic)
        
    variable_names = ['Distance to stream', 'Drainage Density', 'Topographic Roughness Index', 'Topographic Wetness Index', 
                      'Stream Power Index', 'Slope','Available Water Storage', 'Percent Clay','Saturated Hydraulic Conductivity (Ksat)', 
                      'Percent Sand', 'Percent Silt', 'Available Water Capacity','Flooding Frequency Class',  
                      'Ponding Frequency Class',  'Actual Evapotranspiration', 'Precipitation',  'Maximum Air Temperature', 
                      'Minimum Air Temperature ','Developed Low Intensity', 'Developed Medium Intensity', 'Evergreen Forest',
                      'Grassland Or Herbaceous','Cultivated Crops','Emergent Herbaceous Wet lands',
                      'Hydraulic Conductivity',  'Specific Yield', 'Depth to Water Table',  'Aquifer Thickness',  'Recharge [mm/yr]']

    ml_input_all_df_corr = ml_input_all_df[variable_names]
    print(ml_input_all_df_corr.columns)
    ml_input_all_corr_df = ml_input_all_df_corr.rename(columns = corr_col_rename)
    print(ml_input_all_corr_df.columns)
    
    mask = np.triu(np.ones_like(ml_input_all_corr_df.corr(), dtype=bool))
    corr_values =ml_input_all_corr_df.corr().astype('float64')
    corr_values= (corr_values.fillna(0)).round(2)
        
    
    
    figure,ax = plt.subplots(1, 1, figsize=(20, 20),layout="constrained")
    ax = sns.heatmap(corr_values, vmin= -1, square=True ,fmt=".2g",mask=mask,vmax= 1, linecolor='k', annot_kws={"size": 14},
                     cmap="YlGnBu",annot=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for idx, text in zip(np.arange(3,20.5,0.5),text_label_list):
        print(idx, text )
        plt.text(19, idx, text, fontsize=18,)
    plt.text(22.5, 2.5, 'Axis label Keys', fontsize=22, fontweight='bold') 
    plt.grid(False)
    cbar = ax.collections[0].colorbar


    cbar.ax.tick_params(labelsize=16)
    ax.set_title("Pearson correlation heat map of open source ML model input variables",fontdict = {'fontsize': 22,'fontweight': 'bold'})
    plt.tight_layout()
    plt.savefig((eda_plots_path +  'OpenSource_data_correlation_heatmap.png'), dpi=400)
    # plt.close(figure)             

