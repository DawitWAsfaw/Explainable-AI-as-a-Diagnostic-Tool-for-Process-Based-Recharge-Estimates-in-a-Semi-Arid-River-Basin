from glob import glob
import os

import numpy as np
import pandas as pd
import time

import  matplotlib.pyplot as plt
import seaborn as sns

# import joblib
from sklearn.ensemble import RandomForestRegressor
import rasterio as rio
from rasterstats import zonal_stats
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance


def plot_permutationImportance(rf, x_train_data, y_train_data, ml_plots):
    """
    Create feature importance plot (FI)
    :classifier_model: classifier_model used for prediction
    :x_train: predictor training data set used to build random forest model
    : plots_dir : directory path to store  plots 
    : return: none
    """
    result = permutation_importance(rf,x_train_data, y_train_data, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()
    
    mdi_importances = pd.Series(rf.feature_importances_, index=x_train_data.columns)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))
    mdi_importances.sort_values().plot.barh(ax=ax1)
    ax1.set_xlabel("Gini importance")
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=x_train_data.columns[perm_sorted_idx],
    )
    ax2.axvline(x=0, color="k", linestyle="--")
    ax2.set_xlabel("Decrease in accuracy score")
    fig.suptitle(
        "Impurity-based vs. permutation importance on multicollinear features"
    )
    _ = fig.tight_layout()
    plt.savefig(( ml_plots +  'perm_importance.png'), dpi=600)
    print('Permutation importancee plot saved')
    # plt.close( fig)
    

def plot_featureImportance(rf,  x_train_data, ml_plots):
    """
    Create feature importance plot (FI)
    : ranFor_model_optimized: random forest model used for prediction
    : train_x: predictor training data set used to build random forest model
    : plots_dir : directory path to store  plots 
    : return: none
    """

    labels_x_axis= np.array(x_train_data.columns)
    importance = np.array(rf.feature_importances_)
    imp_dict = {'feature_names': labels_x_axis, 'Feature_importance': importance}
    imp_df = pd.DataFrame(imp_dict)
   
    imp_df.sort_values(by=['Feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(16, 14))
    plt.rcParams['font.size'] = 30
    sns.barplot(x=imp_df['Feature_importance'], y=imp_df['feature_names'], color ='gray')
    # plt.xticks(rotation=15)
    plt.ylabel('Variables')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig((ml_plots +  'pred_importance.png'), dpi=600)
    print('Feature importance plot saved')

def create_pd_for_different_predictor_variables_with_condition(ml_input_all_df,ml_plots , numOfsample_data =50000):
    """
    This functions plots partial depedence plots for Precipitatin, ET, Depth2water table, and distance to streams using different values of lulc classes. 
    ----------
    ml_input_all_df : Dataframe with all the predictor variables and target variable
    ml_plots  : Directory path to save partial dependence plots. 
    numOfsample_data: The number of rows used for fitting a RF model. The default is 50,000. 
    var_namees : List of lulc classes. The default is 'evergreenForest', 'grasslandOrHerbaceous', 'cultivatedCrops' and ,'emergentHerbaceousWetlands'.
    Returns
    -------
    None.

    """
     
    # var_names = ['developedlowIntensity','evergreenForest', 'grasslandOrHerbaceous','cultivatedCrops',
    #              'emergentHerbaceousWetlands','slope','Precip'] 
    
    var_names = ['slope'] 
    for var_name in var_names:
        if var_name == 'developedlowIntensity':
            print('Plotting PD for: ',var_name)
            condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 0.1].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[ml_input_all_df[var_name]> 0.1]]
                
            x_train_df_list = []
            rf_list = []
            for conditional_df in condition_based_df_list:
                
                print(conditional_df.shape)
                x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','aws','silt','recharge [m3/day]','recharge [mm/yr]'], axis=1)

                y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
                
                X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
                X_train_data = X_train.drop(columns= ['gridid', 'year'])
                y_train_data = y_train['recharge [mm/yr]']
                
                x_train_df_list.append(X_train_data)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                start_time = time.time()
        
                rf.fit(X_train_data, y_train_data)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Fitting the RandomForestRegressor model took: ', elapsed_time)
                rf_list.append(rf)
                
            print(len(x_train_df_list))
            print(len(rf_list)) 

            all_variables =  { 'group_1': ['evergreenForest','developedlowIntensity','cultivatedCrops','emergentHerbaceousWetlands'],
                               'group_2': ['TWI','SPI','TRI','Tmin'],
                               'group_3': ['k_mday','sy','depth2Water_tbl','thick_m'],
                               'group_4': ['awc', 'clay','sand','ksat'],
                               'group_5': ['pondfcl','floodfcl','dTs','drainage_density'],
                               'group_6': ['ET', 'Precip','slope','Tmax']
                               
                               }
            
         
            
            colors   = ["blue", "green"]
            conditions = ["low", "high"]
            labels   = [ f"{var_name} ≤ 10%",
                        f"{var_name}  > 10%" ]
            
            ylim = (0, 30)
            tick_kw = dict(length=6, width=2)
            ylabel = (' ')

            for group_name in all_variables.keys():
                fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                fig.supylabel("Recharge [mm/year]", fontsize=16)
                axes = axes.ravel()
                for ax, feature in zip(axes, all_variables[group_name]):
                    print(feature)
                    display = None
                    for con, i in zip(conditions, range(len(labels))): 
                        display = PartialDependenceDisplay.from_estimator(
                            rf_list[i],
                            x_train_df_list[i],
                            [feature],            
                            ax=display.axes_ if display is not None else ax,
                            line_kw={"label": labels[i], "color": colors[i]},
                        )
                        plt.setp(display.deciles_vlines_, visible=False)
                        a = display.axes_[0][0]
                        a.tick_params(**tick_kw)
                        a.set_ylim(*ylim)
                        a.set_ylabel(*ylabel)

                fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                plt.show()
                
                
            
        elif var_name == 'evergreenForest' or var_name == 'grasslandOrHerbaceous'or var_name == 'cultivatedCrops':
            print('Plotting PD for: ',var_name)
            condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 0.1].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>0.1) & (ml_input_all_df[var_name]<=0.5)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>0.5) & (ml_input_all_df[var_name]<=0.75)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[ml_input_all_df[var_name]> 0.75].sample(n=numOfsample_data, random_state=42)]
            
            x_train_df_list = []
            rf_list = []
            for conditional_df in condition_based_df_list:
                
                print(conditional_df.shape)
                x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','aws','silt','recharge [m3/day]','recharge [mm/yr]'], axis=1)

                y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
                
                X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
                X_train_data = X_train.drop(columns= ['gridid', 'year'])
                y_train_data = y_train['recharge [mm/yr]']
                
                x_train_df_list.append(X_train_data)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                start_time = time.time()
        
                rf.fit(X_train_data, y_train_data)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Fitting the RandomForestRegressor model took: ', elapsed_time)
                rf_list.append(rf)
                
            print(len(x_train_df_list))
            print(len(rf_list)) 

            all_variables =  { 'group_1': ['TWI','SPI','TRI','Tmin'],
                               'group_2': ['k_mday','sy','depth2Water_tbl','thick_m'],
                               'group_3': ['awc', 'clay','sand','ksat'],
                               'group_4': ['pondfcl','floodfcl','dTs','drainage_density'],
                               'group_5': ['ET', 'Precip','slope','Tmax']
                               
                               }
            
         
            conditions = ["low", "medium", "midium_high","high"]
            
            colors   = ["blue", "green", "purple","red"]
            
            labels   = [ f"{var_name} ≤ 10%",
                         f"{var_name} > 10% and ≤ 50%",
                         f"{var_name}  > 50% and ≤ 75%",
                         f"{var_name}  > 75%" ]
            
            if var_name == 'cultivatedCrops':
                ylim = (0, 30)
                tick_kw = dict(length=6, width=2)
                ylabel = (' ')
                for group_name in all_variables.keys():
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                    fig.supylabel("Recharge [mm/year]", fontsize=16)
                    axes = axes.ravel()
                    for ax, feature in zip(axes, all_variables[group_name]):
                        print(feature)
                        display = None
                        for con, i in zip(conditions, range(len(labels))): 
                            display = PartialDependenceDisplay.from_estimator(
                                rf_list[i],
                                x_train_df_list[i],
                                [feature],            
                                ax=display.axes_ if display is not None else ax,
                                line_kw={"label": labels[i], "color": colors[i]},
                            )
                            plt.setp(display.deciles_vlines_, visible=False)
                            a = display.axes_[0][0]
                            a.tick_params(**tick_kw)
                            a.set_ylim(*ylim)
                            a.set_ylabel(*ylabel)

                        ax.legend()

                    fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                    plt.show()
                print('Plotted PD for: ',var_name)

            elif var_name == 'grasslandOrHerbaceous':
                print(var_name)
                ylim = (0, 80)
                tick_kw = dict(length=6, width=2)
                ylabel = (' ')
                
                
                
                for group_name in all_variables.keys():
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                    fig.supylabel("Recharge [mm/year]", fontsize=16)
                    axes = axes.ravel()
                    for ax, feature in zip(axes, all_variables[group_name]):
                        print(feature)
                        display = None
                        for i in range(len(labels)): 
                            display = PartialDependenceDisplay.from_estimator(
                                rf_list[i],
                                x_train_df_list[i],
                                [feature],            
                                ax=display.axes_ if display is not None else ax,
                                line_kw={"label": labels[i], "color": colors[i]},
                            )
                            plt.setp(display.deciles_vlines_, visible=False)
                            a = display.axes_[0][0]
                            a.tick_params(**tick_kw)
                            a.set_ylim(*ylim)
                            a.set_ylabel(*ylabel)
                
                    fig.savefig(ml_plots  + f'{con}_PDP_{var_name}.jpg',dpi=300)
                    plt.show()
                    
            else:
                
                ylim = (0, 110)
                tick_kw = dict(length=6, width=2)
                ylabel = (' ')
                
                for group_name in all_variables.keys():
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                    fig.supylabel("Recharge [mm/year]", fontsize=16)
                    axes = axes.ravel()
                    for ax, feature in zip(axes, all_variables[group_name]):
                        print(feature)
                        display = None
                        for i in range(len(labels)): 
                            display = PartialDependenceDisplay.from_estimator(
                                rf_list[i],
                                x_train_df_list[i],
                                [feature],            
                                ax=display.axes_ if display is not None else ax,
                                line_kw={"label": labels[i], "color": colors[i]},
                            )
                            plt.setp(display.deciles_vlines_, visible=False)
                            a = display.axes_[0][0]
                            a.tick_params(**tick_kw)
                            a.set_ylim(*ylim)
                            a.set_ylabel(*ylabel)
                
                    fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                    plt.show()
                print('Plotted PD for: ',var_name)
        
        elif var_name == 'emergentHerbaceousWetlands':
            print('Plotting PD for: ', var_name)
            
            condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 0.1].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>0.1) & (ml_input_all_df[var_name]<=0.5)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>0.5) & (ml_input_all_df[var_name]<=0.75)],
                              ml_input_all_df[ml_input_all_df[var_name]> 0.75]]

            x_train_df_list = []
            rf_list = []
            for conditional_df in condition_based_df_list:
                
                print(conditional_df.shape)
                x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','aws','silt','recharge [m3/day]','recharge [mm/yr]'], axis=1)

                y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
                
                X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
                X_train_data = X_train.drop(columns= ['gridid', 'year'])
                y_train_data = y_train['recharge [mm/yr]']
                
                x_train_df_list.append(X_train_data)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                
                start_time = time.time()

                rf.fit(X_train_data, y_train_data)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Fitting the RandomForestRegressor model took: ', elapsed_time)
                rf_list.append(rf)
                
            print(len(x_train_df_list))
            print(len(rf_list))
            
            all_variables =  { 'group_1': ['cultivatedCrops','evergreenForest','developedlowIntensity','emergentHerbaceousWetlands'],
                               'group_2': ['TWI','SPI','TRI','Tmin'],
                               'group_3': ['k_mday','sy','depth2Water_tbl','thick_m'],
                               'group_4': ['awc', 'clay','sand','ksat'],
                               'group_5': ['pondfcl','floodfcl','dTs','drainage_density'],
                               'group_6': ['ET', 'Precip','slope','Tmax']
                               
                               }
        
            
            colors   = ["blue", "green", "purple","red"]
            conditions = ["low", "medium", "midium_high","high"]
            
            labels   = [f"{var_name} ≤ 10%",
                        f"{var_name} > 10% and ≤ 50%",
                        f"{var_name}  > 50% and ≤ 75%",
                        f"{var_name}  > 75%" ]
            
            ylim = (0, 60)
            tick_kw = dict(length=6, width=2)
            ylabel = (' ')


            
            for group_name in all_variables.keys():
                fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                fig.supylabel("Recharge [mm/year]", fontsize=16)
                axes = axes.ravel()
                for ax, feature in zip(axes, all_variables[group_name]):
                    print(feature)
                    display = None
                    for con, i in zip(conditions, range(len(labels))): 
                        display = PartialDependenceDisplay.from_estimator(
                            rf_list[i],
                            x_train_df_list[i],
                            [feature],            
                            ax=display.axes_ if display is not None else ax,
                            line_kw={"label": labels[i], "color": colors[i]},
                        )
                        plt.setp(display.deciles_vlines_, visible=False)
                        a = display.axes_[0][0]
                        a.tick_params(**tick_kw)
                        a.set_ylim(*ylim)
                        a.set_ylabel(*ylabel)

                fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                plt.show()

            print('Plotted PD for: ',var_name)
            
        elif var_name ==  'slope':
            print('Plotting PD for: ', var_name)
            condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 5].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>5) & (ml_input_all_df[var_name]<=10)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>10) & (ml_input_all_df[var_name]<=15)],
                              ml_input_all_df[ml_input_all_df[var_name]> 15]]

            x_train_df_list = []
            rf_list = []
            for conditional_df in condition_based_df_list:
                
                print(conditional_df.shape)
                x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','aws','silt','recharge [m3/day]','recharge [mm/yr]'], axis=1)

                y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
                
                X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
                X_train_data = X_train.drop(columns= ['gridid', 'year'])
                y_train_data = y_train['recharge [mm/yr]']
                
                x_train_df_list.append(X_train_data)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                start_time = time.time()

                rf.fit(X_train_data, y_train_data)
                rf_list.append(rf)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Fitting the RandomForestRegressor model took: ', elapsed_time)
                
                
            print(len(x_train_df_list))
            print(len(rf_list))
            
            all_variables =  { 'group_1': ['evergreenForest','developedlowIntensity','cultivatedCrops','clay'],
                               'group_2': ['TWI','SPI','TRI','slope'],
                               'group_3': ['k_mday','sy','depth2Water_tbl','thick_m'],
                               'group_4': ['dTs','drainage_density','sand','ksat'],
                               'group_5': ['ET', 'Precip','slope','Tmax','Tmin']
                               
                               }
            
         
            conditions = ["low", "medium", "midium_high","high"]
            colors   = ["blue", "green", "purple","red"]
            labels   = [
                "Slope ≤ 5°",
                "Slope > 5° and ≤ 10°",
                "Slope > 10° and ≤ 15°",
                "Slope > 15°"]
            
            ylim = (0, 150)
            tick_kw = dict(length=6, width=2)
            ylabel = (' ')
            
            
            
            for group_name in all_variables.keys():
                fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                fig.supylabel("Recharge [mm/year]", fontsize=16)
                axes = axes.ravel()
                for ax, feature in zip(axes, all_variables[group_name]):
                    print(feature)
                    display = None
                    for i in range(len(labels)): 
                        display = PartialDependenceDisplay.from_estimator(
                            rf_list[i],
                            x_train_df_list[i],
                            [feature],            
                            ax=display.axes_ if display is not None else ax,
                            line_kw={"label": labels[i], "color": colors[i]},
                        )
                        plt.setp(display.deciles_vlines_, visible=False)
                        a = display.axes_[0][0]
                        a.tick_params(**tick_kw)
                        a.set_ylim(*ylim)
                        a.set_ylabel(*ylabel)
                        
                fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                plt.show()
                        
        elif var_name == 'Precip':
            print('Plotting PD for: ',var_name)
            condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 200].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>200) & (ml_input_all_df[var_name]<=600)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[(ml_input_all_df[var_name]>600) & (ml_input_all_df[var_name]<=800)].sample(n=numOfsample_data, random_state=42),
                              ml_input_all_df[ml_input_all_df[var_name]> 800]]

            x_train_df_list = []
            rf_list = []
            for conditional_df in condition_based_df_list:
                
                print(conditional_df.shape)
                x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','aws','silt','recharge [m3/day]','recharge [mm/yr]'], axis=1)

                y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
                
                X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
                X_train_data = X_train.drop(columns= ['gridid', 'year'])
                y_train_data = y_train['recharge [mm/yr]']
                
                x_train_df_list.append(X_train_data)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                start_time = time.time()

                rf.fit(X_train_data, y_train_data)
                rf_list.append(rf)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Fitting the RandomForestRegressor model took: ', elapsed_time)
                
                
            print(len(x_train_df_list))
            print(len(rf_list))
            all_variables =  { 'group_1': ['evergreenForest','developedlowIntensity','TWI','slope'],
                               'group_2': ['depth2Water_tbl','thick_m','dTs','drainage_density'],
                               'group_3': ['ET', 'Precip','Tmin','Tmax']
                               
                               }
            
         
            conditions = ["low", "medium", "midium_high","high"]
            colors   = ["blue", "green", "purple","red"]
            labels   = [
                "Precipitation  ≤ 200 mm",
                "Precipitation  > 200 mm and ≤ 600 mm",
                "Precipitation  > 600 mm and ≤ 800 mm",
                "Precipitation  > 800 mm",
            ]
            ylim = (0, 180)
            tick_kw = dict(length=6, width=2)
            ylabel = (' ')


            
            for group_name in all_variables.keys():
                fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
                fig.supylabel("Recharge [mm/year]", fontsize=16)
                axes = axes.ravel()
                for ax, feature in zip(axes, all_variables[group_name]):
                    print(feature)
                    display = None
                    for con, i in zip(conditions, range(len(labels))): 
                        display = PartialDependenceDisplay.from_estimator(
                            rf_list[i],
                            x_train_df_list[i],
                            [feature],            
                            ax=display.axes_ if display is not None else ax,
                            line_kw={"label": labels[i], "color": colors[i]},
                        )
                        plt.setp(display.deciles_vlines_, visible=False)
                        a = display.axes_[0][0]
                        a.tick_params(**tick_kw)
                        a.set_ylim(*ylim)
                        a.set_ylabel(*ylabel)
                        
                fig.savefig(ml_plots  + f'{group_name}_PDP_{var_name}.jpg',dpi=300)
                plt.show()
            
        else:
            break
                
     
def create_pd_for_different_slope_ranges(ml_input_all_df,ml_plots , numOfsample_data =50000):
    """
    This functions plots partial depedence plots for Precipitatin, ET, Depth2water table, and distance to streams using different values of slope
    Parameters
    ----------
    main_ml_input_df : Dataframe with all the predictor variables and target variable
   ml_plots  : Directory path to save partial dependence plots. 
    numOfsample_data: The number of rows used for fitting a RF model. The default is 50,000. 
    var_name : The default is 'slope'.
    Returns
    -------
    None.

    """
    var_name = 'slope'
    
    condition_based_df_list = [ml_input_all_df[ml_input_all_df[var_name]<= 5].sample(n=numOfsample_data, random_state=42),
                      ml_input_all_df[(ml_input_all_df[var_name]>5) & (ml_input_all_df[var_name]<=10)].sample(n=numOfsample_data, random_state=42),
                      ml_input_all_df[(ml_input_all_df[var_name]>10) & (ml_input_all_df[var_name]<=15)].sample(n=numOfsample_data, random_state=42),
                      ml_input_all_df[ml_input_all_df[var_name]> 15].sample(n=numOfsample_data, random_state=42)]

    x_train_df_list = []
    rf_list = []
    for conditional_df in condition_based_df_list:
        
        print(conditional_df.shape)
        x_data = conditional_df.drop(['area_sqrtm','mTomm','dayInayr','recharge [m3/day]','recharge [mm/yr]'], axis=1)

        y_data = conditional_df[['gridid', 'year', 'recharge [mm/yr]']]
        
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
        X_train_data = X_train.drop(columns= ['gridid', 'year'])
        y_train_data = y_train['recharge [mm/yr]']
        
        x_train_df_list.append(X_train_data)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        start_time = time.time()

        rf.fit(X_train_data, y_train_data)
        rf_list.append(rf)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Fitting the RandomForestRegressor model took: ', elapsed_time)
        
        
    print(len(x_train_df_list))
    print(len(rf_list))

    features = ["Precip", "ET", "depth2Water_tbl", "dTs"] 
    colors   = ["blue", "red", "green", "purple"]
    labels   = [
        "Slope ≤ 5°",
        "Slope > 5° and ≤ 10°",
        "Slope > 10° and ≤ 15°",
        "Slope > 15°",
    ]
    ylim = (0, 150)
    tick_kw = dict(length=6, width=2)
    ylabel = (' ')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
    fig.supylabel("Recharge [mm/year]", fontsize=16)

    axes = axes.ravel()

    for ax, feature in zip(axes, features):
        display = None
        for i in range(len(labels)): 
            display = PartialDependenceDisplay.from_estimator(
                rf_list[i],
                x_train_df_list[i],
                [feature],            
                ax=display.axes_ if display is not None else ax,
                line_kw={"label": labels[i], "color": colors[i]},
            )
            plt.setp(display.deciles_vlines_, visible=False)
            a = display.axes_[0][0]
            a.tick_params(**tick_kw)
            a.set_ylim(*ylim)
            a.set_ylabel(*ylabel)

        ax.legend()

    fig.savefig(ml_plots  + f'Features_PDP_{var_name}.jpg',dpi=300)
    plt.show()
    
def burn_raster_from_shapefile(shp,rast,fname,colname,dtype='float32',nodata=0):
    meta = rast.meta
    meta['dtype'] = dtype
    meta['nodata'] = nodata
    
    with rio.open(fname, 'w+', **meta) as out:
        out_arr = out.read(1)    
        shapes = ((geom,value) for geom, value in zip(shp.geometry, shp[colname]))
        
        burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write(burned,1)
        
    return(burned)

def create_500m_pixel_prediction_rasters(rf, x_train_data, x_test_data, ref_raster_file, grid_500m_shapefile, spatial_500m_plots_dir):
    """
    This function creates spatial prediction raster data
    Parameters
    ----------
    rf : Fitted model
    x_train_data : Training dataframe containing predictor variables to fit a model
    x_test_data : Testing dataframe containing predictor variables to test a model
    ref_raster:  Reference raster file with 500m spatial resolution with study area extent; Type- String
    grid_500m_shapefile:  500m grid shapefile; Type- String
    spatial_500m_plots_dir : Directory path to save spatial prediction raster plots
    Returns
    -------
    None.

    """
    grid_shapfile = gpd.read_file(grid_500m_shapefile)
    ref_raster = rio.open(ref_raster_file)
    train_prediction = rf.predict(x_train_data)
    test_prediction = rf.predict(x_test_data)
    
    train_prediction_df = x_train_data[['gridid', 'year']]
    train_prediction_df['Train Predicted_recharge [mm/yr]'] = train_prediction 
    test_prediction_df = x_test_data[['gridid', 'year']]
    test_prediction_df['Test Predicted_recharge [mm/yr]'] = test_prediction
    
    recharge_train_pred_df = train_prediction_df.rename(columns ={'Train Predicted_recharge [mm/yr]':'Predicted_recharge [mm/yr]' })
    recharge_test_pred_df = test_prediction_df.rename(columns ={'Test Predicted_recharge [mm/yr]':'Predicted_recharge [mm/yr]' })

    predicted_recharge_df = pd.concat([recharge_train_pred_df , recharge_test_pred_df])
    for year in predicted_recharge_df ['year'].unique():
        print(year)
        predicted_df = predicted_recharge_df [predicted_recharge_df ['year']==year]
        merged_gdf = grid_shapfile.merge(predicted_df, on='gridid')
        fname = f'{spatial_500m_plots_dir}predicted_recharge_{year}.tif'
        shp =  merged_gdf
        rast = ref_raster
        colname = 'Predicted_recharge [mm/yr]'
        burn_raster_from_shapefile(shp,rast,fname,colname,dtype='float32',nodata=-9999)
               
def calculate_mean_recharge_huc12_scale(swat_recharge_dir, ml_prediction_recharge_dir, huc12_polygon_shpfile_path, huc12_csv_dir):
    """
    This function calculated mean recharge value at HUC12 polygon scale

    Parameters
    ----------
    swat_recharge_dir :  Directory path for 500m raster SWAT+ recharges
    ml_prediction_recharge_dir : Directory path for 500m raster ml recharge predictions
    huc12_polygon_shpfile_path :  Directory path for HUC12 polygon shapefile
    huc12_csv_dir :  Directory path to save mean values as csv file

    Returns
    -------
    None.

    """
    huc12_polygon = gpd.read_file(huc12_polygon_shpfile_path)
    
    for file in glob(swat_recharge_dir + '*.tif'):
        variable = os.path.basename(file).replace('.tif', '')
        year = variable[-4:]
        print(f'Processing: {variable}')

        with rio.open(file) as raster_data:
            huc12_polygon = huc12_polygon.to_crs(raster_data.crs)

        stats = zonal_stats(
            huc12_polygon,
            file,
            stats=['mean'],
            geojson_out=False
        )

        statistics = pd.DataFrame(stats, index=huc12_polygon['huc12'])
        statistics.insert(loc =1,column ='year',value = year)
        statistics.to_csv(os.path.join(huc12_csv_dir, f'{variable}.csv'))
    
    for file in glob(ml_prediction_recharge_dir + '*.tif'):
        variable = os.path.basename(file).replace('.tif', '')
        year = variable[-4:]
        print(f'Processing: {variable}')

        with rio.open(file) as raster_data:
            huc12_polygon = huc12_polygon.to_crs(raster_data.crs)

        stats = zonal_stats(
            huc12_polygon,
            file,
            stats=['mean'],
            geojson_out=False
        )

        statistics = pd.DataFrame(stats, index=huc12_polygon['huc12'])
        statistics.insert(loc =1,column ='year',value = year)
        statistics.to_csv(os.path.join(huc12_csv_dir, f'{variable}.csv'))
        
        
def create_huc12_scale_rasters(huc12_csv_dir, ref_raster_file, huc12_shapefile, huc12_tif_dir):
    """
    This function creates spatialraster data
    Parameters
    ----------
    huc12_csv_dir: csv directory path for SWAT+ estimates and ML predictions at huc12
    ref_raster:  Reference raster file with 500m spatial resolution with study area extent; Type- String
    huc12_shapefile:  HUC12 polygon shapefile; Type- string
    huc12_tif_dir :Directory path to save spatial prediction raster plots
    Returns
    -------
    None.

    """
    huc12_shapfile = gpd.read_file(huc12_shapefile)
    # print(huc12_shapfile['huc12'].dtype)
    huc12_shapfile['huc12'] = pd.to_numeric(huc12_shapfile['huc12'], errors="coerce").astype("Int64")
    # print(huc12_shapfile['huc12'].dtype)
    ref_raster = rio.open(ref_raster_file)
    for csv_file in glob(huc12_csv_dir + '*.csv'):
        huc12_df = pd.read_csv(csv_file )
        # print(huc12_df['huc12'].dtype)
        variable_tif =os.path.basename(csv_file).split('/')[-1]
        variable = variable_tif[variable_tif.rfind(os.sep) + 1: variable_tif.rfind('.')]
        print(variable)
        
        merged_gdf = huc12_shapfile.merge(huc12_df, on='huc12')
        fname = f'{huc12_tif_dir}{variable}.tif'
        shp =  merged_gdf
        rast = ref_raster
        colname = 'mean'
        burn_raster_from_shapefile(shp,rast,fname,colname,dtype='float32',nodata=-9999)        
    
def ml_spatial_prediction_plot(swat_recharge_dir, pred_recharge_dir, precip_dir, swat_pred_csv_dir, ml_plots):
    """
    This function creates SWAT+ simulated vs ML predicted spatial distribution plots.
    Parameters
    ----------
    swat_recharge_dir : Director path for SWAT+ simulated recharge
    pred_recharge_dir : Directory path for ML predicted recharge
    precip_dir : Directory path for precipitation data
    swat_pred_csv_dir : directory path for SWAT+ simulated and ML predicted csv file
    ml_plots : Directory path where the plot will be saved. 
    Returns
    -------
    None.
    """
    
    
    
    
# swat_recharge_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/deep_percolation_data/mosaic_tiff/'
# ml_prediction_recharge_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/plots/predictions/'
# huc12_polygon_shpfile_path = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/huc12/huc12_subbasins_correct_15N.shp'
# huc12_csv_dir =  'D:/deep_percolotation_ml_project/ml_analysis_results/predictions/csv/huc12/'    
# ref_raster_file = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/ref_raster/larb_ref_raster.tif'
# huc12_shapefile = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/huc12/huc12_subbasins_correct_15N.shp'
# huc12_tif_dir = 'D:/deep_percolotation_ml_project/ml_analysis_results/predictions/spatial/huc12/'


# calculate_mean_recharge_huc12_scale(swat_recharge_dir, ml_prediction_recharge_dir, huc12_polygon_shpfile_path, huc12_csv_dir)    
# create_huc12_scale_rasters(huc12_csv_dir, ref_raster_file, huc12_shapefile, huc12_tif_dir)   
    
    
    
    
    
    
    
    
    
    
    
    