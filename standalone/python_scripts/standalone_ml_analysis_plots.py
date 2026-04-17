
import copy
import matplotlib.pyplot as plt
import shap
import skexplain
import matplotlib as mpl
import matplotlib.colors as mcolors 
import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio as rio
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar

def SWAT_standAlone_predicted_plots(swat_ml_predicted_df, ml_plots):
    '''
    Parameters
    ----------
    swat_ml_predicted_df :  DataFrame containing predicted, simulated recharge , precipitation and residuals (swat- ml_predict)
    plots_path :  Director path for where the shap and ale plots are saved
    Returns
    -------
    None.

    '''
    plt.rcParams["font.weight"] = 'normal'
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['font.family'] = ['arial']
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['lines.color'] = 'k'

    plt.close('all')
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 10),tight_layout= False)


    ax1.plot(swat_ml_predicted_df['year'], swat_ml_predicted_df['recharge [mm/yr]'], linewidth=1,color ='blue',label='Process-based simulation')
    ax1.plot(swat_ml_predicted_df['year'], swat_ml_predicted_df['ml_recharge [mm/yr]'], linewidth=1,linestyle='--', color ='red',label='ML Predicted', marker='o')

    ax1.set_xticks(range(2002, 2015, 3))
    ax1.set_ylabel('Recharge (mm/yr)')
    ax1.set_xlabel('Water Year')
    ax1.legend(loc='upper center',fontsize="14",frameon =False,bbox_to_anchor=(0.68, 0.999),borderaxespad=0.)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(swat_ml_predicted_df['year'], swat_ml_predicted_df['precip [mm]'], linewidth=1,linestyle='--', color ='green',label='Precipitation')
    ax1_twin.set_xticks(range(2002, 2015, 3))
    ax1_twin.set_ylabel('mm/yr')
    ax1_twin.legend(loc='upper center',fontsize="14",frameon =False,bbox_to_anchor=(0.61, 0.87),borderaxespad=0.)

    ax2.scatter( swat_ml_predicted_df['recharge [mm/yr]'], swat_ml_predicted_df['ml_recharge [mm/yr]'], linewidth=1,color ='k',  edgecolors='red')
    ax2.set_ylabel('Annual Mean ML  Recharge Predicted (mm/yr)')   
    ax2.set_xlabel('Annual Mean SWAT+ Recharge (mm/yr)')
    ax2.set_xticklabels([])

    low, high = ax2.get_xlim(), ax2.get_ylim()
    min_val = min(low[0], high[0])
    max_val = max(low[1], high[1])


    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    ax2.legend(loc='upper left', fontsize="14", frameon=False ,borderaxespad=1.5)
    low, high = ax2.get_xlim(), ax2.get_ylim()
    min_val = min(low[0], high[0])
    max_val = max(low[1], high[1])

    ax2.plot([min_val, max_val], [min_val, max_val], 
                 color='black', 
                 linestyle='--', 
                 linewidth=1.5, 
                 zorder=1, 
                 label='1:1 Line')

    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.legend(loc='upper left', fontsize="14", frameon=False ,borderaxespad=1.5)
    all_axes = [ax1, ax2]
    for ax in all_axes:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(True)    
            ax.spines[side].set_linewidth(1)      
            ax.spines[side].set_color('black')  
            ax.spines[side].set_linestyle('-')

    fig.subplots_adjust(wspace=3) 
    plt.savefig((ml_plots +  'standalone_model_simulated_predicted_recharge.png'), dpi=600)

def standalone_shap_ale_plot(standalone_model, x_train_data, y_train_data, ml_plots):
    '''
    Parameters
    ----------
    standalone_model :  Trained standalone model model path
    x_train_data : DataFrame containing train predictor variables 
    y_train_data : DataFrame containing train target variable 
    plots_path :  Director path for where the shap and ale plots are saved
    Returns
    -------
    None.

    '''
    
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    
    
    rf_explainer_model = copy.copy(standalone_model)
    rf_explainer_model.estimators_ = rf.estimators_[:50]
    shap_explainer_open_data = shap.TreeExplainer(rf_explainer_model)
    
    shap_explainer_open_data_sample = x_train_data.sample(n=5000, random_state=42)
    shap_values_open_data = shap_explainer_open_data.shap_values(shap_explainer_open_data_sample ,check_additivity=False)
    
   
    
    fig = plt.figure(figsize=(22, 20),linewidth=4, edgecolor="black") 
    gs = fig.add_gridspec(4, 2)
    
    left_ax = fig.add_subplot(gs[:,0]) 
    
    
    right_0 = fig.add_subplot(gs[0, 1])
    right_1 = fig.add_subplot(gs[1, 1])
    right_2 = fig.add_subplot(gs[2, 1])
    right_3 = fig.add_subplot(gs[3, 1])
    
    axes_flat = [right_0, right_1, right_2, right_3]
    my_features = ['Precip', 'slope', 'sandPcnt','cultivatedCrops'] 
    
    print("Calculating SHAP values...")
    
    
    
    plt.sca(left_ax) 
    
    print(" Making shap.plots.beeswarm...")
    
    renaming_dic = { 
                  'drainage_density': 'Drainage Density',  
                  'TRI': 'Topographic Roughness Index',  
                  'TWI':'Topographic Wetness Index',  
                  'SPI':'Stream Power Index',  
                  'slope': 'Slope',  
                  'distanceTo_stream':'Distance to Streams',
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
                  'thick_m' : 'Aquifer Thickness'   }
    
    shap_explainer_open_data_sample = shap_explainer_open_data_sample.rename(columns=renaming_dic)
    
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values_open_data,
            base_values=shap_explainer_open_data.expected_value,
            data=shap_explainer_open_data_sample , 
            feature_names=shap_explainer_open_data_sample.columns.tolist()
        ),
        max_display=len(shap_explainer_open_data_sample.columns),
        show=False,
        plot_size=None 
    )
    
    
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("Feature value", fontsize=22,fontweight='bold', labelpad=-40)
    cbar_ax.tick_params(labelsize=18,width=4)
    left_ax.set_xlabel("SHAP value (Impact on Groundwater Recharge (mm))", fontsize=22,fontweight='bold')
    left_ax.set_title("Global Feature Importance (SHAP)", fontsize=22,fontweight='bold')
    left_ax.tick_params(axis='both', which='major', labelsize=22)
    print("Calculating ALE values...")
    ale_explainer = skexplain.ExplainToolkit(
        estimators=('RandomForest',standalone_model), 
        X=x_train_data,
        y=y_train_data
    )
    
    ale_data = ale_explainer.ale(features=my_features, subsample=50000)
    name_map = {
        'Precip': 'Precipitation (mm)',
        'slope': 'Slope (Degree)',
        'sandPcnt': 'Percent Sand (%)',
        'cultivatedCrops': 'Cultivated Crops (Fraction)'
    }
    
    for i, feature in enumerate(my_features):
        ale_explainer.plot_ale(
            ale=ale_data,
            features=[feature],  
            ax=axes_flat[i],
            hist_kws={
                       'color': 'teal',
                      'alpha': 0.1}
        )
        clean_name = name_map.get(feature, feature)
        axes_flat[i].set_xlabel(clean_name, fontweight='bold', fontsize=22)
        axes_flat[i].tick_params(axis='both', which='major', labelsize=22)
        
        main_ax = axes_flat[i]
        twin_ax = None
        
        for other_ax in main_ax.figure.axes:
            if other_ax is not main_ax and other_ax.bbox.bounds == main_ax.bbox.bounds:
                twin_ax = other_ax
                break
    
        if twin_ax:
            twin_ax.tick_params(axis='y', labelsize=22)
    
    all_axes = [left_ax, right_0, right_1, right_2, right_3]
    
    for ax in all_axes:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(True)    
            ax.spines[side].set_linewidth(2)      
            ax.spines[side].set_color('black')  
            ax.spines[side].set_linestyle('-')
    
    fig.supylabel(
        "ALE Effect (mm)", 
        x=0.63,           
        y=0.5,              
        fontsize=22, 
        fontweight='bold'
    )
    
    
    fig.text(
        1.00, 0.5,        
        "Frequency (Log-Scale)", 
        va='center', 
        rotation=-270, 
        fontsize=22, 
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(ml_plots + 'standalone_shap_ale_plot.png', dpi = 750)
    
    
def standalone_predictor_ale_plots(standalone_model, x_train_data, y_train_data, ml_plots):
    '''
    Parameters
    ----------
    standalone_model:  Trained standalone_model model
    x_train_data : DataFrame containing train predictor variables 
    y_train_data : DataFrame containing  train target variable 
    plots_path :  Director path for where the shap and ale plots are saved
    Returns
    -------
    None.

    '''
    
    swat_var_names_group_short = {
         'group_1':['drainage_density','TRI', 'TWI','SPI', 'awsPcnt', 'distanceTo_stream', 'awcPcnt', 'ksatPcnt'],
         'group_2':['ET', 'Tmax', 'Tmin', 'developedlowIntensity', 'emergentHerbaceousWetlands', 'evergreenForest','grasslandOrHerbaceous','clayPcnt']
         }
    
    swat_var_names_group_long = {
         'group_1':['Drainage Density (m/Km\u00b2)','Topographic Roughness Index','Topographic Wetness Index','Stream Power Index','Available Water Storage (%)','Distance to Streams (m)', 
                    'Available Water Capacity (%)','Saturated Hydraulic Conductivity (µm/s)'],
         'group_2':['Actual Evapotranspiration (mm)','Maximum Air Temperature(\u00b0C)','Minimum Air Temperature (\u00b0C) ','Developed Low Intensity (Fraction)', 'Emergent Herbaceous Wet lands (Fraction)','Evergreen Forest (Fraction)',
                    'Grassland Or Herbaceous (Fraction)','Percent Clay (%)'] }
    
    ale_explainer = skexplain.ExplainToolkit(
        estimators=('RandomForest', standalone_model), 
        X=x_train_data, 
        y=y_train_data 
    )
    
    for my_features in swat_var_names_group_short.keys():
        ale_data = ale_explainer.ale(features=swat_var_names_group_short[my_features], subsample=50000)
    
        fig, axes = plt.subplots(4, 2, figsize=(22, 20), tight_layout=True)
        axes_flat = axes.ravel()
        for i, feature in enumerate(swat_var_names_group_short[my_features]):
            ale_explainer.plot_ale(
                ale=ale_data,
                features=[feature],  
                ax=axes_flat[i],
                hist_kws={
                           'color': 'teal',
                          'alpha': 0.3}
            )
            
            name_map = {
                swat_var_names_group_short[my_features][i]: swat_var_names_group_long[my_features][i]
            }
            clean_name = name_map.get(feature, feature)
            axes_flat[i].set_xlabel(clean_name, fontweight='bold', fontsize=22)
            axes_flat[i].tick_params(axis='both', which='major', labelsize=22)
            
            main_ax = axes_flat[i]
            twin_ax = None
            
            for other_ax in main_ax.figure.axes:
                if other_ax is not main_ax and other_ax.bbox.bounds == main_ax.bbox.bounds:
                    twin_ax = other_ax
                    break
            if twin_ax:
                twin_ax.tick_params(axis='y', labelsize=22)
    
            
            for side in ['top', 'bottom', 'left', 'right']:
                axes_flat[i].spines[side].set_visible(True)    
                axes_flat[i].spines[side].set_linewidth(2)      
                axes_flat[i].spines[side].set_color('black')  
                axes_flat[i].spines[side].set_linestyle('-')
                
            fig.supylabel(
                "ALE Effect (mm)", 
                x=0.00,             
                y=0.5,              
                fontsize=22, 
                fontweight='bold'
            )
    
    
            fig.text(
                1.00, 0.5,          
                "Frequency (Log-Scale)", 
                va='center', 
                rotation=-270, 
                fontsize=22, 
                fontweight='bold'
            )
            plt.tight_layout()
            plt.savefig(ml_plots +   f'standalone_{my_features}_ale_plot.png', dpi=600, bbox_inches='tight') 
            
def spatio_temporal_actual_vs_predicted_rasterplot(point_xy_loc, swat_tiff_file, ml_predicted_tiff_file, precip_tif_file, plots_dir):
    """
    Create raster, scatter plots for actual vs predicted values and scatter plot predicted residuals vs predicted values
    : plots_dir : directory path to store  plots 
    : return: none
    """
    plt.rcParams.update({'font.size': 34})
    
    mpl.rcParams['xtick.labelsize'] = 34
    mpl.rcParams['ytick.labelsize'] = 34
    
    plt.rcParams["font.weight"] = 'normal'
    plt.rcParams['figure.titlesize'] =34
    plt.rcParams['legend.fontsize'] = 34
    plt.rcParams['axes.labelsize'] = 34
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['font.family'] = ['arial']
    plt.rcParams['xtick.labelsize'] = 34
    plt.rcParams['ytick.labelsize'] = 34
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['lines.color'] = 'k'
    plt.subplots_adjust(wspace=0.0)
    
    df_file = pd.read_csv(point_xy_loc)
    long = df_file['long'].to_numpy()
    lat = df_file['lat'].to_numpy()
    min_long = min(long)
    max_long = max(long)
    min_lat = min(lat)
    max_lat = max(lat)
    
    fig, axs = plt.subplots(2,2,squeeze=False,figsize=(34,27))  
    swat_tiff = rio.open(swat_tiff_file)
    swat_tiff_matrix= swat_tiff.read(1)
    swat_tiff_matrix= ma.masked_less(swat_tiff_matrix , 0,copy=True)
    swat_tiff_matrix[swat_tiff_matrix == 0] = 1
    
    predicted_tiff = rio.open(ml_predicted_tiff_file)
    predicted_tiff_matrix = predicted_tiff.read(1)
    predicted_tiff_matrix  = ma.masked_outside(predicted_tiff_matrix,0,1740, copy=True)
    predicted_tiff_matrix[predicted_tiff_matrix == 0] = 1
    
    precip_tiff = rio.open(precip_tif_file)
    precip_tiff_matrix =  precip_tiff.read(1)
    precip_tiff_matrix  = ma.masked_less( precip_tiff_matrix, 0,copy=True)
    

    cax = axs[0,0].imshow(np.log10(swat_tiff_matrix) , extent=(min_long,max_long, min_lat,max_lat),origin='upper',vmin=0, vmax=3.5,cmap ='YlGnBu_r',aspect='auto')
    cax1 = axs[0,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.set_label('                                mm(log10)',fontsize=34)
    axs[0,0].set_xscale('linear')
    axs[0,0].set_xticks([])
    axs[0,0].set_ylabel('Northing (m)',fontsize=34)
    axs[0,0].set_title('SWAT Recharge  (mm)', pad=30,fontsize=34,fontweight='bold')
    axs[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    cax = axs[0,1].imshow(np.log10(predicted_tiff_matrix) , extent=(min_long,max_long, min_lat,max_lat),origin='upper',vmin=0, vmax=3.5, cmap ='YlGnBu_r',aspect='auto')
    cax2 = axs[0,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.set_label('                                mm(log10)',fontsize=34)
    axs[0,1].set_facecolor("white")
    axs[0,1].set_xscale('linear')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,1].set_title('ML Predicted Recharge (mm)', pad=40,fontsize=34,fontweight='bold')
    
    ml_recharge_precip_ratio = predicted_tiff_matrix/precip_tiff_matrix
  
    base_cmap = plt.get_cmap('tab20')
    new_cmap = mcolors.ListedColormap(base_cmap.colors)
    new_cmap.set_over('red')
    norm = mcolors.Normalize(vmin=0, vmax=1.0)

    cax = axs[1,0].imshow(ml_recharge_precip_ratio, extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap =new_cmap, norm=norm, aspect='auto')
    cax1 = axs[1,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    im = axs[1,0].imshow(ml_recharge_precip_ratio, extent=(min_long,max_long, min_lat,max_lat), cmap=new_cmap, norm=norm)
   
    ab = fig.colorbar( im, cax=cax1,orientation='horizontal', extend='max')
    tick_locations = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = ['0', '0.25', '0.5', '0.75', '> 1']
    ab.ax.set_xticks(tick_locations)
    ab.ax.set_xticklabels(tick_labels,fontsize=22)
    axs[1,0].set_xscale('linear')
    axs[1,0].set_xlabel('Easting (m)',fontsize=34)
    axs[1,0].set_ylabel('Northing (m)',fontsize=34)
    axs[1,0].set_title('ML Predicted Recharge (mm)/Precipitation (mm)', pad=30,fontsize=34,fontweight='bold')
    axs[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
   
    

    residual = swat_tiff_matrix - predicted_tiff_matrix
    mask = (residual > -150) & (residual < 150)
    filtered_residual = np.where(mask, residual, np.nan)
    residual_less250_matrix = filtered_residual.flatten()
    cax = axs[1,1].hist(residual_less250_matrix, color='green',edgecolor='k', linewidth=2,bins=30)
 
    axs[1,1].set_xlabel('Residuals (mm)',fontsize=34,)
    axs[1,1].set_ylabel('Count',fontsize=34,)
    axs[1,1].set_title('Residuals (mm)', pad=40,fontsize=34,fontweight='bold')
  
    
    plt.tight_layout()
    plt.savefig((plots_dir +   'spatio_temporal_SWAT_ml_raster_plot.png'), dpi=600)
    
def create_high_recharge_condtion_rasterplot(point_xy_loc, larb_shpfile, tif_path, plots_dir):
    """
    Create raster, scatter plots for actual vs predicted values and scatter plot predicted residuals vs predicted values
    : point_xy_loc: csv file path contain the centroids for 500m grid
    : larb_shpfile: Larb boundary shapefile polygon
    : tif_path: Tif file path contain raster files
    : plots_dir : directory path to store  plots 
    : return: none
    """
    plt.rcParams.update({'font.size': 30})
    plt.rc('xtick', labelsize=36) 
    plt.rc('ytick', labelsize=36)
    
    plt.subplots_adjust(wspace=0.0)
    
    df_file = pd.read_csv(point_xy_loc)
    long = df_file['long'].to_numpy()
    lat = df_file['lat'].to_numpy()
    min_long = min(long)
    max_long = max(long)
    min_lat = min(lat)
    max_lat = max(lat)
    
    
    larb_bndry = gpd.read_file(larb_shpfile) 
    
    precip_tif = rio.open(tif_path + 'Precip_2015.tif')
    precip_tif_matrix = precip_tif.read(1)
    precip_tif_matrix = ma.masked_less(precip_tif_matrix, 500,copy=True)
    
    
    slope_tif = rio.open(tif_path + 'slope_2015.tif')
    slope_tif_matrix = slope_tif.read(1)
    slope_tif_matrix = ma.masked_less(slope_tif_matrix , 12,copy=True)
    
    sandPcnt_tif = rio.open(tif_path + 'sandPcnt_2015.tif')
    sandPcnt_tif_matrix =  sandPcnt_tif.read(1)
    sandPcnt_tif_matrix = ma.masked_less(sandPcnt_tif_matrix , 65,copy=True)
    
    
    cultivatedCrops_tif = rio.open(tif_path + 'cultivatedCrops_2015.tif' )
    cultivatedCrops_tif_matrix = cultivatedCrops_tif.read(1)
    cultivatedCrops_tif_matrix = ma.masked_less(cultivatedCrops_tif_matrix , 0.70,copy=True)
    
    ET_tif = rio.open(tif_path + 'ET_2015.tif')
    ET_tif_matrix = ET_tif.read(1)
    ET_tif_matrix = ma.masked_less(ET_tif_matrix , 800,copy=True)
    
    evergreen_forest_tif = rio.open(tif_path + 'evergreenForest_2015.tif')
    evergreen_forest_tif_matrix = evergreen_forest_tif.read(1)
    evergreen_forest_tif_matrix = ma.masked_less(evergreen_forest_tif_matrix , 0.75,copy=True)
    
    
    fig, axs = plt.subplots(3,2,squeeze=False,figsize=(40,48))  
    
    
    scalebar = ScaleBar(1.33, location='lower right', box_alpha=0.2, length_fraction=0.2,rotation='horizontal-only')
    
    larb_bndry.plot(ax= axs[0,0], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[0,0].imshow( precip_tif_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper', cmap ='YlGnBu',aspect='auto')
    cax1 = axs[0,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.set_label('                                     mm', fontsize =36)
    axs[0,0].set_xscale('linear')
    axs[0,0].set_xticks([])
    # axs[0,0].set_xlabel('Longitude (m)')
    axs[0,0].set_ylabel('Northing (m)', fontsize =40)
    axs[0,0].set_title('Precipitation ', pad=25, fontsize =40, fontweight='bold')
    axs[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    axs[0,0].add_artist(scalebar)
    # axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="24")
    
    larb_bndry.plot(ax= axs[0,1], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[0,1].imshow(slope_tif_matrix, extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap ='terrain', aspect='auto')
    cax2 = axs[0,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Degree', fontsize =36)
    axs[0,1].set_facecolor("white")
    axs[0,1].set_xscale('linear')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    # axs[0,1].set_xlabel('Easting (m)')
    # axs[0,1].set_ylabel('Northing (m)')
    axs[0,1].set_title('Slope', pad=25,fontsize =40, fontweight='bold')
    
    # axs[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axs[0,1].annotate('c)',xy=(-100, 39.13),fontsize="24")
    
    larb_bndry.plot(ax= axs[1,0], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[1,0].imshow(sandPcnt_tif_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'cividis', aspect='auto')
    cax1 = axs[1,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Percent', fontsize =36)
    axs[1,0].set_xscale('linear')
    # axs[1,0].set_xlabel('Easting (m)')
    axs[1,0].set_ylabel('Northing (m)', fontsize =40)
    axs[1,0].set_xticks([])
    axs[1,0].set_title('Sand Percent', pad=25,  fontsize =40, fontweight='bold')
    axs[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # larb_bndry.plot(ax= axs[1,0], facecolor="none", edgecolor='black', linewidth=2.5)
    
    # axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="24")
    
    larb_bndry.plot(ax= axs[1,1], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[1,1].imshow( ET_tif_matrix ,  extent=(min_long,max_long, min_lat,max_lat),origin='upper', vmin=800, vmax=1100, cmap = 'viridis', aspect='auto')
    cax2 = axs[1,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    # ab.ax.set_yticklabels(['600', '900', '1000','1100'])
    ab.set_label('                                     mm', fontsize =36)
    axs[1,1].set_facecolor("white")
    axs[1,1].set_xscale('linear')
    axs[1,1].set_yticks([])
    axs[1,1].set_xticks([])
    # axs[1,1].set_xlabel('Easting (m)')
    # axs[1,1].set_ylabel('Northing (m)')
    axs[1,1].set_title('Evapotranspiration', pad=25, fontsize =40, fontweight='bold')
    axs[1,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    larb_bndry.plot(ax= axs[2,0], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[2,0].imshow(cultivatedCrops_tif_matrix,  extent=(min_long,max_long, min_lat,max_lat),origin='upper', vmin=0.70, vmax=1.0, cmap = 'Greens', aspect='auto')
    cax2 = axs[2,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Fraction', fontsize =36)
    axs[2,0].set_facecolor("white")
    axs[2,0].set_xscale('linear')
    axs[2,0].set_xlabel('Easting (m)', fontsize =40)
    axs[2,0].set_ylabel('Northing (m)', fontsize =40)
    axs[2,0].set_title('Cultivated Crops', pad=25, fontsize =40, fontweight='bold')
    axs[2,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    larb_bndry.plot(ax= axs[2,1], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[2,1].imshow(evergreen_forest_tif_matrix,  extent=(min_long,max_long, min_lat,max_lat),origin='upper', vmin=0.80, vmax=1.0, cmap = 'brg', aspect='auto')
    cax2 = axs[2,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Fraction', fontsize =36)
    axs[2,1].set_facecolor("white")
    axs[2,1].set_xscale('linear')
    axs[2,1].set_yticks([])
    axs[2,1].set_xlabel('Easting (m)', fontsize =40)
    # axs[2,1].set_ylabel('Northing (m)')
    axs[2,1].set_title('Evergreen Forest', pad=25,fontsize =40, fontweight='bold')
    axs[2,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    
    plt.tight_layout()
    plt.savefig(plots_dir +   'high_recharge_condition_spatio_ml_raster_plots.png', dpi=600,pad_inches=0.2)