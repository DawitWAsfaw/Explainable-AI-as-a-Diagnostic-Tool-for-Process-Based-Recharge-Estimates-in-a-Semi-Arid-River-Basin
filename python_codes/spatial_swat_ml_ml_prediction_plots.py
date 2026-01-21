import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import seaborn as sns
import rasterio as rio
import random
"""
Created on Thu Dec 18 21:56:40 2025

@author: dasfaw
"""

def spatio_temporal_actual_vs_predicted_rasterplot(point_xy_loc, swat_tiff_file, ml_predicted_tiff_file, precip_tif_file, plots_dir):
    """
    Create raster, scatter plots for actual vs predicted values and scatter plot predicted residuals vs predicted values
    : plots_dir : directory path to store  plots 
    : return: none
    """
    plt.rcParams.update({'font.size': 24})
    plt.subplots_adjust(wspace=0.0)
    
    df_file = pd.read_csv(point_xy_loc)
    long = df_file['long'].to_numpy()
    lat = df_file['lat'].to_numpy()
    min_long = min(long)
    max_long = max(long)
    min_lat = min(lat)
    max_lat = max(lat)
    
    fig, axs = plt.subplots(3,1,squeeze=False,figsize=(34,27))  
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
    ab.set_label('                                mm(log10)')
    axs[0,0].set_xscale('linear')
    axs[0,0].set_xticks([])
    # axs[0,0].set_xlabel('Longitude (m)')
    axs[0,0].set_ylabel('Latitude (m)')
    axs[0,0].set_title('SWAT Recharge  (mm)', pad=30)
    axs[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="24")
    
    cax = axs[0,1].imshow(np.log10(predicted_tiff_matrix) , extent=(min_long,max_long, min_lat,max_lat),origin='upper',vmin=0, vmax=3.5, cmap ='YlGnBu_r',aspect='auto')
    cax2 = axs[0,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.set_label('                                mm(log10)')
    axs[0,1].set_facecolor("white")
    axs[0,1].set_xscale('linear')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    # axs[0,1].set_xlabel('Longitude (m)')
    # axs[0,1].set_ylabel('Latitude (m)')
    axs[0,1].set_title('ML Predicted Recharge (mm)', pad=40)
    # axs[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axs[0,1].annotate('c)',xy=(-100, 39.13),fontsize="24")
    

    cax = axs[1,0].imshow(predicted_tiff_matrix/precip_tiff_matrix, extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'YlGnBu_r', aspect='auto')
    cax1 = axs[1,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.set_label('                            mm')
    axs[1,0].set_xscale('linear')
    axs[1,0].set_xlabel('Longitude (m)')
    axs[1,0].set_ylabel('Latitude (m)')
    axs[1,0].set_title('ML Predicted Recharge (mm)/Precipitation (mm)', pad=30)
    axs[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="24")
    
    

    
    residual = swat_tiff_matrix - predicted_tiff_matrix
    residual_less250 =  ma.masked_greater(residual, 250,copy=True)
    cax = axs[1,1].imshow(residual_less250, extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'bwr',aspect='auto')
    cax2 = axs[1,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.set_label('                       mm')
    axs[1,1].set_facecolor("white")
    axs[1,1].set_xscale('linear')
    axs[1,1].set_yticks([])
    axs[1,1].set_xlabel('Longitude (m)')
    # axs[1,1].set_ylabel('Latitude (m)')
    axs[1,1].set_title('Residuals (mm)', pad=40)
    axs[1,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    
    # renaming_dic = {swat_ml_recharge_ave.columns[2]:'SWAT Recharge  (mm)',
    #                 swat_ml_recharge_ave.columns[3]:'ML Predicted Recharge (mm)',
    #               }

    
    # swat_ml_recharge_ave.rename(columns=renaming_dic, inplace=True)
    # swat_ml_recharge_df['Residuals (mm)'] = swat_ml_recharge_df['SWAT Recharge  (mm)']- swat_ml_recharge_df['ML Predicted Recharge (mm)']


    # axs[1,1].axhline(color ='r',linestyle = '--',linewidth=1.5)

    # axs[1,1].plot(swat_ml_recharge_ave['year'], swat_ml_recharge_ave['SWAT Recharge  (mm)'], linewidth=1,color ='blue',label='SWAT Recharge  (mm)')
    # axs[1,1].plot(swat_ml_recharge_ave['year'], swat_ml_recharge_ave['ML Predicted Recharge (mm)'], linewidth=1,color ='red',label='ML Predicted Recharge (mm)')
    # axs[1,1].set_facecolor("white")

    # axs[1,1].set_xlabel('Water Year')
    # axs[1,1].set_ylabel('mm')
    # axs[1,1].annotate('d)',xy=(400,-150),fontsize="24")
    
    plt.tight_layout()
    plt.savefig((plots_dir +   'spatio_temporal_SWAT_ml_raster_plot.png'), dpi=600)
    
    # plt.close('all')
    
    
    
    
main_dir = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/plots/predictions/for_prediction_map_plot/'  
point_xy_loc = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/point/larb_grid_pnt_xy.csv '
swat_tiff_file  = main_dir  + 'cropped/recharge_2015.tif' 
ml_predicted_tiff_file = main_dir  + 'cropped/larb_predicted_recharge_2015.tif'
precip_tif_file = main_dir  + 'cropped/ppt_2015.tif'
plots_dir = main_dir

spatio_temporal_actual_vs_predicted_rasterplot(point_xy_loc, swat_tiff_file, ml_predicted_tiff_file, precip_tif_file, plots_dir) 