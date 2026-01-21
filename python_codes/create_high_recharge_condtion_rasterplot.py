import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import seaborn as sns
import rasterio as rio
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import random
"""
Created on Thu Dec 18 21:56:40 2025

@author: dasfaw
"""

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
    axs[0,0].set_ylabel('Latitude (m)', fontsize =36)
    axs[0,0].set_title('Precipitation ', pad=25, fontsize =36)
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
    # axs[0,1].set_xlabel('Longitude (m)')
    # axs[0,1].set_ylabel('Latitude (m)')
    axs[0,1].set_title('Slope', pad=25, fontsize =36)
    
    # axs[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axs[0,1].annotate('c)',xy=(-100, 39.13),fontsize="24")
    
    larb_bndry.plot(ax= axs[1,0], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[1,0].imshow(sandPcnt_tif_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'cividis', aspect='auto')
    cax1 = axs[1,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Percent', fontsize =36)
    axs[1,0].set_xscale('linear')
    # axs[1,0].set_xlabel('Longitude (m)')
    axs[1,0].set_ylabel('Latitude (m)', fontsize =36)
    axs[1,0].set_xticks([])
    axs[1,0].set_title('Sand Percent', pad=25, fontsize =36)
    axs[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # larb_bndry.plot(ax= axs[1,0], facecolor="none", edgecolor='black', linewidth=2.5)
    
    # axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="24")
    
    larb_bndry.plot(ax= axs[1,1], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[1,1].imshow( ET_tif_matrix ,  extent=(min_long,max_long, min_lat,max_lat),origin='upper', vmin=800, vmax=1100, cmap = 'viridis', aspect='auto')
    cax2 = axs[1,1].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     mm', fontsize =36)
    axs[1,1].set_facecolor("white")
    axs[1,1].set_xscale('linear')
    axs[1,1].set_yticks([])
    axs[1,1].set_xticks([])
    # axs[1,1].set_xlabel('Longitude (m)')
    # axs[1,1].set_ylabel('Latitude (m)')
    axs[1,1].set_title('Evapotranspiration', pad=25, fontsize =36)
    axs[1,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    larb_bndry.plot(ax= axs[2,0], color='whitesmoke', edgecolor='black', linewidth=2.5, zorder=0)
    cax = axs[2,0].imshow(cultivatedCrops_tif_matrix,  extent=(min_long,max_long, min_lat,max_lat),origin='upper', vmin=0.70, vmax=1.0, cmap = 'Greens', aspect='auto')
    cax2 = axs[2,0].inset_axes([0.35, 0.95, 0.6, 0.02])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.tick_params(labelsize=36)
    ab.set_label('                                     Fraction', fontsize =36)
    axs[2,0].set_facecolor("white")
    axs[2,0].set_xscale('linear')
    axs[2,0].set_xlabel('Longitude (m)', fontsize =36)
    axs[2,0].set_ylabel('Latitude (m)', fontsize =36)
    axs[2,0].set_title('Cultivated Crops', pad=25, fontsize =36)
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
    axs[2,1].set_xlabel('Longitude (m)', fontsize =36)
    # axs[2,1].set_ylabel('Latitude (m)')
    axs[2,1].set_title('Evergreen Forest', pad=25, fontsize =36)
    axs[2,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    
    plt.tight_layout()
    plt.savefig(plots_dir +   'high_recharge_condition_spatio_ml_raster_plot_6.png', dpi=600,pad_inches=0.2)
    
    # plt.close('all')




tif_path = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/ml_analysis_results_and_eda/high_recharge_conditon_rast_plots/'
point_xy_loc = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/point/larb_grid_pnt_xy.csv'

larb_shpfile = 'D:/topic_2/deep_percolation_estimates_project/500m_grid_model_larb/gis_files/bndry/larb_bndry_15N.shp'
plots_dir =  'C:/Users/dasfaw/Documents/Fall_2025/research/topic_2/manuscript/plots/'

create_high_recharge_condtion_rasterplot(point_xy_loc, larb_shpfile, tif_path, plots_dir)