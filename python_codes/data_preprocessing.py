import ee
import requests
import zipfile
import os
import shutil
import geopandas as gpd
from glob import glob
import rasterio as rio
from rasterio.merge import merge
import pandas as pd
import numpy as np

from osgeo import gdal
import subprocess
from rasterstats import zonal_stats


from collections import defaultdict 
from itertools import islice

def make_gdal_sys_call_str(gdal_path, gdal_command, args, verbose=True):
    """
    Make GDAL system call string
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param gdal_command: GDAL command to use
    :param args: GDAL arguments as a list
    :param verbose: Set True to print system call info
    :return: GDAL system call string,
    """

    sys_call = [gdal_path + gdal_command] + args
    if os.name == 'nt':
        gdal_path += 'OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call

def get_raster_extents(gdal_raster):
    """
    Get Raster Extents
    :param gdal_raster: Input gdal raster object
    :return: (Xmin, YMax, Xmax, Ymin)
    """
    transform = gdal_raster.GetGeoTransform()
    ulx, uly = transform[0], transform[3]
    xres, yres = transform[1], transform[5]
    lrx, lry = ulx + xres * gdal_raster.RasterXSize, uly + yres * gdal_raster.RasterYSize
    return str(ulx), str(lry), str(lrx), str(uly)

def reproject_raster(input_raster_file, outfile_path, resampling_factor=1, resampling_func=gdal.GRA_NearestNeighbour,
                     downsampling=False, from_raster=None, keep_original=False, gdal_path='C:/OSGeo4W/', verbose=True):
    """
    Reproject raster using GDAL system call
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster
    :param keep_original: Set True to only use the new projection system from 'from_raster'. The original raster extent
    is not changed
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    src_raster_file = gdal.Open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        rfile = gdal.Open(from_raster)
        resampling_factor = 1
    src_band = rfile.GetRasterBand(1)
    transform = rfile.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    extent = get_raster_extents(rfile)
    dst_proj = rfile.GetProjection()
    no_data = src_band.GetNoDataValue()
    if not downsampling:
        resampling_factor = 1 / resampling_factor
    xres, yres = xres * resampling_factor, yres * resampling_factor

    resampling_dict = {gdal.GRA_NearestNeighbour: 'near', gdal.GRA_Bilinear: 'bilinear', gdal.GRA_Cubic: 'cubic',
                       gdal.GRA_CubicSpline: 'cubicspline', gdal.GRA_Lanczos: 'lanczos', gdal.GRA_Average: 'average',
                       gdal.GRA_Mode: 'mode', gdal.GRA_Max: 'max', gdal.GRA_Min: 'min', gdal.GRA_Med: 'med',
                       gdal.GRA_Q1: 'q1', gdal.GRA_Q3: 'q3'}
    resampling_func = resampling_dict[resampling_func]
    args = ['-t_srs', dst_proj, '-te', extent[0], extent[1], extent[2], extent[3],
            '-dstnodata', str(no_data), '-r', str(resampling_func), '-tr', str(xres), str(yres), '-ot', 'Float32',
            '-overwrite', input_raster_file, outfile_path]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdalwarp', args=args, verbose=verbose)
    subprocess.call(sys_call)




def reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True):
    """
    Reproject rasters in a directory
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param ref_raster: Reference raster file to consider while reprojecting
    :param outdir: Output directory for storing reprojected rasters
    :param pattern: Raster extension
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        reproject_raster(raster_file, from_raster=ref_raster, outfile_path=out_raster, gdal_path=gdal_path,
                         verbose=verbose)
        
def mosaic_subbasin_rasters(input_raster_dir, output_raster_dir, var_names, start_yr, end_yr, pattern = '*.tif'):
    """
    Description
    ===========
    This function mosaic individual subbasin raster file download from GEE into a single raster file
    Parameters:
        input_raster_dir: Directory path for subbasin raster files
        output_raster_dir: Directory path to save the final mosaic raster file
        var_name: the list of variable names (For instance, precipitation_year.tif)
        start_year: the start of the study period
        end_year: the end of the study perion
        pattern: the raster file extension. Default is set to .tif
    
    """
    
    for var_name in var_names:
        for year1, year2 in zip(range(start_yr, end_yr),range(start_yr+1, end_yr+1)) :
            pattern = f"*{var_name}_{year2}.tif"
            filename =  output_raster_dir  + var_name  + f'_{year2}' + '.tif'
            
            listoffiles = os.path.join(input_raster_dir, pattern)
            # print(listoffiles)
            raster_files = glob(listoffiles)
            # print(raster_files)
            
            src_files_to_mosaic = []
          
            for file in raster_files:
                src = rio.open(file)
                
                out_meta = src.meta.copy()
                src_files_to_mosaic.append(src)
            print(len(src_files_to_mosaic))
            if len(src_files_to_mosaic) ==0:
                print('There are no raster files to mosaic. Please, provide a proper: ',input_raster_dir )
            print('Now mosaicing: ', var_name, f'for water year {year1} to {year2}')
            mosaic, out_trans = merge(src_files_to_mosaic)

            out_meta.update({"driver": "GTiff",
                              "height": mosaic.shape[1],
                              "width": mosaic.shape[2],
                              "transform": out_trans,
                              "crs": "+proj=longlat +datum=WGS84 +no_defs +type=crs"})
            
            with rio.open(filename, "w", **out_meta) as dest:
                dest.write(mosaic)
                
def extract_data(zip_dir, zip_output_dir, rename_extracted_files=False):
    """
    Extract data from zip file
    :param zip_dir: Input zip directory
    :param zip_output_dir: Output directory to write extracted files
    :param rename_extracted_files: Set True to rename extracted files according the original zip file name
    :return: None
    """

    print('Extracting zip files...')
    for zip_file in glob(zip_dir + '*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_extracted_files:
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_file[zip_file.rfind(os.sep) + 1: zip_file.rfind('.')] + '.tif'
                zip_ref.extract(zip_info, path=zip_output_dir)
            else:
                zip_ref.extractall(path=zip_output_dir)
                
               

def extract_raster_values_at_points(input_raster_dir, point_shapefile, output_csv_dir):
    
    """
    Description
    ===========
    This function extract raster pixel value using a point x and y at the center of 500m grid
    Parameters:
        input_raster_dir: Directory path for raster files 
        extract_points_path: Point shapefile with x and y data
        output_csv_dir: Directory path to save the  extract pixel values to csv file
    
    """
    values = []
    raster = rio.open(input_raster_dir)
    point_gdf = gpd.read_file(point_shapefile)
    variable =os.path.basename(input_raster_dir).split('/')[-1]
    # print(variable[-8:-4])
    variable = variable[variable.rfind(os.sep) + 1: variable.rfind('.')]
    
    year = variable[-4:]
    col_name = variable[:-5]
    print(year)
    print(col_name)
    df1 = point_gdf['gridid']
    for index,point in point_gdf.iterrows():
        x,y = point.geometry.x, point.geometry.y
        row,col = raster.index(x,y)  # get the row and column index of in the raster
        value = raster.read(1, window=((row, row+1), (col, col+1))) # read the raster at the point
        value = np.where(np.isnan(value ) | np.isinf(value), -9999, value )
        value = value.reshape(value.shape[0] * value .shape[1])
        values.append(value)
      
    df2 = pd.DataFrame(data ={col_name:values})
    df2[col_name] = df2[col_name].str.get(0)
    df = pd.concat([df1, df2], axis=1)
    # df.insert(loc =0,column ='month',value = month)
    df.insert(loc =1,column ='year',value = year)
    # df.insert(loc =2,column ='x',value = x)
    # df.insert(loc =3,column ='y',value = y)
    df.to_csv(output_csv_dir + col_name + f'_{year}.csv',index= False)
    raster.close()

#==================================================================================================================
# Extract Raster file from a multiple raster files

#==================================================================================================================
def extract_rasters_values_at_points(input_raster_dir,point_shapefile, output_dir,pattern='*.tif'):

    for file in glob(input_raster_dir + pattern):
        extract_raster_values_at_points(file,point_shapefile,output_dir)
        # print(output_path)

def duplicate_files(input_csv_dir, output_dir, var_names, pattern='.csv' ):
    """
    This functions duplicates csv files for static properties
    ----------
    input_dir : Directory path where csv file is stored.
    output_dir: Directory path where duplicated csv files are stored. 
    var_names : list of variable names
    pattern :  The default is '.csv'.
    Returns
    -------
    None.

    """
    for year in range(2001,2016):
        for var in var_names:
             output_file_name = output_dir + '/' + var + '_' + str(year) +  pattern 
             # print(output_file_name)
             
             input_file = pd.read_csv(input_csv_dir + f'{var}_2000.csv')
             input_file['year'] = year
             input_file.to_csv(output_file_name,index=False)
    
    
def concat_csv_files(main_dir_path, years_list, var_names_list):
    """
    Concatenates raster extracted yealy csv values for individual variables
    : input_csv_file_path: string path director for input csv files
    :  output_file_path: string path director to store concatenated csv files
    : return: none
    """
    
    input_csv_file_path =  main_dir_path + 'csv/'
    output_file_path  =  main_dir_path + 'all_years_csv/'
    names = []
    pattern = '*.csv'
    for file in glob(input_csv_file_path + pattern):
        variable =os.path.basename(file).split('/')[-1]
        name = variable[variable.rfind(os.sep) + 1: variable.rfind('_')+1]
        names.append(name)
    names = list(set(names))
    
    for name in var_names_list:
        
        shutil.move(input_csv_file_path + name + '_2000.csv',  main_dir_path + name + '_2000.csv')
        df =pd.read_csv(main_dir_path + name + '_2000.csv')
        
        print(df.columns) 
        pattern = f'{name}_*.csv'
        for f in glob(input_csv_file_path + pattern):
            df1 = pd.read_csv(f)
            print(df1.columns)
            df= pd.concat([df,df1])
            df.to_csv(output_file_path + name + '_all.csv',index = False)
        return df               
        shutil.move(main_dir_path + name + '_2000.csv',input_csv_file_path + name + '_2000.csv')
        
def process_climate_data(start_yr, end_yr, shps_dir, main_dir_path):
    """
    Generates climate data including ET from openET, Precipitation, Minimum and maximum air temperature data from PRISM data..
    year_list: List of years in %Y format
    start_month: Start month in %m format
    end_month: End month in %m format
    shps_dir: Area of interest shapefiles (must be in WGS84) and also contains raster reference for reprojecting and mosaicing downloaded data
    main_dir_path: Main directory path  which contains folders including 'zipped', 'unzipped', 'mosaic', 'reprojected', 'csv', 'csv_all_years'
    return: 
        climate_date Dataframe
    """

    ee.Initialize()
    ee.Authenticate()
    
    openET = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
    prism= ee.ImageCollection("OREGONSTATE/PRISM/AN81m")
  
    names = []
    pattern = '*.shp'
    for file in glob(shps_dir + pattern):
        variable = os.path.basename(file).split('/')[-1]
        print(variable)
        name = variable[variable.rfind(os.sep) +1: variable.rfind('.')]
        names.append(name)
    names = list(set(names))
    for name in names:
        print('Dowloading', name, '...;')
        files = glob(shps_dir + name + pattern)
        for kk, file in enumerate(files):
            
            aoi_shp = gpd.read_file(file)
            minx, miny, maxx, maxy = aoi_shp.geometry.total_bounds
            gee_aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
            
            for year1, year2 in zip(range(start_yr, end_yr),range(start_yr+1, end_yr+1)) :
                start_mn = '-10' 
                end_mn = '-09'
                start_dy = '-01'
                end_dy = '-30'
                start_date = pd.to_datetime(str(year1) + start_mn +  start_dy)
                end_date = pd.to_datetime(str(year2) + end_mn +  end_dy)
                print('Collecting data for water year of : ', start_date,'to', end_date,'\n')
                    
                openET_Total = openET.select('et_ensemble_mad').filterDate(start_date, end_date).sum().toDouble()
                prism_ppt_Total = prism.select('ppt').filterDate(start_date, end_date).sum().toDouble()
                prismtmaxTotal = prism.select('tmax').filterDate(start_date, end_date).mean().toDouble()
                prismtminTotal = prism.select('tmin').filterDate(start_date, end_date).mean().toDouble()
                
                openET_url = openET_Total.getDownloadUrl({
                    'scale': 500,
                    'crs': 'EPSG:4326',
                    'region': gee_aoi
                })
                prism_url = prism_ppt_Total.getDownloadUrl({
                    'scale': 500,
                    'crs': 'EPSG:4326',
                    'region': gee_aoi
                })
                
             
            
                prism_tmax_url = prismtmaxTotal.getDownloadUrl({
                    'scale': 500,
                    'crs': 'EPSG:4326',
                    'region': gee_aoi
                })
                
                prism_tmin_url = prismtminTotal.getDownloadUrl({
                    'scale': 500,
                    'crs': 'EPSG:4326',
                    'region': gee_aoi
                })
                
     
                gee_vars = ['openET_','ppt_', 'tmax_','tmin_']
                gee_links = [openET_url,prism_url,prism_tmax_url,prism_tmin_url]
                
                for gee_var, gee_url in zip(gee_vars, gee_links):
                    local_file_name = main_dir_path + 'zipped/'+ name +  '_' + gee_var + f'{year2}.zip'
                    print('Dowloading', local_file_name, '...')
                    r = requests.get(gee_url, allow_redirects=True, timeout=60)
                    open(local_file_name, 'wb').write(r.content)
                    
    zip_dir = main_dir_path  + 'zipped/'
    unzip_output_dir  = main_dir_path  + 'unzipped/'
   
    extract_data(zip_dir, unzip_output_dir, rename_extracted_files=True) 
    print('File extraction completed!\n')
    
    
    input_raster_dir = main_dir_path + 'unzipped/'
    output_raster_dir =    main_dir_path + 'mosaic/'
    var_names = ['openET','tmax','tmin','ppt'] 
    
    mosaic_subbasin_rasters(input_raster_dir, output_raster_dir, var_names, start_yr, end_yr, pattern = '*.tif')       
    print('Image mosaiced to study area extent!\n')    
    ref_raster =  shps_dir + 'ref_raster.tif'
    
    input_raster_dir =  main_dir_path + 'mosaic/'
    outdir = main_dir_path +  'reprojected/'
 
    reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)
    print('Image reprojection completed!\n')
    
    input_raster_dir =  main_dir_path +  'reprojected/'
    point_shapefile = 'D:/topic_2/GIS/stuy_area_map/larb_grid_pnt_xy.shp'
    
    output_path = main_dir_path +  'csv/'
    extract_rasters_values_at_points(input_raster_dir,point_shapefile, output_path, pattern='*.tif')     
    print('Images pixel values are now extracted and saved to csv files!\n')
    
    years_list = range(2001,2016)
    var_names_list =  ['openET','tmax','tmin','ppt']    
    concat_csv_files(main_dir_path, years_list, var_names_list)
    print('Yearly csv files are now merged into one csv file!\n')


def process_soil_phys_proporties_data(ref_raster_dir, point_shapefile, main_dir_path):
    """
    This function creates dataframe containing soil data
    Parameters
    ----------
    ref_raster_dir: Reference raster used to reproject soil data to 500m spatial resolution and 'NAD 1983 UTM Zone 15N' Projection Coordinate System
    point_shapefile : Shapefile containing spatial point to extract  pixel values
    main_dir_path : Main directory path containing folders 'csv','tif','csv_ref', 'csv_all_years'
        Note: 'tif' folder should contain tif files for:See the data preprocessing text file to download soil data
            'clay_2000.tif', 'sand_2000.tif', 'silt_2000.tif', 
            'aws_2000.tif', 'awc_2000.tif', 'floodfcl_2000.tif',
            'pondfcl_2000.tif'
    Returns
    -------
        soil data Dataframe
    """
    
    ref_raster =  ref_raster_dir + 'ref_raster.tif'
    
    input_raster_dir =  main_dir_path + 'tif/'
    outdir = main_dir_path +  'reprojected/'
 
    reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)
    print('Image reprojection completed!\n')
    
    input_raster_dir =  main_dir_path +  'reprojected/'
    output_path = main_dir_path +  'csv/'
    extract_rasters_values_at_points(input_raster_dir,point_shapefile, output_path, pattern='*.tif')     
    print('Images pixel values are now extracted and saved to csv files!\n')
    
    input_csv_dir =  output_path
    output_dir = main_dir_path+ 'csv_all_years/'
    var_names =  ['sand','clay','silt','awc','aws','floodfcl','pond_fcl']   
    duplicate_files(input_csv_dir, output_dir, var_names, pattern='.csv' )
    
    years_list = range(2001,2016)
    
    concat_csv_files(main_dir_path, years_list, var_names)
    print('Yearly csv files are now merged into one csv file!\n')

def process_hydrogeological_data(ref_raster_dir,point_shapefile, main_dir_path):
    """
    This function creates dataframe containing hydrogeological data
    Parameters
    ----------
    ref_raster_dir: Reference raster used to reproject hydrogeological data to 500m spatial resolution and 'NAD 1983 UTM Zone 15N' Projection Coordinate System
    point_shapefile : Shapefile containing spatial point to extract  pixel values
    main_dir_path : Main directory path containing folders 'csv','tif','reprojected','csv_ref', 'csv_all_years'
        Note: 'reprojected' folder should contain tif files for:
            'thick_m_2000.tif'(aquifer thickness), 'sy_2000.tif'(specific yield), 'k_mday_2000.tif'(hydraulic conductivity), 
            'depth2Water_tbl_2000.tif'(Depth to water table)
    Returns
    -------
       hydrogeological data Dataframe
    """
    
    ref_raster =  ref_raster_dir + 'ref_raster.tif'
    
    input_raster_dir =  main_dir_path + 'tif/'
    outdir = main_dir_path +  'reprojected/'
 
    reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)
    print('Image reprojection completed!\n')
    
    input_raster_dir =  main_dir_path +  'reprojected/'
    output_path = main_dir_path +  'csv/'
    
    
    extract_rasters_values_at_points(input_raster_dir,point_shapefile, output_path, pattern='*.tif')     
    print('Images pixel values are now extracted and saved to csv files!\n')
    
    input_csv_dir =  output_path
    output_dir = main_dir_path+ 'csv_all_years/'
    var_names =  ['sy','k_mday','thick_m','depth2Water_tbl']   
    duplicate_files(input_csv_dir, output_dir, var_names, pattern='.csv' )
    
    years_list = range(2001,2016)
    
    concat_csv_files(main_dir_path, years_list, var_names)
    print('Yearly csv files are now merged into one csv file!\n')

def process_hydrological_factors_data(ref_raster_dir,point_shapefile, main_dir_path):
    """
    This function creates dataframe containing hydrological factors data
    Parameters
    ----------
    ref_raster_dir: Reference raster used to reproject hydrological factors data to 500m spatial resolution and 'NAD 1983 UTM Zone 15N' Projection Coordinate System.
    point_shapefile : Shapefile containing spatial point to extract  pixel values
    main_dir_path : Main directory path containing folders 'csv','tif','reprojected','csv_ref', 'csv_all_years'
        Note: 'reprojected' folder should contain tif files for:
            'TWI_2000.tif'(Topographic Wetness Index), 'TRI_2000.tif'(Topographic Roughness Index), 
            'SPI_2000.tif'(Stream Power Index),'drainage_density_2000.tif'(Drainage density),
            'dTs_2000.tif' (Distance to Stream),"slope_2000.tif" (Slope derived from SRTM data)
    Returns
    -------
       hydrological factors data Dataframe
    """
    ref_raster =  ref_raster_dir + 'ref_raster.tif'
    input_raster_dir =  main_dir_path + 'tif/'
    outdir = main_dir_path +  'reprojected/'
 
    reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)
    print('Image reprojection completed!\n')
    
    input_raster_dir =  main_dir_path +  'reprojected/'
    
    output_path = main_dir_path +  'csv/'
    extract_rasters_values_at_points(input_raster_dir,point_shapefile, output_path, pattern='*.tif')     
    print('Images pixel values are now extracted and saved to csv files!\n')
    
    input_csv_dir =  output_path
    output_dir = main_dir_path+ 'csv_all_years/'
    var_names =  ['TWI','TRI','SPI','dTs','drainage_density','slope']   
    duplicate_files(input_csv_dir, output_dir, var_names, pattern='.csv' )
    
    years_list = range(2001,2016)
    
    concat_csv_files(main_dir_path, years_list, var_names)
    print('Yearly csv files are now merged into one csv file!\n')
    
    
def process_lulc_data(lulc_ref_raster_dir, grid_shapefile, main_dir_path):
    """
    This function creates percent of lulc class cover within 500m grid using zonal statistics  and saves into dataframe
    Parameters
    ----------
    lulc_ref_raster : Reference raster used to reproject lulc data to 'NAD 1983 UTM Zone 15N' Projection Coordinate System.lulc_ref_raster - 30m spatial resolution
    grid_shapefile : Directory path for shapefile containg 500m grid resolution
    main_dir_path : Main directory path containing folders 'csv','tif','reprojected', 'csv_ref', 'csv_all_years'
        Note: 'reprojected' folder should contain Annual lulc data. See the data preprocessing text file to download lulc data
    Returns
    -------
    None.

    """
    
    ref_raster =  lulc_ref_raster_dir + 'lulc_ref_raster.tif'
    input_raster_dir =  main_dir_path + 'tif/'
    outdir = main_dir_path +  'reprojected/'
 
    reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)
    print('Image reprojection completed!\n')
    
    def calculate_lulc_calss_percent(class_value):
        def percent_ofclass(data):
            mask = data == class_value
            pixel_count = np.count_nonzero(~np.isnan(data))
            return np.sum(mask) / pixel_count if pixel_count > 0 else np.nan
        return percent_ofclass
    
    lulc_classes = {
        22: 'developed_lowIntensity',
        23: 'developed_mediumIntensity',
        42: 'evergreenForest',
        71: 'grasslandOrHerbaceous',
        82: 'cultivatedCrops',
        95: 'emergentHerbaceousWetlands'
    }

    lulc_stats_dict = {
        f'pct_{name}': calculate_lulc_calss_percent(code)
        for code, name in lulc_classes.items()
    }
    
    grid_500m = gpd.read_file(grid_shapefile)
    csv_dir = main_dir_path + 'csv/'
    for file in glob(outdir + '*.tif'):
        variable = os.path.basename(file).replace('.tif', '')
        year = variable[-4:]
        print(f'Processing: {variable}')

        with rio.open(file) as raster_data:
            stats = zonal_stats(
                grid_500m,
                raster_data,
                stats=['majority', 'unique'],
                add_stats=lulc_stats_dict,
                geojson_out=False
            )

            statistics = pd.DataFrame(stats, index=grid_500m['gridid'])
            statistics.insert(loc =1,column ='year',value = year)
            statistics.to_csv(os.path.join(csv_dir, f'{variable}.csv'))
            
    years_list = range(2001,2016)
    var_names = ['Annual_NLCD_LndCov']
    concat_csv_files(main_dir_path, years_list, var_names)
    print('Yearly csv files are now merged into one csv file!\n')
      
def convert_recharge_data_from_text_to_raster(list_of_folder_names, start_year, end_year, point_shapefile, main_dir_path):
    
    """
    Description
    ===========
    This function convert raster in .txt format to .tif format
    Parameters:
        list_of_folder_names: A list contain names of folder for huc8 basins
        start_year: Start of the study period year
        end_year : End of the study perion  year
        point_shapefile : Shapefile containing spatial point to extract  pixel values
        main_dir_path: Main directory which contains folder 'text_files','text_files_yrly', 'tif','csv','csv_ref','csv_all_years'
        Note: 'text_files' folder should contain a folder for individual huc8 basins within the study area and each huc8 basin folder should contain a text file 'gwflow_flux_rech'
    Return:
        None
    """
    # Step 1 - filter yearly text files from a text file which contains array data for the period of study
    line_num_dic = defaultdict(list)
    line_num_dic =  {
     'Year' : [],
     'Basin': [],
     'Line number':[]
     
     }
    
    def find_word_line_number(filename, target_word):
    	line_number = 0

    	with open(filename, 'r') as file:
    		for line in file:
    			line_number += 1
    			if target_word in line:
    				return line_number
    	return None
    
    text_file_name = 'gwflow_flux_rech'
    
    text_dir = main_dir_path + 'text_files/'
    output_dir = main_dir_path + 'text_files_yrly/'
    for folder_name in list_of_folder_names:
        for year in range(2000, 2016):
            # print(folder_name,year)
            word_to_find = "Recharge for year (m3/day):        {}".format(year)
            # print(word_to_find)
            filename = text_dir + folder_name + '/' +  text_file_name
            line_number = find_word_line_number(filename,  word_to_find)
            line_number = find_word_line_number(filename, word_to_find)
            line_num_dic['Year'].append(year)
            line_num_dic['Basin'].append(folder_name)
            line_num_dic['Line number'].append(line_number)
            line_num_dic_df = pd.DataFrame(line_num_dic)


    line_num_dic_df = pd.DataFrame(line_num_dic)
    
    list_1  = []
    list_2 = []
    
    for folder_name in list_of_folder_names:
        name_list = line_num_dic_df[line_num_dic_df['Basin']==folder_name]
        list_1 = name_list['Line number'].tolist()
        print(len(name_list))
        list_2 =  list_1[1:]
        list_2.append(max(list_1 ) + (list_1 [1]-list_1[0]))
        list_2 =list_2[0:]
        print(list_1)
        print(list_2)

        for year, x, y in zip(range(2000, 2016),list_1, list_2):
            print(year, x, y)
        
            with open(text_dir  + folder_name + '/' +  text_file_name, 'r') as file, open( output_dir + folder_name + '_' + 'recharge_{}.txt'.format(year), 'w') as f_out:
                for line in islice(file,  x, y-1):
                    f_out.write(line)
    year_list = []

    for y in range(2000, 2016):
        year_list.append(y) 

    print(year_list)

    years_list = np.concatenate([([i]*6) for i in year_list], axis=0) 

    print(years_list)

    # Step two - Add geospatial information to yearly text files
    folder_names = ['1 Arkansas Headwaters', '2 Upper Arkansas','3 Fountain Creek','4 Chico Creek',
                    '5 Lake Meredith','6 Huerfano','7 Apishapa','8 Horse Creek','9 John Martin',
                    '10 Purgatoire','11 Big Sandy','12 Rush Creek','13 Two Butte']

    for folder_name in folder_names:
        for idx in range(len(years_list)):
            file = open(output_dir + folder_name + \
                 '_' + 'recharge_{}.txt'.format(years_list[idx]), 'r+')
            print(file)
            if folder_name =='1 Arkansas Headwaters':
                add_lines = ["ncols 219\n",
                               "nrow 359\n",
                               "xllcorner -679873.860172\n",
                               "yllcorner 4266565.449588\n",
                               "cellsize 500\n",
                               "nodata_value 0\n"]
                file.writelines(add_lines)
                file.close() 
                
            elif folder_name == '2 Upper Arkansas':
                add_lines = ["ncols 179\n",
                               "nrow 240\n",
                               "xllcorner -591582.908683\n",
                               "yllcorner 4258279.582847\n",
                               "cellsize 500\n",
                               "nodata_value 0\n"]
                file.writelines(add_lines)
                file.close() 
            elif folder_name == '3 Fountain Creek':
               add_lines = ["ncols 96\n",
                               "nrow 206\n",
                               "xllcorner -553094.067556\n",
                               "yllcorner 4298047.682848\n",
                               "cellsize 500\n",
                               "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close() 
               
            elif  folder_name == '4 Chico Creek':
              add_lines = ["ncols 74\n",
                           "nrow 191\n",
                           "xllcorner -511676.991859\n",
                           "yllcorner 4294297.310484\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
              file.writelines(add_lines)
              file.close() 
              
            elif  folder_name =='5 Lake Meredith':
                add_lines = ["ncols 191\n",
                           "nrow 294\n",
                           "xllcorner -501337.222736\n",
                           "yllcorner 4205434.903315\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
                file.writelines(add_lines)
                file.close() 
                
            elif  folder_name ==  '6 Huerfano':
               add_lines = ["ncols 241\n",
                           "nrow 190\n",
                           "xllcorner -605755.967095\n",
                           "yllcorner 4197039.197529\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close() 
               
            elif  folder_name == '7 Apishapa':
               add_lines = ["ncols 216\n",
                           "nrow 176\n",
                           "xllcorner -567900.523575\n",
                           "yllcorner 4189479.541429\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close() 
               
            elif  folder_name == '8 Horse Creek':
               add_lines = ["ncols 151\n",
                           "nrow 241\n",
                           "xllcorner -482121.875712\n",
                           "yllcorner 4262622.440884\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close()
               
            elif  folder_name == '9 John Martin':
               add_lines = ["ncols 280\n",
                           "nrow 311\n",
                           "xllcorner -429122.562976\n",
                           "yllcorner 4196667.325832\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close()  
               
            elif  folder_name == '10 Purgatoire':
               add_lines = ["ncols 390\n",
                           "nrow 243\n",
                           "xllcorner -587661.403606\n",
                           "yllcorner 4141492.033396\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close()            
               
            elif  folder_name == '11 Big Sandy':
               add_lines = ["ncols 360\n",
                           "nrow 307\n",
                           "xllcorner -492721.518975\n",
                           "yllcorner 4260961.995870\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close()            

            elif  folder_name == '12 Rush Creek':
               add_lines = ["ncols 272\n",
                           "nrow 214\n",
                           "xllcorner -467277.735584\n",
                           "yllcorner 4289774.477125\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close() 

            elif  folder_name == '13 Two Butte':
               add_lines = ["ncols 243\n",
                           "nrow 154\n",
                           "xllcorner -422566.746218\n",
                           "yllcorner 4173293.798080\n",
                           "cellsize 500\n",
                           "nodata_value 0\n"]
               file.writelines(add_lines)
               file.close() 
                         
            else:
               file.close()
               
    # Final step convert yearly text files to tif files
    raster_output_dir = main_dir_path + 'tif_yrly_files/'  
      
    for folder_name in list_of_folder_names: 
        header_rows = 7
        row_ite = 1 
        header ={} 
        for year in range(start_year,end_year):
            swat_file = text_dir + folder_name + '_recharge_' + str(year) + '.txt'
            swat_tif =  raster_output_dir + folder_name + '_recharge_' + str(year) + '.tif'
            with open(swat_file, 'rt') as file_h:
                for line in file_h:
                    if row_ite <= header_rows:
                        line = [c for c in line.split("\n")[0].split(" ") if c != '']
                        header[line[0]] = float(line[-1])
                    else:
                        break
                    row_ite = row_ite + 1
                   
                swat_data = np.abs(np.loadtxt(swat_file, skiprows=header_rows)) * 1
                left = header['xllcorner']
                top = header['yllcorner'] + header['nrow'] * header['cellsize']
                cellsize = header['cellsize']
                affine = rio.Affine(cellsize, 0, left, 0, -cellsize, top)
                nodata = header['nodata_value']
                
                with rio.open(
                    swat_tif,
                    'w',
                    driver='GTiff',
                    height=swat_data.shape[0],
                    width=swat_data.shape[1],
                    crs='epsg:32615',
                    dtype=swat_data.dtype,
                    transform=affine,
                    count=1,
                    nodata=nodata
                ) as dst:
                    dst.write(swat_data, 1)    
                    
    input_raster_dir =  main_dir_path +  'tif/'
    
    output_path = main_dir_path +  'csv/'
    extract_rasters_values_at_points(input_raster_dir, point_shapefile, output_path, pattern='*.tif')     
    print('Images pixel values are now extracted and saved to csv files!\n')
    
    input_csv_dir =  output_path
    output_dir = main_dir_path+ 'csv_all_years/'
    var_names =  ['recharge']   
    duplicate_files(input_csv_dir, output_dir, var_names, pattern='.csv' )
    
    years_list = range(2001,2016)
    
    concat_csv_files(main_dir_path, years_list, var_names)
    print('Yearly csv files are now merged into one csv file!\n')
    
    
    
    
    
    
    
