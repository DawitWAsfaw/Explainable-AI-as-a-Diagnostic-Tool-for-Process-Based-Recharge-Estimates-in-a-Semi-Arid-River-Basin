# This read me text provides detailed step-by-step guide how the machine learning pipeline was constructed for the article titled 
"Analysis of drivers of deep percolation in an alluvial aquifer system supporting irrigated agriculture using machine learning and SWAT+"

File structure and data contained in Folders

The recharge data estimated using SWAT+ and gwflow which is used as target for this study. 
The data is available in two formats. These formats are cvs and text array file.The csv file is at hru polygon scale and has yearly temporal resolution. 
The "percolation" column name in the csv file is the deep percolation estimates. 
The recharge (deep percolation) is found  under the hru_scale folder in a csv file named
"hru_scale_SWAT_gwflow_input_output_data_2000_2015.csv". This file contains all the input and output data from SWAT+ and gwflow simulation results. 
 
The  text array data has 500 meter spatial and yearly temporal resolution. 
To convert the text data to .tif format 

File management
step 1: Create main and subdirectory to store downloaded data, preprocessing and machine learning analysis

Convert recharge data from text file to raster.tif format
step one: Extract the yearly data using 'change_textRecharge_files_to_yrly_data.py' file
step two: Convert yearly data from step on to .tif file format using 'recharge_from_textTo_raster.py' file

Download climatic data from Google Earth Engine -using "gee_download.py". The shapefiles in "gis->gee_download" 


 1. Evapotranspiration - openET
 2. Precipitation  - PRISM
 3. Elevation - PRISM
 4. Minimum Air Temperature - PRISM
 5. Maximum Air Temperature - PRISM

Create hydrological factors maps using ArcGIS Pro using "Elevation" data downloaded from gee (1 -3) and larb_rivers.shp (4 -5)
 1. Topographic wetness index (TWI),
 2. Topographic roughness index (TRI, 
 3. Steam Power Index(SPI), 
 4. Drainage Density and 
 5. Distance to River  

Download physical soil properties data from SSURGO Portal under "Soil Physical Properties" (1 -7) and "Water Features" (8 -9) Menus
 1. Available Water Capacity
 2. Available Water Storage 
 3. Percent Clay 
 4. Percent Sand 
 5. Percent Silt 
 6. Saturated Hydraulic Conductivity (Ksat)
 7. Surface Texture
 8. Flooding Frequency Class
 9. Ponding Frequency Class

Download Annual Land Use Land Cover Data from Multi-Resolution Land Characteristics Consortium - https://www.mrlc.gov/viewer/ Accessed 11/7/2025 at 6:27pm)
step one: Go to the left pane and select "Continental U.S" under Contents Menu and " Land Cover" under "Annual NLCD" submenu options
step two: Click the "Open Data Downloader Tool" - down arrow symbol and go " Tool" menu on the right pane, under "Data Download", "Method:" select "Shapefile" and click on "Choose File" icon
browse to the shapefile folder under "GIS->shapefiles->lulc_download->EPSG3857.zip" and under options  "GeoTIFF". Once your shapefile is uploaded and is visible on the CONUS map , 
click with in the boundary of the uploaded shapefile and it should show as selected. Under the "Select Categories:" options, select "Land Cover" and under the "Select Years" slider, select year from 2000 - 2015.
Provide your email address, click on "Download" icon, and a link will be emailed from no-reply@usgs.gov with a subject "MRLC Product Download Is Ready" to the email address you provided and  click on the the link provide to download LULC data. 
click on the link and unzip the files and save it in "soil_data" folder. 


Analyze Annual LULC Percentage per HUC12 polygon scale
step one: Reproject the LULC .tiff files and save the reproject file under "500m_scale/tiff/" folder.
          The reference raster file is provided in "GIS/ref_raster_files/lulc_raster_ref.tif" folder 
		  Use 'reproject_rasters.py' to reproject the LULC files. 
1. Use the python code "Annual_LULC_zonalStat_huc12.py' file to calculate the percentage of area
 coverage for the different land cover classes within huc12 basin boundaries. 
2. The huc12 polygon shapefile is provide under "GIS->shapefiles->huc12_scale->huc12_basins.shp"
3. Save the final .csv file in to the "huc12_scale -> yrly_csv" folder
Analyze Annual LULC Percentage per at 500m grid scale  
1. Use the python code "Annual_LULC_zonalStat_500m_grid.py' file to calculate the percentage of area
 coverage for the different land cover classes within 500m by 500m grid cell. 
2. The 500m grid shapefile is provide under "GIS->shapefiles->500m_grid_scale->500m_grid.shp"
3. Save the final .csv file in to the "500m_scale -> yrly_csv" folder
Analyze Annual LULC Percentage per at 500m grid scale.   
 
You will need ArcGIS Pro and Python IDE to download and preprocess physical soil properties and Water Features data
step one: Go to https://www.nrcs.usda.gov/sites/default/files/2023-09/SSURGO-Bulk-Downloader-ArcGIS-Pro-Installation-and-User-Guide.pdf (Accessed 11/7/2025 at 4:36pm)
			and downloaded "SSURGO Bulk Downloader" and use the instruction in the link to add the SSURGO Bulk Downloader Toolbox to ArcGIS Pro Toolbox. 
step two: To find the area symbol for the soil data boundary go to "https://casoilresource.lawr.ucdavis.edu/gmap/" (Accessed 11/7/2025 at 4:39pm) and 
			identify the codes to add to the SSURGO Bulk Downloader Toolbox Search by "Areasymon" from dropdown menu. 
step three: Go to https://www.nrcs.usda.gov/sites/default/files/2024-11/SSURGO-Portal-Quick-Start-Guide.pdf  (Accessed 11/7/2025 at 4:42pm)
			and follow the instruction to install SSURGO Portal. 
			step three: Create a database and import SSURGO data you downloaded using SSURGO Bulk Downloader (takes upt0 30 minutes to import data to SSURGO Portal. 
			Under the Soil Data Viewer Menu on the left side- select "Soil Physical Properties",  and select 'Available Water Capacity',
			' Available Water Storage', 'Percent Clay', 'Percent Sand', ' Percent Silt', 'Saturated Hydraulic Conductivity (Ksat)', 
			'Surface Texture' and use "All Layers (Weighted Average)" and "Weighted Average" under Rating options. Next, select 'Water Features" options and select
			'Flooding Frequency Class ' and 'Ponding Frequency Class' separately. Select "Dominant Component" and "More Frequent" under " Tie Break Rule:" options
			include "January" as Beginning Month and " December" as Ending month. 

step four: Go to ArcGIS Pro and add the database (created using SSURGO Portal), makes maps of physical soil properties and water features individual and 
export the maps to 'physical_properties_name.tif" format

Save the downloaded data under " grid_500m" folder under subfolders corresponding to each variables
 1. "climate_data" 
 2. "soil_data" and 
 3. "hydrological_factors_data"
 4. "lulc_data'

Reproject these data using 'reproject_rasters.py' and save the reprojected files in 'reproject' folder
The reference raster file is provided in "GIS/ref_raster_files/raster_ref.tif" folder 
save all the reprojected files under "500m_scale/tiff/" folder

Calculate zonal statistics - Majority, Unique classes and Percentage of area coverage for individual lulc classes
huc12_scale:
1. Use the python code "zonalStat_huc12.py' file to calculate the percentage of area
 coverage for the different land cover classes within huc12 basin boundaries. 
2. The huc12 polygon shapefile is provide under "GIS->shapefiles->huc12_basins->huc12_basins.shp"
3. Save the final .csv file in to the "huc12_scale -> yrly_csv" folder
500m grid scale  
1. Use the python code "extract_raster_value.py'. 
2. The 500m point shapefile is provide under "GIS->shapefiles->500m_grid->500m_grid_pnt_xy.shp"
3. Save the final .csv file in to the "500m_scale -> yrly_csv" folder


Combine yearly csv files into a single csv file for all the variables 
(For intance, precipitation_2000.csv - precipitation_2015.csv to precipitation_all.csv)
Use "concat_csv_files.py" to combine year csv files

Finally,
For huc12 scale, merge on the variable_all.csv file into f huc12_scale_ml_input.csv
Use "merge_csv.py" file to merge csv files

for 500m_scale, import variable_all.csv files to your ml pipeline directly due to the large data size. 


...........................*End*............................................................................