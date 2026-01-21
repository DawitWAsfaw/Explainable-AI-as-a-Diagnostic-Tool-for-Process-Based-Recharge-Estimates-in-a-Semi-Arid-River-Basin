import os

def create_directories(main_dir_path):
    """
    Creates directories for saving data processing and machine learning analsis results
    Parameters
    ----------
    main_dir_path : Directory path under which sub-folders will be created 
    Returns
    -------
    None.

    """
    
    dir_dict = {
        'main_folders': ['raw_data','ml_input','ml_analysis_results'],
        'var_group_names':[['climate'],['lulc'],['soil_phy_prop', 'hydrological_factors','hydrogeological'],['recharge']],
        'climate_sub_folder_names':['tif','reprojected', 'csv','csv_ref','zipped','csv_all_years'],
        'lulc_sub_folder_names':['tif','reprojected', 'csv','csv_ref','zipped','csv_all_years'],
        'recharge_sub_folder_names':['tif','text_file','text_yrly_files', 'csv','csv_ref','csv_all_years'],
        'static_var_sub_folder_names':['tif','reprojected', 'csv','csv_ref','csv_all_years'],
        'ml_analysis_results': ['model','plots','predictions','error_metrics']
        
        }
            
    for var_name in dir_dict['main_folders']:
        os.makedirs(main_dir_path + var_name + '/', exist_ok=True)
        if var_name == 'raw_data':
            for var_group_name in dir_dict['var_group_names']:
                for name in var_group_name:
                    os.makedirs(main_dir_path + var_name + '/' + name + '/', exist_ok=True)
                    
                    if name  == 'climate':
                        for sub_folder in dir_dict['climate_sub_folder_names']:
                            
                            os.makedirs(main_dir_path + var_name + '/' + name + '/' + sub_folder + '/', exist_ok=True)
                    elif name == 'lulc':
                        for sub_folder in dir_dict['lulc_sub_folder_names']:
                            os.makedirs(main_dir_path + var_name + '/' + name + '/' + sub_folder + '/', exist_ok=True)
                         
                    elif name == 'recharge':
                        for sub_folder in dir_dict['recharge_sub_folder_names']:
                            os.makedirs(main_dir_path + var_name + '/' + name + '/' + sub_folder + '/', exist_ok=True)
                        
                    else:
                        for sub_folder in dir_dict['static_var_sub_folder_names']:
                            os.makedirs(main_dir_path + var_name + '/' + name + '/' + sub_folder + '/', exist_ok=True)
        
            

        elif var_name == 'ml_input':
            os.makedirs(main_dir_path + var_name + '/csv_all_years/', exist_ok=True)
            
        else:
            for sub_folder in dir_dict['ml_analysis_results']:
                os.makedirs(main_dir_path + var_name + '/' + sub_folder  +'/', exist_ok=True)
                if sub_folder == 'predictions':
                    pred_types = ['spatial/','csv/','temporal/']
                    for pred_type in pred_types:
                        os.makedirs(main_dir_path + var_name + '/' + sub_folder  +'/' + pred_type, exist_ok=True)
                        for file_type in ['csv/','tif/']:
                            os.makedirs(main_dir_path + var_name + '/' + sub_folder  +'/' + pred_type  + file_type , exist_ok=True)
                        
                elif sub_folder == 'plots':
                    plot_types = ['eda/', 'ml_plots/']
                    for plot_type in plot_types:
                        os.makedirs(main_dir_path + var_name + '/' + sub_folder  +'/' + plot_type, exist_ok=True)
                        
main_dir_path = 'D:/deep_percolotation_ml_project/'
create_directories(main_dir_path)