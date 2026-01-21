import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import skexplain
import shap
import joblib


def create_shape_ale_plots(model_path, train_x_df_path, train_y_df_path, test_x_df_path, plots_dir):
    """
    This function create Accumulative Local Effects (ALE) and SHapley Additive exPlanations (SHAP )
    Parameters
    ----------
    model : Trained ml model
    train_x_df_path : Data frame path containing predictor variables used to train ml model
    train_y_df_path : Data frame path containing target variable used to train ml model
    test_x_df _path: Data frame path containing predictor variables used to test ml model
    plots_dir : Directory path to save ALE and SHAP plots
    Returns
    -------
        None.

    """
    rf_model = joblib.load(model_path)
    x_train_df = pd.read_csv(train_x_df_path)
    y_train_df = pd.read_csv(train_y_df_path)
    x_test_df = pd.read_csv(test_x_df_path)

   
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14

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

    X_test_data= x_test_df.rename(columns=renaming_dic)
    
    explainer = shap.TreeExplainer(rf_model)

    X_test_data = x_test_df.sample(n=500, random_state=42)

    shap_values = explainer.shap_values(X_test_data)

    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test_data, 
            feature_names=X_test_data.columns.tolist()
        ),
        max_display=len(X_test_data.columns),
        show=False,
        plot_size=None
    )


    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("Feature value", fontsize=16,fontweight='bold', labelpad=-40)
    cbar_ax.tick_params(labelsize=18,width=4)
    left_ax.set_xlabel("SHAP value (Impact on Groundwater Recharge (mm))", fontsize=14,fontweight='bold')
    left_ax.set_title("Global Feature Importance (SHAP)", fontsize=16,fontweight='bold')
    left_ax.tick_params(axis='both', which='major', labelsize=16)
    print("Calculating ALE values...")
    ale_explainer = skexplain.ExplainToolkit(
        estimators=('RandomForest', rf_model), 
        X=x_train_df, 
        y=y_train_df
    )

    ale_data = ale_explainer.ale(features=my_features, subsample=50000)
    name_map = {
        'Precip': 'Precipitation',
        'slope': 'Slope',
        'sandPcnt': 'Percent Sand',
        'cultivatedCrops': 'Cultivated Crops'
    }

    for i, feature in enumerate(my_features):
        ale_explainer.plot_ale(
            ale=ale_data,
            features=[feature],  
            ax=axes_flat[i],
            hist_kws={
                       'color': 'teal',
                      'alpha': 0.3}
        )
       
        clean_name = name_map.get(feature, feature)
        axes_flat[i].set_xlabel(clean_name, fontweight='bold', fontsize=14)
        axes_flat[i].tick_params(axis='both', which='major', labelsize=14)
        
        main_ax = axes_flat[i]
        twin_ax = None
        
        for other_ax in main_ax.figure.axes:
            if other_ax is not main_ax and other_ax.bbox.bounds == main_ax.bbox.bounds:
                twin_ax = other_ax
                break
        if twin_ax:
            twin_ax.tick_params(axis='y', labelsize=14)
            # twin_ax.set_yscale('linear')

    all_axes = [left_ax, right_0, right_1, right_2, right_3]

    for ax in all_axes:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(True)    
            ax.spines[side].set_linewidth(2)      
            ax.spines[side].set_color('black')  
            ax.spines[side].set_linestyle('-')
          
    fig.supylabel(
        "ALE Effect (mm)", 
        x=0.59,             
        y=0.5,              
        fontsize=16, 
        fontweight='bold'
    )


    fig.text(
        1.00, 0.5,          
        "Frequency (Log-Scale)", 
        va='center', 
        rotation=-270, 
        fontsize=16, 
        fontweight='bold'
    )
    plt.tight_layout()
    
    plt.savefig((plots_dir +   'shap_ale_open_data_model_plot.png'), dpi=600)
