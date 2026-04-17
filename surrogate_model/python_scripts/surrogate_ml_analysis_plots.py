import copy
import matplotlib.pyplot as plt
import shap
import skexplain
import matplotlib as mpl
import matplotlib.colors as mcolors 
import pandas as pd
import numpy as np

def SWAT_surrogate_predicted_plots(swat_ml_predicted_df, ml_plots):
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
    plt.savefig((ml_plots +  'surrogate_model_simulated_predicted_recharge.png'), dpi=600)

def surrogate_shap_ale_plot(surrogate_model, x_train_data, y_train_data, ml_plots):
    '''
    Parameters
    ----------
    surrogate_model :  Trained surrogate model model path
    x_train_data : DataFrame containing train predictor variables 
    y_train_data : DataFrame containing train target variable 
    plots_path :  Director path for where the shap and ale plots are saved
    Returns
    -------
    None.

    '''
    
    explainer = shap.TreeExplainer(surrogate_model)

    X_test_data = x_train_data.sample(n=5000, random_state=42)

    shap_values = explainer.shap_values(X_test_data)
    fig = plt.figure(figsize=(22, 20),linewidth=4, edgecolor="black") 
    gs = fig.add_gridspec(4, 2)

    left_ax = fig.add_subplot(gs[:,0]) 


    right_0 = fig.add_subplot(gs[0, 1])
    right_1 = fig.add_subplot(gs[1, 1])
    right_2 = fig.add_subplot(gs[2, 1])
    right_3 = fig.add_subplot(gs[3, 1])

    axes_flat = [right_0, right_1, right_2, right_3]
    my_features = ['precip [mm]', 'sw_ave [mm]', 'et [mm]','snomlt [mm]'] 
    plt.sca(left_ax) 
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test_data, 
            feature_names=X_test_data.columns.tolist()
        ),
        max_display=27,
        show=False,
        plot_size=None
    )

    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20) 
    ax.xaxis.label.set_size(20)
    ax.tick_params(axis='x', labelsize=20)

    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("Feature value", fontsize=20,fontweight='bold', labelpad=-42)
    cbar_ax.tick_params(labelsize=20,width=4)
    left_ax.set_xlabel("SHAP value (Impact on Groundwater Recharge (mm/year))", fontsize=20,fontweight='bold')
    left_ax.set_title("Global Feature Importance (SHAP)", fontsize=20,fontweight='bold')
    left_ax.tick_params(axis='both', which='major', labelsize=20)
    print("Calculating ALE values...")
    ale_explainer = skexplain.ExplainToolkit(
        estimators=('RandomForest', surrogate_model), 
        X=x_train_data, 
        y=y_train_data
    )

    ale_data = ale_explainer.ale(features=my_features, subsample=50000)
    name_map = {
        'precip [mm]' : 'Precipitation (mm)',
        'sw_ave [mm]': 'Average soil water content (mm)',
        'et [mm]':'Actual evapotranspiration (mm)' ,
        'snomlt [mm]':'Snowmelt (mm)'
    }

    for i, feature in enumerate(my_features):
        ale_explainer.plot_ale(
            ale=ale_data,
            features=[feature],  
            ax=axes_flat[i],
            hist_kws={
                       'color': 'blue',
                      'alpha': 0.3}
        )
       
        clean_name = name_map.get(feature, feature)
        axes_flat[i].set_xlabel(clean_name, fontweight='bold', fontsize=20)
        axes_flat[i].tick_params(axis='both', which='major', labelsize=20)
        
        main_ax = axes_flat[i]
        twin_ax = None
        
        for other_ax in main_ax.figure.axes:
            if other_ax is not main_ax and other_ax.bbox.bounds == main_ax.bbox.bounds:
                twin_ax = other_ax
                break
        if twin_ax:
            twin_ax.tick_params(axis='y', labelsize=20)
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
        fontsize=20, 
        fontweight='bold'
    )


    fig.text(
        1.00, 0.5,          
        "Frequency (Log-Scale)", 
        va='center', 
        rotation=-270, 
        fontsize=20, 
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(ml_plots +   'surrogate_model_shap_ale_swat_surrogatel_plot.pdf', bbox_inches='tight')
    
def surrogate_model_predictor_ale_plots(standalone_model, x_train_data, y_train_data, ml_plots):
    '''
    Parameters
    ----------
    surrogate_model:  Trained surrogate_model 
    x_train_data : DataFrame containing train predictor variables 
    y_train_data : DataFrame containing  train target variable 
    plots_path :  Director path for where the shap and ale plots are saved
    Returns
    -------
    None.

    '''
    swat_var_names_group_short = {
         'group_1':['surq_gen [mm]','latq [mm]', 'wateryld [mm]','ecanopy [mm]', 'eplant [mm]', 'esoil [mm]', 'surq_cont [mm]', 'cn' ],
         'group_2':['sw_init [mm]', 'sw_final [mm]','sw_300 [mm]', 'sno_init [mm]', 'sno_final [mm]', 'snopack [mm]', 'pet [mm]', 'irr [mm]']
         }

    swat_var_names_group_long = {
         'group_1':['Generated surface runoff (mm)','Lateral flow (mm)','Water yield (mm)','Canopy evaporation (mm)','Plant transpiration (mm)','Soil evaporation (mm)', 'Contributing surface runoff (mm)','Curve Number' ],
         'group_2':['Initial soil water content (mm)','Final soil water content (mm)','Top 300 mm of soil (Ave.Soil.Wat) (mm)','Initial snow water content (mm)', 'Final snow water content (mm)','Average snow water content (mm)',
                    'Potential Evapotranspiration (mm)','Irrigation (mm)']
         }


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
                fontsize=24, 
                fontweight='bold'
            )


            fig.text(
                1.00, 0.5,          
                "Frequency (Log-Scale)", 
                va='center', 
                rotation=-270, 
                fontsize=24, 
                fontweight='bold'
            )
            plt.tight_layout()
            plt.savefig(ml_plots +   f'surrogate_model_shap_ale_{ my_features}_plot.png', dpi=600, bbox_inches='tight')