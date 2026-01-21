# Explainable AI as a Diagnostic Tool for Process-Based Recharge Estimates in a Semi-Arid River Basin
## Graphical Abstract
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%201.%20Graphical%20Abstract_large.png)
##  Abstract
Groundwater recharge is a vital component of the groundwater system and is challenging to measure due to its complex dynamics. There are multiple process-based tools utilized to estimate recharge which has significant uncertainty. The analysis of the recharge estimates from these tools and the interactions of the input variables with recharge is time consuming and computationally expensive. In this study, we propose a machine learning (ML) predictor model and explainable AI (SHAP and ALE) as a diagnostic tool to understand the drivers of groundwater recharge, capture critical threshold values, and identify high recharge zones. The model is trained with a calibrated groundwater recharge obtained from SWAT+ model as target and open-source data which includes climate, soil physical properties, hydrogeological, topographic, and land use land cover as predictors. The data has annual temporal and 500m spatial resolution for the periods of 2000 – 2015. The study is implemented in the Lower Arkansas river basin in Colorado, USA which is characterized by semi-arid climate and alluvial aquifer supporting irrigated agriculture. The ML model demonstrated high predictive performance with R2, RMSE, MAE values of 0.91, 11.14 mm/year, and 2.56 mm/year respectively for test data. Explainable AI assessments show that groundwater recharge occurs primary when precipitation exceeds 500 mm/year. Furthermore, regions with slope values ≥ 12 degrees, sand percent values ≥ 65%, and cultivated land fractions ≥ 0.70 are identified as high recharge zones. The methods adopted in this study successfully extracted meaningful thresholds hidden within the complex outputs of the process-based model. Consequently, this approach serves as a valuable diagnostic too that yields insights essential for effective groundwater management and planning in river basins with extensive irrigation. 
Key words; machine learning, process-based models, explainable AI, groundwater recharge, open-source data
##  Data processing
Data approcessing guidelines are presented in [Data Preprocessing](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/README_data_collection%20and%20preprocessing.txt).
##  Model environment set up and steps description
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in [Dependencies](https://github.com/DawitWAsfaw/groundwater_recharge_estimates_using_ML/blob/main/deep_percolation_ml.yml) file and can be installed on local computer by copying the code snippet provided below.
```
conda env create -f  deep_percolation_ml.yml
```
##  Method workflow
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%203.%20Methods%20Flow%20Chart.png)
##  Spatial prediction
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%207.%20spatial%20prediction.png)

##  High recharge zones
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%208.%20High%20Recharge%20Zones.png)
## Affiliations
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/affiliations.png)
