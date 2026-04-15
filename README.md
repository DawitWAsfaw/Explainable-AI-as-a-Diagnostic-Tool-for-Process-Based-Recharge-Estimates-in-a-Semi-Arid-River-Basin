# Explainable AI as a Diagnostic Tool for Analyzing Spatiotemporal Variability in Simulated Groundwater Recharge: Application to a Semi-Arid River Basin
## Graphical abstract
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Graphical%20Abstract.png)
##  Abstract
Groundwater recharge is a vital component for water budget and water availability assessments but is challenging to measure due to its complex dynamics and subsurface occurrence. Process-based modeling tools are often used to simulate recharge. Analyzing the numerous, interacting factors that control recharge is time consuming and computationally expensive. In this study, we develop machine learning (ML) models that predict recharge and use explainable AI (XAI) as a diagnostic tool to understand the drivers of groundwater recharge, capture critical threshold values, and identify high recharge zones. ML models are trained and tested using calibrated SWAT+ recharge values as the response variable. Two ML models are developed: (i) a surrogate model that captures the physics from the SWAT+ simulator by utilizing selected model inputs with a directly comparable definition of distributed parameter values; and (ii) a stand-alone predictor that utilizes commonly available open-source data to predict recharge and has annual temporal and 500 m spatial resolution for the period of 2002 – 2015. The study is implemented in the semi-arid Lower Arkansas river basin in Colorado, USA which contains an alluvial aquifer that is used for agricultural irrigation. Both the surrogate and stand-alone models demonstrated high predictive accuracy with NSE values of 0.98 and 0.91, respectively, when evaluated against test data. Explainable AI assessments show that average diffuse groundwater recharge increases significantly primarily when precipitation exceeds approximately 500 mm/year. Irrigation return flow is also identified as a major source of recharge where cultivated land cover is dominant. Unlike the surrogate model, which correctly identifies key drivers of recharge (e.g., precipitation, snow melt), XAI results from the stand-alone model identify some predictor variables that are correlated with recharge (e.g., high slope and certain LULC classes) but don’t physically drive recharge. These results highlight the effectiveness of ML surrogate models to meaningfully analyze physics-based simulation results. Stand-alone ML models may have high predictive accuracy but limited explanatory power. The methods adopted in this study provide a valuable approach for investigating recharge dynamics in unconfined aquifers. 
**Key words**; machine learning, process-based models, explainable AI, groundwater recharge, open-source data
## Sugggested citation
Dawit Asfaw, Ryan Smith, Micheal Ronayne, Sayantan Majumdar, Salam A. Abbas, Ryan T. Bailey,(2026).Explainable AI as a Diagnostic Tool for Analyzing Spatiotemporal Variability in Simulated Groundwater Recharge: Application to a Semi-Arid River Basin. (In Prep for journal submission). 

**Corresponding author** (dawit.asfaw@colostate.edu)
##  Data processing
Data approcessing guidelines are presented in [Data Preprocessing](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/data_collection%20and%20preprocessing.txt).
##  Model environment set up 
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in [Dependencies](https://github.com/DawitWAsfaw/groundwater_recharge_estimates_using_ML/blob/main/deep_percolation_ml.yml) file and can be installed on local computer by copying the code snippet provided below.
```
conda env create -f  deep_percolation_ml.yml
```
##  Method workflow
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%202.png)
##  Spatial prediction
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%206.png)

##  High recharge zones
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/Figure%208.png)
## Affiliations
![image](https://github.com/DawitWAsfaw/Explainable-AI-as-a-Diagnostic-Tool-for-Process-Based-Recharge-Estimates-in-a-Semi-Arid-River-Basin/blob/main/plots/affiliations.png)
