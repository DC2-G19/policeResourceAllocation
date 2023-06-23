# Police Resource Allocation
In this project we should develop an automated, data-driven police demand forecasting system to aid decision-makers in allocating police resources for residential burglary in the London borough of Barnet. The goal is to reduce burglary by accurately predicting when and where burglaries are likely to occur. The team will use historical crime data and consider three main approaches to tackling burglary: catching suspects in the act, detective work, and prevention. The borough of Barnet is split into 24 wards, and the team can assume that 100 police officers are available to patrol between 0800 and 2200, with 200 hours per day dedicated to burglary prevention. The team can also plan special operations with more officers once every four months. The project must consider ethical implications and what works in crime reduction approaches.

# File Description
## DATABASE:
data_cleaner.py : Takes the original crime files, preprocesses it and forms an SQL db
dbMaker.py : appends auxiliary datasets
## EDA:
main_crime_data_DAE.ipynb

## BURGLARY MODELING:
XGBoost_Model.ipynb : Predictive model for burglaries
XGBoost_experimental.ipynb : draft forecasting of burglaries
## Unemployment Modeling
auto_arima.ipynb
unemployment_forecasting.ipynb : Making of unemployment models + Performance metrics
predicting_unemployment.py : actual forecasting of 2019 for VIS purposes

## VISUALIZATION
### Unemployment:
unemployment_visualization_2019.ipynb
sidebysideGif.ipynb
## INTERACTIVE MAP
xgboost_with_maps.py
