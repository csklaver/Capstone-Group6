## Code Overview
To download all data from [NHANES website](https://wwwn.cdc.gov/nchs/nhanes/) the python scripts in this [NHANES-Downloader](https://github.com/mrwyattii/NHANES-Downloader) were ran.
### Feature importing folder
Includes all .ipynb files for importing features from each year, and an .ipynb file to pre-process and export the merged final dataframe ('data_raw_ext.cvs')
### Missing Value Imputation
Statistical_ML_Imputations.ipynb imputes missing values using mean, median, progressive MLP and progressive KNN and exports imputed dataframes. Should be ran prior to the feature engineering and DNN below.
### Feature Engineering & Deep Neural Networks
- Feature_engineering_models-KNN.ipynb
- Feature_engineering_models-MLP.ipynb
- Feature_engineering_models-MEAN.ipynb
- Feature_engineering_models-MEDIAN.ipynb

These notebooks contain the deep artificial neural network and convolutional neural network with each feature engineering method for each imputation type.

#### Extra analysis
- nan_helper.py - helper function to get proportion of missing values in each column of dataframe
- Preliminary_ML_Analysis.ipynb - runs Logistic Regression, Naive Bayes, DTs, RF, XGBoost, Perceptron, and MLP on original data each imputation type
- keras_resampling.py - testing outcomes of Random Oversampling, Random Undersampling, and SMOTE on original data each imputation type
- Verifying_DANN_CNN.ipynb - testing deep neural networks on different target and UCI breast cancer data
- Feature_Corr_Selection.ipynb - looking at distributions and correlations of features

#### Data
Original data with fewer columns is 'df_raw_v2.csv' within the data folder. Data with extended number of features that is used for the bulk of this project is titled 'df_raw_ext.csv'.

