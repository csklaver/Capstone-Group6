## Code Overview
To download all data from [NHANES website](https://wwwn.cdc.gov/nchs/nhanes/) the python scripts in this [NHANES-Downloader](https://github.com/mrwyattii/NHANES-Downloader) were ran.
### Feature importing folder
Includes all .ipynb files for importing features from each year, and an .ipynb file to pre-process and export the merged final dataframe ('data_raw_ext.cvs')

- Feature_Importing folder - .ipynb files for importing features for each year, file to pre-process and export merged, final dataframe
- nan_helper.py - helper function to get proportion of missing values in each column of dataframe
- Statistical_ML_Imputations.ipynb - implements both mean/median and progressive MLP/KNN imputation strategies, exports imputed dataframes
- Preliminary_ML_Analysis.ipynb - runs Logistic Regression, Naive Bayes, DTs, RF, XGBoost, Perceptron, and MLP on original data each imputation type
- keras_resampling.py - testing outcomes of Random Oversampling, Random Undersampling, and SMOTE on original data each imputation type
- Verifying_DANN_CNN.ipynb - testing deep neural networks on different target and UCI breast cancer data
- Feature_Corr_Selection.ipynb - looking at distributions and correlations of features (not neccessary to run)
- Feature_engineering_models.ipynb - Implementation of all feature engineering and deep neural networks for each imputation type

