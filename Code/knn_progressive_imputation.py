# %%%%%% Capstone %%%%%%
# %%%%%% Caroline Sklaver %%%%%%
# %%%%%% Date: June - 23 - 2020 %%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%% Imputing with KNN %%%%%%
#%%-----------------------------------------------------------------------

#%%-----------------------------------------------------------------------
# Importing the required packages

import numpy as np
import pandas as pd


path = ('/Users/carolinesklaver/Desktop/Capstone/NHANES/data/csv_data/')

import os
os.chdir(path)

#%%-----------------------------------------------------------------------
# Importing the data

df_raw = pd.read_csv('df_raw_v2.csv')

# bring year and depressed to the beginning of df
year = df_raw.pop('year')
df_raw.insert(1, 'year', year)

dep = df_raw.pop('depressed')
df_raw.insert(2, 'depressed', dep)

# drop marijuana use
df_raw.drop(['used_marijuana'],axis=1, inplace=True)



#%%-----------------------------------------------------------------------
# Function to identify missing values
from nan_helper import nan_helper


# call the function
nan_df = nan_helper(df_raw)
print(nan_df.head())

#continuous features
cont = ['#_ppl_household', 'age', 'triglyceride','caffeine', 'lifetime_partners',
       'glycohemoglobin', 'CRP', 'tot_cholesterol','systolic_BP','diastolic_BP', 'BMI', 'waist_C', '#meals_fast_food',
       'min_sedetary', 'bone_mineral_density']

# categorical features
cat = ['race_ethnicity', 'edu_level', 'gender', 'marital_status', 'annual_HI',
       'doc_diabetes', 'how_healthy_diet', 'used_CMH',
       'health_insurance', 'doc_asthma', 'doc_overweight', 'doc_arthritis',
       'doc_CHF', 'doc_CHD', 'doc_heart_attack', 'doc_stroke',
       'doc_chronic_bronchitis', 'doc_liver_condition', 'doc_thyroid_problem',
       'doc_cancer', 'difficult_seeing', 'doc_kidney', 'broken_hip',
       'doc_osteoporosis', 'vigorous_activity', 'moderate_activity',
       'doc_sleeping_disorder', 'smoker', 'sexual_orientation',
       'alcoholic','herpes_2', 'HIV', 'doc_HPV','difficult_hearing', 'doc_COPD']

# multi-class features
cat_encode = ['race_ethnicity', 'edu_level', 'gender', 'marital_status', 'annual_HI','how_healthy_diet',
              'sexual_orientation']


#%%-------------------------------------------------------------------------
# KNN Progressive Model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=5)
knc = KNeighborsClassifier(n_neighbors=5)

def impute_progressive_knn(df, nan_df):

    predictions = []
    target = None
    df_copy=df.copy()

    for i in range(9,len(nan_df)+1):
        # handling identifiers
        df_start = df_copy[nan_df.iloc[2:i, ].index]

        # get the target name
        target = df_start.columns[-1]

        # get the name of the features
        features = np.setdiff1d(df_start.columns, [target])

        # Get NaN rows
        null_data = df_start[df_start.isnull().any(axis=1)]

        # drop target from NaN rows to get what rows we want to predict with
        imp = null_data[features].to_numpy()

        # drop Na rows from df
        df_1 = df_start.dropna()

        # split into training and target
        X_train = df_1[features].to_numpy()
        y_train = df_1[target].astype(int).to_numpy()

        #knr = KNeighborsRegressor(n_neighbors=5)
        #knc = KNeighborsClassifier(n_neighbors=5)

        # fit the model
        if target in cont:
            predictions = knr.fit(X_train, y_train).predict(imp)
        else:
            predictions = knc.fit(X_train, y_train).predict(imp)

        # find the values and indices to impute
        fill = pd.DataFrame(index=df_start.index[df_start.isnull().any(axis=1)],
                            data=predictions, columns=[target])

        # fill the df with predictions
        df_copy.fillna(fill, inplace=True)


    return df_copy


df_progressive_knn = impute_progressive_knn(df_raw, nan_df)


print(df_progressive_knn.head())