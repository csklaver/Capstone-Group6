# %%%%%%%%%%%%% Capstone %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caroline Sklaver
# %%%%%%%%%%%%% Date:
# June - 20 - 2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Imputing with MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%p
#%%-----------------------------------------------------------------------

#%%-----------------------------------------------------------------------
# Importing the required packages

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
# The LabelEncoder
le = LabelEncoder()

from sklearn.preprocessing import StandardScaler
# The StandardScaler
ss = StandardScaler()


path = ('/Users/carolinesklaver/Desktop/Capstone/NHANES/data/csv_data/')

import os
os.chdir(path)

#%%-----------------------------------------------------------------------
# Importing the data


df_raw = pd.read_csv('df_raw_v2.csv')


year = df_raw.pop('year')
df_raw.insert(1, 'year', year)

# bring year and target col to the beginning of df
dep = df_raw.pop('depressed')
df_raw.insert(2, 'depressed', dep)

# drop marijuana use
df_raw.drop(['used_marijuana'],axis=1, inplace=True)
# help!
#df_raw.drop(['year'],axis=1, inplace=True)


#%%-----------------------------------------------------------------------
# Function to identify missing values
from models_test import nan_helper


# call the function
nan_df = nan_helper(df_raw)
print(nan_df)

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
# MLP Model
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# Function to One-hot-encode the categorical features
def one_hot_encode(df):
    cols = df.columns[:-1]
    e_cols = list(set(cols).intersection(set(cat_encode)))
    df_encode = pd.get_dummies(df, columns=e_cols)

    return (df_encode)


def impute_mlp(df, nan_df):

    predictions = []
    target = None
    df_copy=df

    for i in range(9,len(nan_df)+1):
        # handling identifiers
        df_start = df_copy[nan_df.iloc[2:i, ].index]

        # get the target name
        target = df_start.columns[-1]

        # Function to One-hot-encode the categorical features
        # call encoding function for categorical features
        df_1 = one_hot_encode(df_start)

        # Encode the categorical target in the combined data
        if target in cat_encode:
            df_1[target] = le.fit_transform(df_1[target].astype(str))

        # get the name of the features
        features = np.setdiff1d(df_1.columns, [target])

        # Get NaN rows
        null_data = df_1[df_1.isnull().any(axis=1)]

        # drop target from NaN rows to get what rows we want to predict with
        imp = null_data[features].to_numpy()

        # drop Na rows from df
        df_1 = df_1.dropna()

        # split into training and target
        X_train = df_1[features].to_numpy()
        y_train = df_1[target].astype(int).to_numpy()

        # Standardize the training data
        X_train = ss.fit_transform(X_train)

        # fit the model
        # should this have different functions for cont/cat??
        clf.fit(X_train, y_train)

        # get the predictions
        predictions = clf.predict(imp)

        # find the values and indices to impute
        fill = pd.DataFrame(index=df_start.index[df_start.isnull().any(axis=1)],
                            data=predictions, columns=[target])

        # fill the df with predictions
        df_copy = df_copy.fillna(fill)


    return df_copy


dt = impute_mlp(df_raw, nan_df)

