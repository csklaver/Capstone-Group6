# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold



# %% --------------------------------------- Set-Up --------------------------------------------------------------------
import pandas as pd

path = ('/Users/carolinesklaver/Desktop/Capstone/NHANES/data/csv_data/')

import os
os.chdir(path)

SEED = 42

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
path = ('/Users/carolinesklaver/Desktop/Capstone/NHANES/data/csv_data/')

import os
os.chdir(path)

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

# target binary feature
target = 'depressed'

df_raw = pd.read_csv('df_raw_v2.csv')

# bring year and target col to the beginning of df
year = df_raw.pop('year')
df_raw.insert(1, 'year', year)

dep = df_raw.pop('depressed')
df_raw.insert(2, 'depressed', dep)



# drop marijuana use
df_raw.drop(['used_marijuana'],axis=1, inplace=True)
# drop year?
df_raw.drop(['year'],axis=1, inplace=True)


# ---------------------------- Read in Data ------------------------
# read in data
# not encoded
df_mlp_impute = pd.read_csv('df_progressive_mlp_2.csv')
df_mlp_impute.drop(['year'],axis=1, inplace=True)
# df_knn_impute = pd.read_csv('df_progressive_knn_2.csv')
# df_knn_impute.drop(['year'],axis=1, inplace=True)

# from nan_helper import nan_helper
# from mean_median_imputation import missing_values
#
# nan_df = nan_helper(df_raw)
# df_mean_50 = missing_values(df_raw, 0.50,0.50, 'mean')
# df_mean_50.drop(['SEQN'],axis=1, inplace=True)
# # encoded MLP imputed
# df_updated =  pd.read_csv('updated_train.csv')
#
# # encoded KNN imputed
# df_updated =  pd.read_csv('updated_train_knn.csv')

# encoded mean imputed
# df_updated =  pd.read_csv('updated_train_mean.csv')

from sklearn.model_selection import train_test_split

# Function to One-hot-encode the categorical features
def one_hot_encode(df):
    cols = df.columns
    e_cols = list(set(cols).intersection(set(cat_encode)))
    df_encode = pd.get_dummies(df, columns=e_cols)

    return (df_encode)

df = one_hot_encode(df_mlp_impute)

# get the name of the features
features = np.setdiff1d(df.columns, [target])



# split data into active and inactive
N_active = df[df['depressed']==1]
N_inactive = df[df['depressed']==0]

print('Shape of positive:', N_active.shape)
print('Shape of negative:', N_inactive.shape)

# get testing data with 30% of the active, 70% inactive, and we want this to be 20% of total data points
import math
n_test_rows = math.floor(df.shape[0]*0.2)
n_pos_test = math.floor(n_test_rows*0.3)
n_neg_test = n_test_rows-n_pos_test


TsN_active = N_active.sample(n=n_pos_test, replace=False)
TsN_inactive = N_inactive.sample(n=n_neg_test, replace=False)

print('Shape of test positive:', len(TsN_active))
print('Shape of test negative:', len(TsN_inactive))

TrN_active = N_active.drop(TsN_active.index)
TrN_inactive = N_inactive.drop(TsN_inactive.index)
print('Shape of train positive:', len(TrN_active))
print('Shape of test negative:', len(TrN_inactive))

# generate testing data
testing = pd.concat([TsN_active, TsN_inactive])


# Get the number of training sub-samples we need
NoS = math.floor(len(TrN_inactive)/len(TrN_active))

# number to sample to get balanced data
n_bal = len(TrN_active)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

rfc = RandomForestClassifier(n_estimators=100,max_features=9)
svm = SVC()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

def get_matrix(df):
    t_features = df[features]
    # scale continuous features
    t_features[cont] = ss.fit_transform(t_features[cont])

    t_target = df[target]

    return t_features, t_target

f1_scores = []
prec_scores = []
recall_scores = []


# repeat and average scores
for i in range(20):
    # split data into active and inactive
    N_active = df[df['depressed'] == 1]
    N_inactive = df[df['depressed'] == 0]

    # get testing data with 30% of the active, 70% inactive, and we want this to be 20% of total data points

    n_test_rows = math.floor(df.shape[0] * 0.2)
    n_pos_test = math.floor(n_test_rows * 0.3)
    n_neg_test = n_test_rows - n_pos_test

    TsN_active = N_active.sample(n=n_pos_test, replace=False)
    TsN_inactive = N_inactive.sample(n=n_neg_test, replace=False)

    TrN_active = N_active.drop(TsN_active.index)
    TrN_inactive = N_inactive.drop(TsN_inactive.index)

    # generate testing data
    testing = pd.concat([TsN_active, TsN_inactive])

    # Get the number of training sub-samples we need
    NoS = math.floor(len(TrN_inactive) / len(TrN_active))

    # number to sample to get balanced data
    n_bal = len(TrN_active)
    predictions = []

    for j in range(NoS):
        # sample the negative instances
        T = TrN_inactive.sample(n=n_bal, replace=False)
        # combine training data
        training = pd.concat([T, TrN_active])

        # get the features and targets, standardized
        train_features, train_target = get_matrix(training)

        # train the model and save output/y_pred
        rfc.fit(train_features, train_target)

        testing_features, testing_target = get_matrix(testing)
        y_pred = rfc.predict(testing_features)
        predictions.append(y_pred)

        # remove T.index from TrN_inactive
        TrN_inactive.drop(T.index, inplace=True)


    # do the remaining TrN_inactive?
    # get majority vote for every prediction to create final_y_pred

# make an empty list to hold the majority predictions
    majority_pred = []

    # get majority vote of every prediction
    for k in range(len(predictions[0])):
        l = [item[k] for item in predictions]
        c = Counter(l)
        value, count = c.most_common()[0]
        majority_pred.append(value)


    print(Counter(majority_pred))

    # evaluate the model with testing targets
    test_labels = testing[target]

    print('--------------------------------------------------------')
    print('Round:', i)
    print('\nConfusion Matrix:\n', confusion_matrix(test_labels, majority_pred))
    f1 = f1_score(test_labels, majority_pred, average='micro')
    prec = precision_score(test_labels, majority_pred)
    recall = recall_score(test_labels, majority_pred)

    print('F1-micro score:', f1)
    print('Precision:', prec)
    print('Recall:',recall)
    print('--------------------------------------------------------')

    f1_scores.append(f1)
    prec_scores.append(prec)
    recall_scores.append(recall)


print('Mean F1:', np.mean(f1_scores))
print('Mean Precision:', np.mean(prec_scores))
print('Mean Recall:', np.mean(recall_scores))