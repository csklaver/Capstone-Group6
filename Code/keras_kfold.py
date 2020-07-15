# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
from time import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
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

# divide into training and testing
df_raw_train, df_raw_test = train_test_split(df, test_size=0.2, random_state=SEED)
# Reset the index
df_raw_train, df_raw_test = df_raw_train.reset_index(drop=True), df_raw_test.reset_index(drop=True)

# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)
# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)

train_inputs = df_train[features]
test_inputs = df_test[features]
train_target = df_train[target]
test_target = df_test[target]

print('Training data shape:', df_train.shape)
print('Test data shape:', df_test.shape)


# -------------------- Random Over-Sampling ---------------
import imblearn
from imblearn.over_sampling import RandomOverSampler
#
ros = RandomOverSampler(random_state=SEED)
def ROS(features, target):
    X_train_ros, y_train_ros = ros.fit_sample(features, target)
    print(X_train_ros.shape[0] - features.shape[0], 'new random picked points')
    return X_train_ros, y_train_ros
#
# X_train_ros = pd.DataFrame(X_train_ros)
# X_train_ros.columns = train_df.keys().tolist()
# y_train_ros = pd.DataFrame(y_train_ros, columns=['depressed'])
#
#
# print(y_train_ros.depressed.value_counts())

# -------------------- Random Under-Sampling ---------------
# import imblearn
from imblearn.under_sampling import RandomUnderSampler
#
rus = RandomUnderSampler(random_state=SEED)
def RUS(features, target):
    X_train_rus, y_train_rus = rus.fit_sample(features, target)
    print(X_train_rus.shape[0] - features.shape[0], 'randomly picked points')

# X_train_rus = pd.DataFrame(X_train_rus)
# X_train_rus.columns = train_df.keys().tolist()
    y_train_rus_df = pd.DataFrame(y_train_rus, columns=['depressed'])
#
#
    print(y_train_rus_df.depressed.value_counts())
    return X_train_rus, y_train_rus

# ------------------------------- SMOTE ----------------------------
# import imblearn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=SEED, sampling_strategy='minority')

def smote_os(features, target):
    X_train_sm, y_train_sm = smote.fit_sample(features, target)
# X_train_sm = pd.DataFrame(X_train_sm)
# X_train_sm.columns = train_df.keys().tolist()
    y_train_sm_df = pd.DataFrame(y_train_sm, columns=['depressed'])
# print('smote training df', X_train_sm.head())
    print('smote value counts:', y_train_sm_df.depressed.value_counts())
    return X_train_sm, y_train_sm
#

# ------------------------------- SMOTETomek ----------------------------

# over and undersampling
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=SEED)
def sm_tomek(features, target):
    X_train_smt, y_train_smt = smt.fit_sample(features, target)

    print(X_train_smt.shape[0] - features.shape[0], 'new random picked points')

    y_smt = pd.DataFrame(y_train_smt, columns=['depressed'])


    print(y_smt.depressed.value_counts())
    return X_train_smt, y_train_smt

# %% -------------------------------------- Keras Model --------------------------------------------------------------
import keras
from keras import layers
from sklearn.model_selection import KFold
import numpy as np
num_folds = 20
import keras.backend as K
K.clear_session()
tf.config.experimental.set_visible_devices([], 'GPU')

train_inputs.astype('float32')
test_inputs.astype('float32')

# scale the features
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_inputs[cont] = ss.fit_transform(train_inputs[cont])
test_inputs[cont] = ss.transform(test_inputs[cont])



# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
prec_per_fold = []
recall_per_fold = []
TP_per_fold = []
FP_per_fold = []
TN_per_fold = []
FN_per_fold = []
AUC_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((train_inputs, test_inputs), axis=0)
targets = np.concatenate((train_target, test_target), axis=0)

#
METRICS = [
    keras.metrics.BinaryAccuracy(name='Accuracy'),
    keras.metrics.Precision(name='Precision'),
    keras.metrics.Recall(name='Recall'),
    keras.metrics.AUC(name='AUC'),
    keras.metrics.TruePositives(name='True Positives'),
    keras.metrics.FalsePositives(name='False Positives'),
    keras.metrics.TrueNegatives(name='True Negatives'),
    keras.metrics.FalseNegatives(name='False Negatives'),
]

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

    # Define the model architecture
    model = keras.Sequential()

    model.add(layers.Dense(64, activation='tanh' , input_dim=84))
    model.add(Dropout(0.5))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    class_weights = {1:14,
                    0:1}

    opt = keras.optimizers.rmsprop(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,metrics=['accuracy', keras.metrics.Precision(name='precision'),
                                        keras.metrics.Recall(name='recall'),
                                        keras.metrics.AUC(name='AUC'),
                                        keras.metrics.TruePositives(name='TP'),
                                        keras.metrics.FalsePositives(name='FP'),
                                        keras.metrics.TrueNegatives(name='TN'),
                                        keras.metrics.FalseNegatives(name='FN'),])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # RESAMPLING Training set
    train_ros_inputs, train_ros_targets = ROS(inputs[train], targets[train])
    #train_smote_inputs, train_smote_targets = smote_os(inputs[train], targets[train])
    #train_rus_inputs, train_rus_targets = RUS(inputs[train], targets[train])
    #train_smt_inputs, train_smt_targets = sm_tomek(inputs[train], targets[train])

    # Fit data to model
    history = model.fit(train_ros_inputs, train_ros_targets,
                        batch_size=32,
                        epochs=20, verbose=0)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)

    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
        f' {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]};'
        f' {model.metrics_names[3]} of {scores[3]}; {model.metrics_names[4]} of {scores[4]};'
        f' {model.metrics_names[5]} - {scores[5]}; {model.metrics_names[6]} - {scores[6]};'
        f' {model.metrics_names[7]} - {scores[7]}; {model.metrics_names[8]} - {scores[8]};')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    prec_per_fold.append(scores[2])
    recall_per_fold.append(scores[3])
    AUC_per_fold.append(scores[4])
    TP_per_fold.append(scores[5])
    FP_per_fold.append(scores[6])
    TN_per_fold.append(scores[7])
    FN_per_fold.append(scores[8])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - Precision: {prec_per_fold[i]}'
        f'- Recall: {recall_per_fold[i]} - AUC: {AUC_per_fold[i]} - TP: {TP_per_fold[i]}'
        f'- FP: {FP_per_fold[i]} - TN: {TN_per_fold[i]} - FN: {FN_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> Precision: {np.mean(prec_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)}')
print(f'> AUC: {np.mean(AUC_per_fold)}')
print(f'> TP: {np.sum(TP_per_fold)}')
print(f'> FP: {np.sum(FP_per_fold)}')
print(f'> TN: {np.sum(TN_per_fold)}')
print(f'> FN: {np.sum(FN_per_fold)}')
print('------------------------------------------------------------------------')

