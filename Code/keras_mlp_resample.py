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

# # %% -------------------------------------- Data Prep ------------------------------------------------------------------

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
# df_mlp_impute = pd.read_csv('df_progressive_mlp_2.csv')
# df_mlp_impute.drop(['year'],axis=1, inplace=True)
# df_knn_impute = pd.read_csv('df_progressive_knn_2.csv')
# df_knn_impute.drop(['year'],axis=1, inplace=True)
# encoded

# from nan_helper import nan_helper
# from mean_median_imputation import missing_values
#
# nan_df = nan_helper(df_raw)
# df_mean_50 = missing_values(df_raw, 0.50,0.50, 'mean')
# df_mean_50.drop(['SEQN'],axis=1, inplace=True)
df_updated =  pd.read_csv('updated_train.csv')

print(df_updated.head())


from sklearn.model_selection import train_test_split

# divide into training and testing
df_raw_train, df_raw_test = train_test_split(df_updated, test_size=0.2, random_state=SEED)

# Reset the index
df_raw_train, df_raw_test = df_raw_train.reset_index(drop=True), df_raw_test.reset_index(drop=True)

# drop the target col in the testing data
# df_raw_test.drop([target], axis=1, inplace=True)

# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)


# Divide the training data into training (80%) and validation (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42, stratify=df_train[target])

# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


features = np.setdiff1d(df_updated.columns, [target])
# training set features and target
X_updated = df_train[features] # Features
y_updated = df_train[target] # Target
print(X_updated.head())

# get testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_updated, y_updated,
                                                    test_size=0.2,
                                                    random_state=SEED)
# getting training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2, random_state=SEED)




from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

# scaling testing data
X_test[cont] = ss.fit_transform(X_test[cont])
X_test_ss = X_test.copy()
X_test_ss.columns = X_test_ss.keys().tolist()
print('scaled testing\n', X_test_ss.head())


# scaling validation data
X_val[cont] = ss.fit_transform(X_val[cont])
X_val_ss = X_val.copy()
X_val_ss.columns = X_val_ss.keys().tolist()
print('scaled validation\n', X_val_ss.head())

# -------------------- Random Over-Sampling -------------------------
import imblearn
from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler(random_state=10)
# X_ros, y_ros = ros.fit_sample(X_updated, y_updated)
#
# print(X_ros.shape[0] - X_updated.shape[0], 'new random picked points')
#
#
# X_ros = pd.DataFrame(X_ros)
# X_ros.columns = X_updated.keys().tolist()
# y_ros = pd.DataFrame(y_ros, columns=['depressed'])
#
#
# print(y_ros.depressed.value_counts())

# ------------------------- SMOTE -------------------------------------
import imblearn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=SEED, sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X_updated, y_updated)
X_sm = pd.DataFrame(X_sm)
X_sm.columns = X_updated.keys().tolist()
y_sm = pd.DataFrame(y_sm, columns=['depressed'])
print('smote df', X_sm.head())
print('smote value counts', y_sm.depressed.value_counts())




# -------------------- Scaling Data -------------------------------------

# X=X_ros.copy()
# y=y_ros.depressed

X=X_sm.copy()
y=y_sm.depressed

# X[cont] = ss.fit_transform(X[cont])
# X_norm = pd.DataFrame(X, index=X_ros.index, columns=X_ros.columns)
# X_norm.head()
#
# scaling the training data
X[cont] = ss.fit_transform(X[cont])
X_norm = X.copy()
X_norm.columns = X.keys().tolist()
print('scaled training\n', X_norm.head())

# whole data that is SMOTE upsampled and scaled
X_df = X_norm.copy()


# ----------------------- Train test split ----------------------------------

# X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(X_df, y,
#                                                     test_size=0.2,
#                                                     random_state=SEED)

# Getting the oversampled, scaled training data
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_df, y,
                                                    test_size=0.2,
                                                    random_state=SEED)




# %% -------------------------------------- Training & Evaluating --------------------------------------------------------------
import keras
from keras import layers

model = keras.Sequential()

model.add(layers.Dense(64, activation='relu' , input_dim=83))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history = model.fit(X_train_sm, y_train_sm, epochs=15, batch_size=128, validation_data=(X_val_ss, y_val))

score = model.evaluate(X_test_ss, y_test, batch_size=128)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

train_predictions_resampled = model.predict(X_train_sm, batch_size=128)
test_predictions_resampled = model.predict(X_test_ss, batch_size=128)

from matplotlib import pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()