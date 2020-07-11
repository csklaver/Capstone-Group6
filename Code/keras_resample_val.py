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
# df_mlp_impute = pd.read_csv('df_progressive_mlp_2.csv')
# df_mlp_impute.drop(['year'],axis=1, inplace=True)
# df_knn_impute = pd.read_csv('df_progressive_knn_2.csv')
# df_knn_impute.drop(['year'],axis=1, inplace=True)
# encoded
df_updated =  pd.read_csv('updated_train.csv')

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

print(df_updated.head())


# split and shuffle our dataset.
train_df, test_df = train_test_split(df_updated, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('depressed'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('depressed'))
test_labels = np.array(test_df.pop('depressed'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

features = np.setdiff1d(df_updated.columns, [target])

# scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

# -------------------- Random Over-Sampling ---------------
import imblearn
# from imblearn.over_sampling import RandomOverSampler
#
# ros = RandomOverSampler(random_state=SEED)
# X_train_ros, y_train_ros = ros.fit_sample(train_features, train_labels)
#
# print(X_train_ros.shape[0] - train_features.shape[0], 'new random picked points')
#
# X_train_ros = pd.DataFrame(X_train_ros)
# X_train_ros.columns = train_df.keys().tolist()
# y_train_ros = pd.DataFrame(y_train_ros, columns=['depressed'])
#
#
# print(y_train_ros.depressed.value_counts())

# -------------------- Random Under-Sampling ---------------
# import imblearn
# from imblearn.under_sampling import RandomUnderSampler
#
# rus = RandomUnderSampler(random_state=SEED)
# X_train_rus, y_train_rus = rus.fit_sample(train_features, train_labels)
#
# print(X_train_rus.shape[0] - train_features.shape[0], 'new random picked points')
#
# X_train_rus = pd.DataFrame(X_train_rus)
# X_train_rus.columns = train_df.keys().tolist()
# y_train_rus = pd.DataFrame(y_train_rus, columns=['depressed'])
#
#
# print(y_train_rus.depressed.value_counts())


# ------------------------------- SMOTE ----------------------------
import imblearn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=SEED, sampling_strategy='minority')
X_train_sm, y_train_sm = smote.fit_sample(train_features, train_labels)
X_train_sm = pd.DataFrame(X_train_sm)
X_train_sm.columns = train_df.keys().tolist()
y_train_sm = pd.DataFrame(y_train_sm, columns=['depressed'])
print('smote training df', X_train_sm.head())
print('smote value counts', y_train_sm.depressed.value_counts())



# %% -------------------------------------- Keras Model --------------------------------------------------------------
import keras
from keras import layers

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

model = keras.Sequential()

model.add(layers.Dense(64, activation='relu' , input_dim=83))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=METRICS)

history = model.fit(X_train_sm, y_train_sm, epochs=20, batch_size=128, validation_data=(val_features, val_labels))

score = model.evaluate(test_features, test_labels, batch_size=128)

print('Test loss:', score[0])
print('Test accuracy:', score[5])
print('Test precision:', score[6])
print('Test recall:', score[7])
print('Test auc:', score[8])
print('Test TP:', score[1])
print('Test FP:', score[2])
print('Test TN:', score[3])
print('Test FN:', score[4])

# train_predictions_resampled = model.predict(X_train_sm, batch_size=128)
# test_predictions_resampled = model.predict(X_test_ss, batch_size=128)

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