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

# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
#tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# getting different types of imputation
# Importing the data
#
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


#%%-----------------------------------------------------------------------
# Handling Missing values

from nan_helper import nan_helper

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


from nan_helper import nan_helper
from mean_median_imputation import missing_values
from knn_progressive_imputation import impute_progressive_knn


nan_df = nan_helper(df_raw)
df_median_75 = missing_values(df_raw, 0.75,0.75, 'median')
df_median_75.drop(['SEQN'],axis=1, inplace=True)
# df_median = missing_values(df_raw, 0.75,0.75, 'median')
# df_knn_impute = impute_progressive_knn(df_raw, nan_df)
# df_mlp_impute = pd.read_csv('df_progressive_mlp_2.csv')
# df_mlp_impute.drop(['year'],axis=1, inplace=True)
# df_mlp_impute.drop(['SEQN'],axis=1, inplace=True)


#%%-----------------------------------------------------------------------
# divide into training and validation sets

from sklearn.model_selection import train_test_split

# divide into training and testing
df_raw_train, df_raw_test = train_test_split(df_median_75, test_size=0.2, random_state=SEED)

# Reset the index
df_raw_train, df_raw_test = df_raw_train.reset_index(drop=True), df_raw_test.reset_index(drop=True)

# drop the target col in the testing data
# df_raw_test.drop([target], axis=1, inplace=True)

# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)

df = pd.concat([df_train, df_test], sort=False)
# get the name of the features
features = np.setdiff1d(df.columns, [target])


# Get the feature matrix
X_train = df_train[features].to_numpy()
X_test = df_test[features].to_numpy()

# Get the target vector
y_train = df_train[target].astype(int).to_numpy()
y_test = df_test[target].astype(int).to_numpy()


from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()

# Standardize the training data
X_train = ss.fit_transform(X_train)

# Standardize the testing data
X_test = ss.transform(X_test)



# %% -------------------------------------- Training Prep --------------------------------------------------------------
import keras
from keras import layers

model = keras.Sequential()

model.add(layers.Dense(64, activation='relu' , input_dim=49))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(X_test, y_test, batch_size=128)

print('Test loss:', score[0])
print('Test accuracy:', score[1])