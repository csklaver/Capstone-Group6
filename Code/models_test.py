# %%%%%%%%%%%%% Capstone %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caroline Sklaver
# %%%%%%%%%%%%% Date:
# June - 20 - 2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Testing Models  %%%%%%%%%%%%%%%%%%%%%%%%%%%%p
#%%-----------------------------------------------------------------------

#%%-----------------------------------------------------------------------
# Importing the required packages

import pandas as pd

path = ('/Users/carolinesklaver/Desktop/Capstone/NHANES/data/csv_data/')

import os
os.chdir(path)

SEED = 42

#%%-----------------------------------------------------------------------
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


#%%-----------------------------------------------------------------------
# getting different types of imputation

from nan_helper import nan_helper
from mean_median_imputation import missing_values
from knn_progressive_imputation import impute_progressive_knn


nan_df = nan_helper(df_raw)
# df_median_75 = missing_values(df_raw, 0.75,0.75, 'median')
# df_median = missing_values(df_raw, 0.75,0.75, 'median')
# df_knn_impute = impute_progressive_knn(df_raw, nan_df)
df_mlp_impute = pd.read_csv('df_progressive_mlp_2.csv')
df_mlp_impute.drop(['year'],axis=1, inplace=True)


#%%-----------------------------------------------------------------------
# divide into training and validation sets

from sklearn.model_selection import train_test_split

# divide into training and testing
df_raw_train, df_raw_test = train_test_split(df_mlp_impute, test_size=0.2, random_state=SEED)

# Reset the index
df_raw_train, df_raw_test = df_raw_train.reset_index(drop=True), df_raw_test.reset_index(drop=True)

# drop the target col in the testing data
df_raw_test.drop([target], axis=1, inplace=True)

# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)


# Divide the training data into training (80%) and validation (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42, stratify=df_train[target])

# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


#%%-----------------------------------------------------------------------
# Handing the identifiers
# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)


def id_checker(df):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe

    Returns
    ----------
    The dataframe of identifiers
    """

    # Get the identifiers
    df_id = df[[var for var in df.columns
                if df[var].nunique(dropna=True) == df[var].notnull().sum()]]

    return df_id


# Call id_checker on df
df_id = id_checker(df)

# Print the first 5 rows of df_id
print(df_id.head())


# removing the identifier
import numpy as np

# Remove the identifiers from df_train
df_train = df_train.drop(columns=np.intersect1d(df_id.columns, df_train.columns))

# Remove the identifiers from df_valid
df_valid = df_valid.drop(columns=np.intersect1d(df_id.columns, df_valid.columns))

# Remove the identifiers from df_test
df_test = df_test.drop(columns=np.intersect1d(df_id.columns, df_test.columns))


#%%---------------------------------------------------------------------------
# encoding and scaline the data
# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)

#df.drop(['year'],axis=1, inplace=True)

# Function to One-hot-encode the categorical features
# def one_hot_encode(df):
#     cols = df.columns[1:]
#     e_cols = list(set(cols).intersection(set(cat_encode)))
#     df_encode = pd.get_dummies(df, columns=e_cols)
#
#     return (df_encode)
#
# df = one_hot_encode(df)
#
# # encoding the categorical target
# from sklearn.preprocessing import LabelEncoder
#
# # The LabelEncoder
# le = LabelEncoder()
#
# # Encode the categorical target in the combined data
# df[target] = le.fit_transform(df[target].astype(str))

# separeating training, testing, validation
# Separating the training data
df_train = df.iloc[:df_train.shape[0], :].copy(deep=True)

# Separating the validation data
df_valid = df.iloc[df_train.shape[0]:df_train.shape[0] + df_valid.shape[0], :].copy(deep=True)

# Separating the testing data
df_test = df.iloc[df_train.shape[0] + df_valid.shape[0]:, :].copy(deep=True)


# get the name of the features
features = np.setdiff1d(df.columns, [target])


# Get the feature matrix
X_train = df_train[features].to_numpy()
X_valid = df_valid[features].to_numpy()
X_test = df_test[features].to_numpy()

# Get the target vector
y_train = df_train[target].astype(int).to_numpy()
y_valid = df_valid[target].astype(int).to_numpy()


from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()

# Standardize the training data
X_train = ss.fit_transform(X_train)

# Standardize the validation data
X_valid = ss.transform(X_valid)

# Standardize the testing data
X_test = ss.transform(X_test)


#%%-------------------------------------------------------------------------------
# running models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



models = {'lr': LogisticRegression(class_weight='balanced', random_state=SEED),
          'nb': GaussianNB(),
          'dtc': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=SEED),
          #'hgbc': HistGradientBoostingClassifier(random_state=42),
        'xgbc': XGBClassifier(seed=SEED),
        'mlpc': MLPClassifier(early_stopping=True, random_state=SEED)} #'svm': SVC(random_state=42)}


from sklearn.pipeline import Pipeline

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])

from sklearn.model_selection import PredefinedSplit

# Combine the feature matrix in the training and validation data
X_train_valid = np.vstack((X_train, X_valid))

# Combine the target vector in the training and validation data
y_train_valid = np.append(y_train, y_valid)

# Get the indices of training and validation data
train_valid_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_valid.shape[0], 0))

# The PredefinedSplit
ps = PredefinedSplit(train_valid_idxs)

param_grids = {}



# The grids for C
C_grids = [10 ** i for i in range(-2, 3)]

# The grids for tol
tol_grids = [10 ** i for i in range(-6, -1)]

# Update param_grids
param_grids['lr'] = [{'model__C': C_grids,
                      'model__tol': tol_grids}]

# Naive bayes param grids = none
param_grids['nb'] = {}


# The grids for min_samples_split
min_samples_split_grids = [2, 20, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 20, 100]

# Update param_grids
param_grids['dtc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]



# The grids for min_samples_split
min_samples_split_grids = [2, 20, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 20, 100]

# Update param_grids
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]



# # Update param_grids
# param_grids['abc'] = [{'model__learning_rate': learning_rate_grids}]
#

#
# # The grids for learning_rate
# learning_rate_grids = [10 ** i for i in range(-3, 2)]
#
# # The grids for min_samples_leaf
# min_samples_leaf_grids = [1, 20, 100]
#
# # Update param_grids
# param_grids['hgbc'] = [{'model__learning_rate': learning_rate_grids,
#                         'model__min_samples_leaf': min_samples_leaf_grids}]

#
# The grids for eta
eta_grids = [10 ** i for i in range(-4, 1)]

# The grids for gamma
gamma_grids = [0, 10, 100]

# The grids for lambda
lambda_grids = [10 ** i for i in range(-4, 5)]

#Update param_grids
param_grids['xgbc'] = [{'model__eta': eta_grids,
                        'model__gamma': gamma_grids,
                        'model__lambda': lambda_grids}]


# The grids for alpha
alpha_grids = [10 ** i for i in range(-6, -1)]

# The grids for learning_rate_init
learning_rate_init_grids = [10 ** i for i in range(-5, 0)]

# Update param_grids
param_grids['mlpc'] = [{'model__alpha': alpha_grids,
                        'model__learning_rate_init': learning_rate_init_grids}]


# The grids for C
# C_grids = [10 ** i for i in range(-2, 3)]
#
# # The grids for gamma
# gamma_grids = [0, 10, 100]
#
# # The grids for kernel
# kernel_grids = ['rbf', 'poly', 'sigmoid']
#
# # Update param_grids
# param_grids['svm'] = [{'model__C': C_grids,
#                         'model__gamma': gamma_grids,
#                         'model__kernel': kernel_grids}]
#


import os
from sklearn.metrics import confusion_matrix, classification_report

# Make directory
directory = os.path.dirname('cv_results/mlp_3/')
if not os.path.exists(directory):
    os.makedirs(directory)

from sklearn.model_selection import GridSearchCV, KFold

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_param_estimator_gs = []

for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_micro',
                      n_jobs=-1,
                      cv= KFold(n_splits=10, shuffle=True, random_state=SEED),
                      return_train_score=True)

    # Fit the pipeline
    gs = gs.fit(X_train_valid, y_train_valid)

    y_pred = gs.predict(X_valid)

    print(pipes[acronym], confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))

    # Update best_score_param_estimator_gs
    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(path_or_buf='cv_results/mlp_3/' + acronym + '.csv', index=False)



# Sort best_score_param_estimator_gs in descending order of the best_score_
best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_param_estimator_gs
print(pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator']))









