
import numpy as np
import pandas as pd
from nan_helper import nan_helper

# #continuous features
# cont = ['#_ppl_household', 'age', 'triglyceride','caffeine', 'lifetime_partners',
#        'glycohemoglobin', 'CRP', 'tot_cholesterol','systolic_BP','diastolic_BP', 'BMI', 'waist_C', '#meals_fast_food',
#        'min_sedetary', 'bone_mineral_density']

# # categorical features
# cat = ['race_ethnicity', 'edu_level', 'gender', 'marital_status', 'annual_HI',
#        'doc_diabetes', 'how_healthy_diet', 'used_CMH',
#        'health_insurance', 'doc_asthma', 'doc_overweight', 'doc_arthritis',
#        'doc_CHF', 'doc_CHD', 'doc_heart_attack', 'doc_stroke',
#        'doc_chronic_bronchitis', 'doc_liver_condition', 'doc_thyroid_problem',
#        'doc_cancer', 'difficult_seeing', 'doc_kidney', 'broken_hip',
#        'doc_osteoporosis', 'vigorous_activity', 'moderate_activity',
#        'doc_sleeping_disorder', 'smoker', 'sexual_orientation',
#        'alcoholic','herpes_2', 'HIV', 'doc_HPV','difficult_hearing', 'doc_COPD']

# # multi-class features
# cat_encode = ['race_ethnicity', 'edu_level', 'gender', 'marital_status', 'annual_HI','how_healthy_diet',
#               'sexual_orientation']


def var_type(df):
    # input dataframe
    # return continuous, non-continous, and multi-categorical features
 
    cols = df.columns
    cont = []
    non_cont = []
    cat_encode = []
    
    # get continuous variables
    for c in cols:
        if df[c].nunique() > 6:
            cont.append(c)
    
    # non cont is the difference
    non_cont = np.setdiff1d(cols, [cont])
    
    # multi-cat are those greater than 2
    for n in non_cont:
        if df[n].nunique() > 2:
            cat_encode.append(n)
    
    return cont, non_cont, cat_encode


def missing_values(df, threshold_col, threshold_row, impute_type):
    """
    Handle Missing Values

    Parameters
    ----------
    df : dataframe
    threshold_col: the proportion of missing values at which  to drop whole column
    threshold_row: the proportion of missing values at which to drop rows
    impute_type: mean or median imputation for continuous variables

    Returns
    ----------
    The dataframe without missing values

    """

    # Dropping Cols and Rows
    # call NaN helper function
    df_nan = nan_helper(df)

    # drop columns with higher proportion missing than threshold col
    df = df.drop((df_nan[df_nan['proportion'] > threshold_col]).index, 1)

    # drop rows with higher proportion missing than threshold row
    df_nan_2 = df_nan[df_nan['proportion'] > threshold_row]
    df = df.dropna(subset=np.intersect1d(df_nan_2.index, df.columns),
                   inplace=False)

    # Imputing values
    # Impute continuous variables with mean
    if impute_type == 'mean':
        for col in cont:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)
    # Impute continuous variables with median
    elif impute_type == 'median':
        for col in cont:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)

    # Impute categorical variables with most frequent/mode
    for col in cat:
        if col in df.columns:
            df[col].fillna(df[col].value_counts().index[0], inplace=True)

    return df
