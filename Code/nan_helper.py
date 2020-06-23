import pandas as pd


def nan_helper(df):
    """
    The NaN helper

    Parameters
    ----------
    df : dataframe

    Returns
    ----------
    The dataframe of variables with NaN (index),
    raw number missing, and their proportion

    """

    # get the raw number of missing values & sort
    missing = df.isnull().sum().sort_values(ascending=True)

    # get the proportion of missing values (%)
    proportion = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=True)

    # create table of missing data
    nan_data = pd.concat([missing, proportion], axis=1, keys=['missing', 'proportion'])

    return nan_data