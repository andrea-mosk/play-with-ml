import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import dataframefunctions


def remove_null_rows(x):
    null_rows = x.isnull().all(axis=1)
    x_prep = x.copy()
    return x_prep[~null_rows]


def useless(column):
    return len(column[column.notnull()].unique()) <= 1 or \
           (str(column.dtype) in ["object", "category"] and len(column.value_counts()) > 30)


def drop_high_zero_columns(df, percentage=0.95):
    zeros_threshold = df.shape[0] * percentage
    for column in df.columns:
        if df[column].values.tolist().count(0) > zeros_threshold:
            df.drop(column, axis=1, inplace=True)


def remove_useless_columns(df, null_percentage=0.95):

    # Remove columns with too many NaN
    null_threshold = int(df.shape[0] * (1 - null_percentage))
    df.dropna(axis=1, thresh=null_threshold, inplace=True)

    # Removing columns with no signal
    for col in df.columns:
        if useless(df[col]):
            df.drop(col, axis=1, inplace=True)


def remove_null_rows(x, y):
    x_prep = x.copy()
    y_prep = y.copy()
    null_rows = x_prep.isnull().all(axis=1)
    # TODO check the size of x and y, again (should be (N, M), (N, 1))
    return x_prep[~null_rows], y_prep[~null_rows]


def scale(df):
    nf = df.dtypes[df.dtypes != "object"].index
    scaler = StandardScaler()
    df.loc[:, nf] = scaler.fit_transform(df.loc[:, nf])


# TODO fare prova con "None" invece di most popular
def replace_missing_values(df):
    for col in df.columns:
        if df[col].isnull().any():
            col_dtype = str(df[col].dtype)
            if (col_dtype == "object" or col_dtype == "category"):
                # if (len(df[column].value_counts()) <= 30):
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

def encode(y):
    lc = LabelEncoder().fit(y)
    return lc.transform(y)


def preprocess(dataframe):

    # TODO check the size of x and y (should be (N, M), (N, 1))
    x = dataframe.drop(dataframe.columns[-1], axis=1)
    y = dataframe[dataframe.columns[-1]]

    remove_useless_columns(x)
    x, y = remove_null_rows(x, y)
    replace_missing_values(x)
    scale(x)
    x = pd.get_dummies(x)
    drop_high_zero_columns(x)

    if dataframefunctions.is_categorical(y):
        y = encode(y)

    return x, y

