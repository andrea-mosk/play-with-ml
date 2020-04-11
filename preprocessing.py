import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import dataframefunctions
import streamlit as st
import time


def useless(column):
    """Returns true if the column has only 1 value or if it is categorical and with more than 30 possible values."""

    return len(column[column.notnull()].unique()) <= 1 or \
                column.is_monotonic or \
                (str(column.dtype) in ["object", "category"] and len(column.value_counts()) > 30)


def drop_high_zero_columns(df, percentage=0.95):
    """Drop columns having the percentage of 0 higher than the input threshold (default is 0.95)."""

    zeros_threshold = df.shape[0] * percentage
    for column in df.columns:
        if df[column].values.tolist().count(0) > zeros_threshold:
            df.drop(column, axis=1, inplace=True)


def remove_useless_columns(df, null_percentage=0.8):
    """Removes columns with too many NaNs values (threshold is 0.8 by default) or with too little information."""

    null_threshold = int(df.shape[0] * (1 - null_percentage))
    df.dropna(axis=1, thresh=null_threshold, inplace=True)

    for col in df.columns:
        if useless(df[col]):
            df.drop(col, axis=1, inplace=True)


def remove_null_rows(x, y):
    """Removes, from both features and label, rows with all NaNs."""

    x_prep = x.copy()
    y_prep = y.copy()
    null_rows = x_prep.isnull().all(axis=1)
    return x_prep[~null_rows], y_prep[~null_rows]


def scale(df):
    """Scales numerical feature of the dataset."""

    nf = df.dtypes[df.dtypes != "object"].index
    scaler = StandardScaler()
    df.loc[:, nf] = scaler.fit_transform(df.loc[:, nf])


def replace_missing_values(df):
    """Replace missing values with mean for numerical features and most popular for categorical features."""

    for col in df.columns:
        if df[col].isnull().any():
            col_dtype = str(df[col].dtype)
            if col_dtype == "object" or col_dtype == "category":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)


def encode_label(y):
    """Transforms a categorical label into a numerical one."""

    lc = LabelEncoder().fit(y)
    return lc.transform(y)


def preprocess(dataframe):
    """Puts together all the preprocessing functions and returns the preprocessed dataset."""

    current_operation = st.empty()

    current_operation.text("starting preprocessing")
    progress_bar = st.progress(0)

    x = dataframe.drop(dataframe.columns[-1], axis=1)
    y = dataframe[dataframe.columns[-1]]

    current_operation.text("Removing useless columns")
    # time.sleep(2)
    remove_useless_columns(x)
    progress_bar.progress(15)

    current_operation.text("Removing useless rows")
    # time.sleep(2)
    x, y = remove_null_rows(x, y)
    progress_bar.progress(30)

    current_operation.text("Replacing missing values")
    replace_missing_values(x)
    progress_bar.progress(45)

    current_operation.text("Scaling numerical values")
    scale(x)
    progress_bar.progress(60)

    current_operation.text("Getting dummies")
    x = pd.get_dummies(x)
    progress_bar.progress(75)

    current_operation.text("Dropping more useless columns")
    drop_high_zero_columns(x)
    progress_bar.progress(90)

    if dataframefunctions.is_categorical(y):
        current_operation.text("Transforming the label")
        y = encode_label(y)
        progress_bar.progress(95)

    current_operation.text("Preprocessing completed!")
    progress_bar.progress(100)

    # Removing the progress bar and the text displaying the current operation.. (st.empty() does not seem to work).
    time.sleep(0.2)
    current_operation.text("")
    progress_bar.text("")

    return x, y
