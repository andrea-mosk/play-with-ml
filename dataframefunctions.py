import pandas as pd
import streamlit as st


@st.cache
def get_dataframe(dataset_file):
    """If the input dataset is not none, returns the equivalent Pandas Dataframe."""

    return pd.read_csv(dataset_file) if dataset_file is not None \
        else None


def get_missing_values(dataframe):
    """Returns the missing values and the missing percentages for each column."""

    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    missing_percentage = (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending=False)
    return missing_values, missing_percentage


@st.cache
def get_linear_correlations(df, label_name):
    """returns two dataframes: the first contains positive correlations between features and label,
    and the other contains negative correlations."""

    corr_matrix = df.corr()
    positive_corr = get_signed_correlations(corr_matrix, label_name, positive=True)
    negative_corr = get_signed_correlations(corr_matrix, label_name, positive=False)
    positive_df = pd.DataFrame(positive_corr).rename(columns={label_name: 'Correlation'})
    negative_df = pd.DataFrame(negative_corr).rename(columns={label_name: 'Correlation'})
    return positive_df, negative_df


def get_signed_correlations(corr_matrix, label_name, positive=True):
    """Get positive or negative correlations, based on the value of the input."""

    correlation = corr_matrix[label_name][corr_matrix[label_name] >= 0] \
        if positive else corr_matrix[label_name][corr_matrix[label_name] < 0]

    return correlation.iloc[:-1].sort_values(ascending=not positive)


@st.cache
def get_columns_and_label(df):
    """Returns the columns and the label of the input dataframe."""

    column_names = list(df.columns.values)
    return column_names, column_names[len(column_names) - 1]


@st.cache
def get_categorical_columns(df):
    """Returns the list of categorical columns of the input dataframe."""

    return list(df.select_dtypes(exclude=['number']).columns.values)


@st.cache
def get_numeric_columns(df):
    """Returns the list of numerical columns of the input dataframe."""

    return list(df.select_dtypes(['number']).columns.values)


def is_categorical(column):
    return column.dtype.name == 'object'


def color_null_red(val):
    """Coloring in red the NaN values."""

    return 'color: red' if pd.isnull(val) else 'color: black'
