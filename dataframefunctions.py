import pandas as pd
import streamlit as st


@st.cache
def get_dataframe(uploaded_file):
    return pd.read_csv(uploaded_file) if uploaded_file is not None \
        else None


def get_missing_values(dataframe):
    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    missing_percentage = (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending=False)
    return missing_values, missing_percentage


# TODO
def missing_data_advanced_analysis(missing_percentage, comments_list):
    pass

@st.cache
def analyze_missing_data(missing_values, missing_percentage):
    list_of_comments = []
    first_value = missing_values[0]
    if first_value == 0:
        list_of_comments.append("The dataset has no missing values, you're good!")
    elif first_value < 0.05:
        list_of_comments.append("The dataset has very few missing values!")
    else:
        missing_data_advanced_analysis(missing_percentage, list_of_comments)
    return list_of_comments


def get_linear_correlations(df, label_name):
    # TODO Transform categorical label into numerical
    corr_matrix = df.corr()
    positive_corr = get_signed_correlations(corr_matrix, label_name, positive=True)
    negative_corr = get_signed_correlations(corr_matrix, label_name, positive=False)
    positive_df = pd.DataFrame(positive_corr).rename(columns={label_name: 'Correlation'})
    negative_df = pd.DataFrame(negative_corr).rename(columns={label_name: 'Correlation'})
    return positive_df, negative_df


def get_signed_correlations(corr_matrix, label_name, positive=True):
    correlation = corr_matrix[label_name][corr_matrix[label_name] >= 0] if positive \
                                                                       else corr_matrix[label_name][corr_matrix[label_name] < 0]
    return correlation.iloc[:-1].sort_values(ascending=not positive)


def get_columns_and_labels(df):
    column_names = list(df.columns.values)
    return column_names, column_names[len(column_names) - 1]


def get_categorical_columns(df):
    return list(df.select_dtypes(exclude=['number']).columns.values)


def get_numeric_columns(df):
    return list(df.select_dtypes(['number']).columns.values)


def is_categorical(column):
    return column.dtype.name == 'object'