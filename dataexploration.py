import time

import streamlit as st
import pandas as pd
import dataframefunctions
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import exploreattributes



POSSIBLE_DATAEXP_ACTIONS = ["Dataset first look", "Explore attributes", "Interactive data cleaning"]


def render_data_explorations(dataframe):
    if dataframe is None:
        st.error("Please upload your dataset!")
    else:
        dataexp_action = st.sidebar.selectbox("What do you want to explore?", POSSIBLE_DATAEXP_ACTIONS)

        if dataexp_action == "Dataset first look":
            render_first_look(dataframe)
            render_missing_data(dataframe)
            render_linear_correlation(dataframe)

        elif dataexp_action == "Explore attributes":
            exploreattributes.explore_attributes(dataframe)


def render_missing_data(dataframe):
    missing_values, missing_percentage = dataframefunctions.get_missing_values(dataframe)
    st.markdown("## **Missing values :mag:** ##")
    st.dataframe(pd.concat([missing_values, missing_percentage], axis=1, keys=["Total", "percent"]))


def render_first_look(dataframe):
    number_of_rows = st.sidebar.slider('Number of rows', 1, 150, 10)
    st.markdown("## **Exploring the dataset :mag:** ##")
    st.dataframe(dataframe.head(number_of_rows).style.applymap(dataframefunctions.color_null_red))
    display_firstlook_comments(dataframe)


def display_firstlook_comments(dataframe):
    num_instances, num_features = dataframe.shape
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numerical_columns = dataframefunctions.get_numeric_columns(dataframe)
    random_cat_column = categorical_columns[0] if len(categorical_columns) > 0 else ""
    random_num_column = numerical_columns[0] if len(numerical_columns) > 0 else ""
    total_missing_values = dataframe.isnull().sum().sum()
    st.write("* The dataset has **%d** observations and **%d** variables. Hence, the _instances-features ratio_ is ~**%d**."
             % (num_instances, num_features, int(num_instances/num_features)))
    st.write("* The dataset has **%d** categorical columns (e.g. %s) and **%d** numerical columns (e.g. %s)."
             % (len(categorical_columns), random_cat_column, len(numerical_columns), random_num_column))
    st.write("* Total number of missing values: **%d** (~**%.2f**%%)."
             % (total_missing_values, 100*total_missing_values/(num_instances*num_features)))


def render_linear_correlation(dataframe):
    label_name = list(dataframe.columns)[-1]
    if dataframefunctions.is_categorical(dataframe[label_name]):
        display_correlation_error()
        return
    st.markdown("## **Linear correlation ** ##")
    positive_corr, negative_corr = dataframefunctions.get_linear_correlations(dataframe, label_name)
    st.write('Positively correlated features :chart_with_upwards_trend:', positive_corr)
    st.write('Negatively correlated features :chart_with_downwards_trend:', negative_corr)


def display_correlation_error():
    st.write(":no_entry::no_entry::no_entry:")
    st.write("It's **not** possible to determine a linear correlation with a categorical label.")
    st.write("For more info, please check [this link.](https://stackoverflow.com/questions/47894387/how-to-correlate-an-ordinal-categorical-column-in-pandas)")

