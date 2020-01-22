import streamlit as st
import pandas as pd
import dataframefunctions
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import exploreattributes



POSSIBLE_DATAEXP_ACTIONS = ["Missing data", "Explore attributes", "Dataset first look", "Interactive data cleaning"]


def render_data_explorations(dataframe):
    if dataframe is None:
        st.error("Please upload your dataset!")
    else:
        dataexp_action = st.sidebar.selectbox("What do you want to explore?", POSSIBLE_DATAEXP_ACTIONS)
        if dataexp_action == "Missing data":
            render_missing_data(dataframe)

        elif dataexp_action == "Dataset first look":
            render_first_look(dataframe)

        elif dataexp_action == "Explore attributes":
            exploreattributes.explore_attributes(dataframe)


def render_missing_data(dataframe):
    missing_values, missing_percentage = dataframefunctions.get_missing_values(dataframe)
    st.subheader("Exploring the missing values of the dataset:")
    st.dataframe(pd.concat([missing_values, missing_percentage], axis=1, keys=["Total", "percent"]))


def render_first_look(dataframe):
    number_of_rows = st.sidebar.slider('Number of rows', 1, 150, 10)
    st.markdown("### Exploring the dataset :mag: ###")
    st.dataframe(dataframe.head(number_of_rows))
    display_firstlook_comments(dataframe)


def display_firstlook_comments(dataframe):
    num_instances, num_features = dataframe.shape
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numerical_columns = dataframefunctions.get_numeric_columns(dataframe)
    random_cat_column = categorical_columns[0] if len(categorical_columns) > 0 else ""
    random_num_column = numerical_columns[0] if len(numerical_columns) > 0 else ""
    st.write("* The dataset has **%d** instances and **%d** features. Hence, the _feature-instances ratio_ is ~**%d**." % (num_instances, num_features, int(num_instances/num_features)))
    st.write("* The dataset has **%d** categorical columns (e.g. %s) and **%d** numerical columns (e.g. %s)." % (len(categorical_columns), random_cat_column, len(numerical_columns), random_num_column))


