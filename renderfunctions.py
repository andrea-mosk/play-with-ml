import streamlit as st
import pandas as pd
from dataframefunctions import *
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


POSSIBLE_DATAEXP_COMMANDS =["Missing data", "Features exploration", "Dataset first look", "Interactive data cleaning"]
POSSIBLE_FEATURE_EXPLORATIONS = ["Scatterplot", "Boxplot", "Linear correlation"]


def render_frame(selected_page):
    if selected_page == "Data exploration":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset and see all the statistics!", type='csv')
        if uploaded_file is not None:
            dataframe = get_dataframe(uploaded_file)
            render_data_exploration(dataframe)
    # Run predictions case, TODO
    else:
        return


def render_data_exploration(df):
    if df is not None:
        selected_option = st.sidebar.selectbox("What do you want to explore?", POSSIBLE_DATAEXP_COMMANDS)
        if selected_option == "Missing data":
            missing_values, missing_percentage = get_missing_values(df)
            st.subheader("Exploring the missing values of the dataset:")
            st.dataframe(pd.concat([missing_values, missing_percentage], axis=1, keys=["Total", "percent"]))
            comments = analyze_missing_data(missing_values, missing_percentage)
            for comment in comments:
                st.write(comment)

        elif selected_option == "Dataset first look":
            number_of_rows = st.sidebar.slider('Number of rows', 0, 100, 5)
            st.dataframe(df.head(number_of_rows))

        elif selected_option == "Features exploration":
            render_feature_exploration(df)
    else:
        st.warning("Please upload a dataset in the sidebar")


def render_feature_exploration(df):

    column_names, label_name = get_columns_and_labels(df)
    selected_method = st.sidebar.selectbox("Select the method", POSSIBLE_FEATURE_EXPLORATIONS)
    if selected_method == "Scatterplot":
        first_attribute = st.selectbox('Which feature on x?', column_names)
        second_attribute = st.selectbox('Which feature on y?', column_names, index=len(column_names) - 1)
        alpha_value = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0)
        if st.sidebar.checkbox("Color based on Label", value=True):
            fig = px.scatter(df, x=first_attribute, y=second_attribute, color=label_name, opacity=alpha_value)
        else:
            fig = px.scatter(df, x=first_attribute, y=second_attribute, opacity=alpha_value)
        st.plotly_chart(fig)

    elif selected_method == "Linear correlation":
        positive_corr, negative_corr = get_linear_correlations(df, label_name)
        st.write('Positively correlated features', positive_corr)
        st.write('Negatively correlated features', negative_corr)

    elif selected_method == "Boxplot":
        categorical_columns = get_categorical_columns(df)
        numeric_columns = get_numeric_columns(df)
        boxplot_att1 = st.selectbox('Which feature on x? (only categorical features)', categorical_columns)
        boxplot_att2 = st.selectbox('Which feature on y? (only numeric features)', numeric_columns, index=len(numeric_columns) - 1)
        sns.boxplot(x=boxplot_att1, y=boxplot_att2, data=df)
        st.pyplot()
