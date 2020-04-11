import streamlit as st
import pandas as pd
import dataframefunctions
import plots
import featuresanalysis


POSSIBLE_DATAEXP_ACTIONS = ["Dataset first look", "Plots", "Features"]


def load_page(dataframe):
    if dataframe is None:
        st.error("Please upload your dataset!")
    else:
        dataexp_action = st.sidebar.selectbox("What do you want to explore?", POSSIBLE_DATAEXP_ACTIONS)

        if dataexp_action == "Dataset first look":
            render_first_look(dataframe)
            if st.sidebar.checkbox("Compute missing values"):
                render_missing_data(dataframe)
            if st.sidebar.checkbox("Compute linear correlation"):
                render_linear_correlation(dataframe)

        elif dataexp_action == "Plots":
            plots.load_page(dataframe)

        elif dataexp_action == "Features":
            featuresanalysis.load_page(dataframe)


def render_missing_data(dataframe):
    """Renders the missing values and the missing percentages for each column."""

    missing_values, missing_percentage = dataframefunctions.get_missing_values(dataframe)
    st.markdown("## **Missing values :mag:** ##")
    st.dataframe(pd.concat([missing_values, missing_percentage], axis=1, keys=["Total", "percent"]))


def render_first_look(dataframe):
    """Renders the head of the dataset (with nan values colored in red),
     and comments regarding instances, columns, and missing values."""

    number_of_rows = st.sidebar.slider('Number of rows', 1, 150, 10)
    st.markdown("## **Exploring the dataset :mag:** ##")
    if st.sidebar.checkbox("Color NaN values in red", value=True):
        st.dataframe(dataframe.head(number_of_rows).style.applymap(dataframefunctions.color_null_red))
    else:
        st.dataframe(dataframe.head(number_of_rows))
    render_firstlook_comments(dataframe)


# TODO improve such that all the type of columns are considered
def render_firstlook_comments(dataframe):
    """Makes a first analysis of the dataset and shows comments based on that."""

    num_instances, num_features = dataframe.shape
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numerical_columns = dataframefunctions.get_numeric_columns(dataframe)
    cat_column = categorical_columns[0] if len(categorical_columns) > 0 else ""
    num_column = numerical_columns[0] if len(numerical_columns) > 0 else ""
    total_missing_values = dataframe.isnull().sum().sum()

    st.write("* The dataset has **%d** observations and **%d** variables. \
             Hence, the _instances-features ratio_ is ~**%d**."
             % (num_instances, num_features, int(num_instances/num_features)))

    st.write("* The dataset has **%d** categorical columns (e.g. %s) and **%d** numerical columns (e.g. %s)."
             % (len(categorical_columns), cat_column, len(numerical_columns), num_column))

    st.write("* Total number of missing values: **%d** (~**%.2f**%%)."
             % (total_missing_values, 100*total_missing_values/(num_instances*num_features)))


def render_linear_correlation(dataframe):
    """If the label is not categorical, renders the linear correlation between the features and the label."""

    st.markdown("## **Linear correlation ** ##")
    df_columns = list(dataframe.columns.values)
    label_name = df_columns[len(df_columns) - 1]

    # If the label is not categorical, show an error
    if dataframefunctions.is_categorical(dataframe[label_name]):
        display_correlation_error()
        return

    positive_corr = dataframefunctions.get_linear_correlation(dataframe, label_name, positive=True)
    negative_corr = dataframefunctions.get_linear_correlation(dataframe, label_name, positive=False)
    st.write('Positively correlated features :chart_with_upwards_trend:', positive_corr)
    st.write('Negatively correlated features :chart_with_downwards_trend:', negative_corr)


def display_correlation_error():
    st.write(":no_entry::no_entry::no_entry:")
    st.write("It's **not** possible to determine a linear correlation with a categorical label.")
    st.write("For more info, please check [this link.]\
             (https://stackoverflow.com/questions/47894387/how-to-correlate-an-ordinal-categorical-column-in-pandas)")

