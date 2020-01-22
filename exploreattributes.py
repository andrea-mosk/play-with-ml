import streamlit as st
import dataframefunctions
import plotly.express as px
import seaborn as sns

POSSIBLE_ATTRIBUTES_EXPLORATIONS = ["Linear correlation", "Scatterplot", "Boxplot"]


def explore_attributes(dataframe):
    attrexp_action = st.sidebar.selectbox("Select the method", POSSIBLE_ATTRIBUTES_EXPLORATIONS)
    if attrexp_action == "Linear correlation":
        render_linear_correlation(dataframe)

    elif attrexp_action == "Scatterplot":
        render_scatterplot(dataframe)

    elif attrexp_action == "Boxplot":
        render_boxplot(dataframe)


def render_linear_correlation(dataframe):
    label_name = list(dataframe.columns)[-1]
    if dataframefunctions.is_categorical(dataframe[label_name]):
        display_correlation_error()
        return
    positive_corr, negative_corr = dataframefunctions.get_linear_correlations(dataframe, label_name)
    st.write('Positively correlated features', positive_corr)
    st.write('Negatively correlated features', negative_corr)


def render_scatterplot(dataframe):
    column_names = list(dataframe.columns)
    label_name = column_names[-1]
    first_attribute = st.selectbox('Which feature on x?', column_names)
    second_attribute = st.selectbox('Which feature on y?', column_names, index=2)
    alpha_value = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0)
    colored = st.sidebar.checkbox("Color based on Label", value=True)
    sized = st.checkbox("Size based on other attribute", value=False)
    if sized:
        size_attribute = st.selectbox('Which attribute?', column_names)
    fig = px.scatter(dataframe,
                     x=first_attribute,
                     y=second_attribute,
                     color=label_name if colored else None,
                     opacity=alpha_value,
                     size=None if not sized else size_attribute)
    st.plotly_chart(fig)


def render_boxplot(dataframe):
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numeric_columns = dataframefunctions.get_numeric_columns(dataframe)
    label_name = list(dataframe.columns)[-1]
    boxplot_att1 = st.selectbox('Which feature on x? (only categorical features)', categorical_columns)
    boxplot_att2 = st.selectbox('Which feature on y? (only numeric features)', numeric_columns,
                                index=len(numeric_columns) - 1)
    show_points = st.sidebar.checkbox("Show points", value=False)
    fig = px.box(dataframe,
                 x=boxplot_att1,
                 y=boxplot_att2,
                 color=label_name,
                 points='all' if show_points else 'outliers')
    st.plotly_chart(fig)


def display_correlation_error():
    st.write(":no_entry::no_entry::no_entry:")
    st.write("It's **not** possible to determine a linear correlation with a categorical label.")
    st.write("For more info, please check [this link.](https://stackoverflow.com/questions/47894387/how-to-correlate-an-ordinal-categorical-column-in-pandas)")
