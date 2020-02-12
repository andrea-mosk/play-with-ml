import streamlit as st
import dataframefunctions
import plotly.express as px


POSSIBLE_ATTRIBUTES_EXPLORATIONS = ["Scatterplot", "Boxplot"]


def load_page(dataframe):
    attrexp_action = st.sidebar.selectbox("Select the method", POSSIBLE_ATTRIBUTES_EXPLORATIONS)

    if attrexp_action == "Scatterplot":
        render_scatterplot(dataframe)

    elif attrexp_action == "Boxplot":
        render_boxplot(dataframe)


def render_scatterplot(dataframe):
    """Renders a scatterplot based on the user's input."""

    column_names, label_name = dataframefunctions.get_columns_and_label(dataframe)

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
    """Renders a boxplot based on the user's input."""

    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numeric_columns = dataframefunctions.get_numeric_columns(dataframe)

    boxplot_att1 = st.selectbox('Which feature on x? (only categorical features)', categorical_columns)
    boxplot_att2 = st.selectbox('Which feature on y? (only numeric features)',
                                numeric_columns,
                                index=len(numeric_columns) - 1)
    show_points = st.sidebar.checkbox("Show points", value=False)

    fig = px.box(dataframe,
                 x=boxplot_att1,
                 y=boxplot_att2,
                 points='all' if show_points else 'outliers')
    st.plotly_chart(fig)


