import streamlit as st
import dataframefunctions
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff


POSSIBLE_ATTRIBUTES_EXPLORATIONS = ["Scatter plot",
                                    "Box plot",
                                    "Correlation matrix",
                                    "Count plot",
                                    "Distribution plot"]


def load_page(dataframe):
    attrexp_action = st.sidebar.selectbox("Select the method", POSSIBLE_ATTRIBUTES_EXPLORATIONS)

    if attrexp_action == "Scatter plot":
        render_scatterplot(dataframe)

    elif attrexp_action == "Box plot":
        render_boxplot(dataframe)

    elif attrexp_action == "Correlation matrix":
        render_corr_matrix(dataframe)

    elif attrexp_action == "Count plot":
        render_count_plot(dataframe)

    elif attrexp_action == "Distribution plot":
        render_distplot(dataframe)


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

    with st.spinner("Plotting data.."):
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

    with st.spinner("Plotting data.."):
        fig = px.box(dataframe,
                     x=boxplot_att1,
                     y=boxplot_att2,
                     points='all' if show_points else 'outliers')

        st.plotly_chart(fig)


def render_corr_matrix(dataframe):
    """Renders a correlation matrix based on the user's input."""

    if len(dataframefunctions.get_numeric_columns(dataframe)) > 30:
        st.warning("Warning: since the dataset has more than 30 features, the figure might be inaccurate.")

    corr = dataframe.corr()

    # Masking the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    f, ax = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(10, 140, n=9, s=90, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    with st.spinner("Plotting data.."):
        st.pyplot()


def render_count_plot(dataframe):
    """Renders a count plot based on the user's input."""

    feature = st.selectbox('Which feature?', list(dataframe.columns))
    with st.spinner("Plotting data.."):
        sns.countplot(x=feature, data=dataframe)
        st.pyplot()


def render_distplot(dataframe):
    """Renders a distribution plot based on the user's input."""

    feature = st.selectbox('Which feature?', dataframefunctions.get_numeric_columns(dataframe))
    with st.spinner("Plotting distribution.."):
        sns.distplot(dataframe[feature], color='g')
        st.pyplot()


