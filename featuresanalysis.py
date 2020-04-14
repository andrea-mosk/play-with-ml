import streamlit as st
import dataframefunctions
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


def load_page(dataframe):
    selected_feature = st.selectbox("Which feature?", list(dataframe.columns.values))
    num_instances = dataframe.shape[0]
    current_feature = dataframe[selected_feature]
    if dataframefunctions.is_categorical(current_feature):
        describe_common_statistics(current_feature, num_instances)
        render_countplot(current_feature, selected_feature, dataframe)
        render_boxplot(dataframe, selected_feature)
    else:
        describe_numerical_feature(current_feature, num_instances)
        render_countplot(current_feature, selected_feature, dataframe)
        render_scatterplot(dataframe, selected_feature)


# TODO solve problem with ID
def describe_numerical_feature(feature, num_instances):
    """All the statistics for numerical features, such as min, max, skew, .."""

    st.markdown("**Mean**: %.2f, &nbsp; &nbsp; &nbsp; **Median**: %.2f" % (feature.mean(), feature.median()))
    st.markdown("**Max**: %.2f, &nbsp; &nbsp; &nbsp; **Min**: %.2f" % (feature.max(), feature.min()))
    st.write("**95th percentile**: %.2f" % (feature.quantile(.95)))
    st.markdown("**Skewness**: %.2f, &nbsp; &nbsp; &nbsp; **Kurtosis**: %.2f" % (feature.skew(), feature.kurt()))

    num_zeros = feature[feature == 0].count()
    st.markdown("**Num. of zeros**: %d (%.2f%%)" % (num_zeros, num_zeros*100/num_instances))

    describe_common_statistics(feature, num_instances)
    if feature.nunique() >= 20:
        feature = feature[~feature.isnull()]
        sns.distplot(feature)
        st.pyplot()


def describe_common_statistics(feature, num_instances):
    """All the statistics in common between all features, such as used memory."""
    st.write("**Memory usage**: %.2f Bytes" % (feature.memory_usage()))
    st.write("**Unique elements**: %d (%.2f%%)" % (feature.nunique(), feature.nunique()*100/num_instances))
    st.write("**Most popular element**: %s" % (feature.mode()[0]))
    missing_values = feature.isna().sum()
    st.write("**Missing values**: %d (%.2f%%)" % (missing_values, missing_values * 100 / num_instances))


def render_countplot(current_feature, selected_feature, dataframe):
    if current_feature.nunique() < 30:
        count_series = current_feature.value_counts()
        count_series.rename("Count", inplace=True)
        st.write(count_series)
        chart = sns.countplot(x=selected_feature, data=dataframe)
        if current_feature.dtype.name == "object":
            plt.gcf().subplots_adjust(bottom=0.25)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        st.pyplot()


def render_scatterplot(dataframe, feature):
    st.write("Correlation with label:")
    df_columns = list(dataframe.columns.values)
    label_name = df_columns[len(df_columns) - 1]
    fig = px.scatter(dataframe, feature, label_name, color=label_name)
    st.plotly_chart(fig)


def render_boxplot(dataframe, feature):
    if dataframe[feature].nunique() < 30:
        st.write("Correlation with label:")
        df_columns = list(dataframe.columns.values)
        label_name = df_columns[len(df_columns) - 1]

        chart = sns.boxplot(x=feature, y=label_name, data=dataframe)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        st.pyplot()