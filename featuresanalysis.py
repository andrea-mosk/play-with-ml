import streamlit as st
import dataframefunctions
import seaborn as sns

def load_page(dataframe):
    selected_feature = st.selectbox("Which feature?", list(dataframe.columns.values))
    num_instances = dataframe.shape[0]
    current_feature = dataframe[selected_feature]
    if dataframefunctions.is_categorical(current_feature):
        describe_categorical_feature(current_feature, num_instances)
        render_countplot(current_feature, selected_feature, dataframe)
    else:
        describe_numerical_feature(current_feature, num_instances)
        render_countplot(current_feature, selected_feature, dataframe)


# TODO solve problem with ID
def describe_numerical_feature(feature, num_instances):
    st.markdown("**Mean**: %.2f, &nbsp; &nbsp; &nbsp; **Median**: %.2f" % (feature.mean(), feature.median()))
    st.markdown("**Max**: %.2f, &nbsp; &nbsp; &nbsp; **Min**: %.2f" % (feature.max(), feature.min()))
    st.write("95th percentile: %.2f" % (feature.quantile(.95)))

    describe_common_statistics(feature, num_instances)

    sns.distplot(feature)
    st.pyplot()
    # feature.value_counts()


def describe_categorical_feature(column, num_instances):
    # further divide by number of different values
    describe_common_statistics(column, num_instances)


def describe_common_statistics(feature, num_instances):
    st.write("**Memory usage**: %.2f Bytes" % (feature.memory_usage()))
    st.write("**Unique elements**: %d (%.2f%%)" % (feature.nunique(), feature.nunique()*100/num_instances))
    st.write("**Most popular element**: %s" % (feature.mode()[0]))
    missing_values = feature.isna().sum()
    st.write("**Missing values**: %d (%.2f%%)" % (missing_values, missing_values * 100 / num_instances))


def render_countplot(current_feature, selected_feature, dataframe):
    if current_feature.nunique() < 20:
        sns.countplot(x=selected_feature, data=dataframe)
        st.pyplot()