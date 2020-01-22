import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import dataframefunctions
import pandas as pd
import streamlit as st

POSSIBLE_MODEL = ["XGBoost", "Random Forest", "Support Vector Machine"]


def render_run_predictions(dataframe):
    label_name = list(dataframe.columns)[-1]
    x_prepared, y_prepared = preprocess_dataset(dataframe, label_name)
    x_dataset = pd.DataFrame(x_prepared, columns=dataframe.columns[:-1], index=dataframe.index)
    y_dataset = pd.DataFrame(y_prepared, columns=[label_name], index=dataframe.index)
    make_predictions(x_dataset, y_dataset)


def preprocess_dataset(dataframe, label_name):

    x = dataframe.drop(label_name, axis=1)
    y = dataframe[label_name].copy()

    x_prepared = standard_data_preprocessing(x)
    y_prepared = y if not dataframefunctions.is_categorical(y) else standard_label_preprocessing(y)

    return x_prepared, y_prepared


def standard_data_preprocessing(train_data):

    numerical_att = dataframefunctions.get_numeric_columns(train_data)
    categorical_att = dataframefunctions.get_categorical_columns(train_data)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('oneHot', OneHotEncoder()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numerical_att),
        ("cat", cat_pipeline, categorical_att),
    ])

    return full_pipeline.fit_transform(train_data)


def standard_label_preprocessing(y):
    lc = LabelEncoder().fit(y)
    return lc.transform(y)


def make_predictions(x_dataset, y_dataset):
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    test_size = st.sidebar.slider('Test set size', 0.0, 1.0, 0.2)
    st.write("Running %s with a test set size of %d%%" % (selected_model, int(test_size*100)))
    if st.button("Run predictions"):
        if selected_model == "XGBoost":
            run_xgboost_predictions(test_size, x_dataset, y_dataset)
        elif selected_model == "Random Forest":
            run_rfc_predictions(test_size, x_dataset, y_dataset)
        elif selected_model == "Support Vector Machine":
            run_svm_predictions(test_size, x_dataset, y_dataset)


# TODO
def run_xgboost_predictions(test_size, x_dataset, y_dataset):
    st.write("xgboost")



# TODO
def run_rfc_predictions(test_size, x_dataset, y_dataset):
    st.write("RFC")


# TODO
def run_svm_predictions(test_size, x_dataset, y_dataset):
    st.write("SVM")