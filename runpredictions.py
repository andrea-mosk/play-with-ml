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
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    model_parameters = display_hyperparameter(selected_model)
    X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=test_size, random_state=42)
    st.write("Running **%s** with a test set size of **%d%%**." % (selected_model, int(test_size*100)))
    st.write("There are **%d** instances in the training set and **%d** instances in the test set." % (len(X_train), len(X_test)))
    if st.button("Run predictions"):
        if selected_model == "XGBoost":
            model = XGBClassifier(**model_parameters)
            st.write(model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(**model_parameters)
            st.write("Hyperparameters: ", model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)
        elif selected_model == "Support Vector Machine":
            model = SVC(**model_parameters)
            st.write(model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)


def model_train_predict(model, x_train, y_train, x_test, y_test):
    with st.spinner('Training the model..'):
        model.fit(x_train, y_train)
    with st.spinner('Predicting..'):
        y_pred = model.predict(x_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        st.write("Accuracy: %.2f%%" % (accuracy * 100.0))


def display_hyperparameter(selected_model):
    hyperparameters = {}
    if selected_model == "XGBoost":
        hyperparameters.clear()
    elif selected_model == "Random Forest":
        hyperparameters.clear()
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 200, 100)
        hyperparameters['min_samples_split'] = st.sidebar.slider("Min. samples  split", 1, 20, 2)
    elif selected_model == "Support Vector Machine":
        hyperparameters.clear()
    return hyperparameters