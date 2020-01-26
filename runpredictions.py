import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import dataframefunctions
import pandas as pd
import streamlit as st
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from math import log


POSSIBLE_MODEL = ["XGBoost", "Random Forest", "Support Vector Machine", "K-nearest neighbors"]
KERNEL_OPTIONS = ['Rbf', 'Linear', 'Poly', 'Sigmoid']
INDEX_COLUMNS = ["ID", "id", "Id", "iD", "INDEX", "Index", "index"]
EVALUATION_METRICS = ["Accuracy", "RMSE", "F1", "Precision", "Recall", "MSE"]
WEIGHT_FUNCTION_OPTION = ["Uniform", "Distance"]
ALGORITHM = ["Auto", "Ball tree", "Kd tree", "Brute"]


def load_page(dataframe):
    st.sidebar.subheader("Experiment parameters:")
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    evaluation_metrics = st.sidebar.multiselect("Select the evaluation metrics",
                                                EVALUATION_METRICS,
                                                default=["Accuracy"])
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    model_parameters = display_hyperparameters(selected_model)
    display_experiment_stats(dataframe, test_size, selected_model)
    if st.button("Run predictions"):
        if len(evaluation_metrics) == 0:
            st.error("Please select at least one evaluation metric!")
        else:
            run_predictions(dataframe, test_size, selected_model, model_parameters, set(evaluation_metrics))


def run_predictions(dataframe, test_size, model, parameters, metrics):
    st.markdown(":chart_with_upwards_trend: Hyperparameters used: ")
    st.write(parameters)
    with st.spinner("Preprocessing data.."):
        x, y = preprocess_dataset(dataframe)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    with st.spinner("Training model.."):
        model = get_trained_model(X_train, y_train, model, parameters)
        if model is None:
            return
    with st.spinner("Testing model.."):
        test_model(model, X_train, y_train, X_test, y_test, metrics)



def display_experiment_stats(dataframe, test_size, selected_model):
    num_instances = dataframe.shape[0]
    training_instances = round(num_instances * (1 - test_size), 0)
    test_instances = round(num_instances * test_size, 0)
    st.write("Running **%s** with a test set size of **%d%%**." % (selected_model, round(test_size * 100, 0)))
    st.write("There are **%d** instances in the training set and **%d** instances in the test set." %
             (training_instances, test_instances))


def drop_label_and_index(dataframe):
    columns = list(dataframe.columns)
    label_name = columns[-1]
    x = dataframe.drop(label_name, axis=1)
    set_columns = set(columns)
    for index_column in INDEX_COLUMNS:
        if index_column in set_columns:
            x.drop(index_column, axis=1, inplace=True)
    return x, label_name


def preprocess_dataset(dataframe):
    x, label_name = drop_label_and_index(dataframe)
    y = dataframe[label_name].copy()

    x_prepared = standard_data_preprocessing(x)
    y_prepared = y \
        if not dataframefunctions.is_categorical(y) \
        else standard_label_preprocessing(y, dataframe, label_name)

    return x_prepared, y_prepared


def standard_data_preprocessing(train_data):
    numerical_att = dataframefunctions.get_numeric_columns(train_data)
    categorical_att = dataframefunctions.get_categorical_columns(train_data)

    # full_pipeline = get_full_pipeline(numerical_att, categorical_att)
    # return full_pipeline.fit_transform(train_data)

    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(train_data[numerical_att])
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_att, index=train_data.index)

    return scaled_df


def standard_label_preprocessing(y, dataframe, label_name):
    lc = LabelEncoder().fit(y)
    y_encoded = lc.transform(y)
    y_prepared = pd.DataFrame(y_encoded, columns=[label_name], index=dataframe.index)
    return y_prepared


def display_hyperparameters(selected_model):
    hyperparameters = {}
    st.sidebar.subheader("Model parameters:")

    if selected_model == "XGBoost":
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 20, 3)
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.01)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)

    elif selected_model == "Random Forest":
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 200, 100)
        hyperparameters['min_samples_split'] = st.sidebar.slider("Min. samples  split", 2, 20, 2)
        hyperparameters['criterion'] = st.sidebar.selectbox("Select the criteria", ['Gini', 'Entropy']).lower()
        hyperparameters['min_samples_leaf'] = st.sidebar.slider("Min. samples  leaf", 1, 50, 1)

    elif selected_model == "Support Vector Machine":
        hyperparameters['C'] = st.sidebar.number_input('Regularization', min_value=1.0, max_value=50.0, value=1.0)
        hyperparameters['kernel'] = st.sidebar.selectbox("Select the kernel", KERNEL_OPTIONS).lower()
        hyperparameters['gamma'] = st.sidebar.radio("Select the kernel coefficient", ['Scale', 'Auto']).lower()

    elif selected_model == "K-nearest neighbors":
        hyperparameters['n_neighbors'] = st.sidebar.slider("Num. neighbors", 1, 50, 5)
        hyperparameters['weights'] = st.sidebar.selectbox("Select the weight function", WEIGHT_FUNCTION_OPTION).lower()
        hyperparameters['algorithm'] = st.sidebar.selectbox("Select the algorithm", ALGORITHM).lower().replace(" ", "_")

    return hyperparameters


def get_trained_model(x, y, model, parameters):
    if model == "XGBoost":
        model = XGBClassifier(**parameters)
        return train_model(x, y, model)
    elif model == "Random Forest":
        model = RandomForestClassifier(**parameters)
        return train_model(x, y, model)
    elif model == "Support Vector Machine":
        model = SVC(**parameters)
        return train_model(x, y, model)
    elif model == "K-nearest neighbors":
        model = KNeighborsClassifier(**parameters)
        return train_model(x, y, model)


def train_model(x, y, model):
    # if y.nunique().values[0] == 1:
    #     st.error(":warning: There is only **one** class in the training set. " +
    #              "Please increase the size of the training set or adjust the dataset accordingly.")
    #     return
    trained_model = model.fit(x, y)
    return trained_model


def test_model(model, X_train, y_train, X_test, y_test, metrics):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if "RMSE" in metrics:
        # train_mse = mean_squared_error(get_log(y_train), get_log(model.predict(X_train)))
        # test_mse = mean_squared_error(get_log(y_test), get_log(model.predict(X_test)))
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        st.write("Train rmse: %.4f" % (np.sqrt(train_mse)))
        st.write("Test rmse: %.4f" % (np.sqrt(test_mse)))

    if "Accuracy" in metrics:
        st.write("Train accuracy: %.2f%%" % (accuracy_score(y_train, y_train_pred) * 100.0))
        st.write("Test accuracy: %.2f%%" % (accuracy_score(y_test, y_test_pred) * 100.0))

    if "F1" in metrics:
        st.write("Train F1-score: %.2f%%" % (f1_score(y_train, y_train_pred, average='micro') * 100.0))
        st.write("Test F1-score: %.2f%%" % (f1_score(y_test, y_test_pred, average='micro') * 100.0))

    if "Precision" in metrics:
        st.write("Train precision: %.2f%%" % (precision_score(y_train, y_train_pred, average='micro') * 100.0))
        st.write("Test precision: %.2f%%" % (precision_score(y_test, y_test_pred, average='micro') * 100.0))

    if "Recall" in metrics:
        st.write("Train recall: %.2f%%" % (recall_score(y_train, y_train_pred, average='micro') * 100.0))
        st.write("Test recall: %.2f%%" % (recall_score(y_test, y_test_pred, average='micro') * 100.0))

    if "MSE" in metrics:
        st.write("Train mse: %.4f" % mean_squared_error(y_train, y_train_pred))
        st.write("Test mse: %.4f" % mean_squared_error(y_test, y_test_pred))



def get_log(list):
    return [log(elem, 10) for elem in list]