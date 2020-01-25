import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
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

POSSIBLE_MODEL = ["XGBoost", "Random Forest", "Support Vector Machine"]
KERNEL_OPTIONS = ['Rbf', 'Linear', 'Poly', 'Sigmoid']
INDEX_COLUMNS = ["ID", "id", "Id", "iD", "INDEX", "Index", "index"]


def load_page(dataframe):
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    st.sidebar.subheader("Experiment parameters:")
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    model_parameters = display_hyperparameters(selected_model)
    display_experiment_stats(dataframe, test_size, selected_model)
    if st.button("Run predictions"):
        run_predictions(dataframe, test_size, selected_model, model_parameters)


def run_predictions(dataframe, test_size, model, parameters):
    st.markdown(":chart_with_upwards_trend: Hyperparameters used: ")
    st.write(parameters)
    with st.spinner("Preprocessing data.."):
        x, y = preprocess_dataset(dataframe)



def display_experiment_stats(dataframe, test_size, selected_model):
    num_instances = dataframe.shape[0]
    training_instances = round(num_instances * (1 - test_size), 0)
    test_instances = round(num_instances * test_size, 0)
    st.write("Running **%s** with a test set size of **%d%%**." % (selected_model, round(test_size * 100, 0)))
    st.write("There are **%d** instances in the training set and **%d** instances in the test set." %
             (training_instances, test_instances))


def render_run_predictions(dataframe):
    label_name = list(dataframe.columns)[-1]
    x_prepared, y_prepared = preprocess_dataset(dataframe, label_name)
    st.write(x_prepared.head())
    st.write(y_prepared.head())
    # x_dataset = pd.DataFrame(x_prepared, columns=dataframe.columns[:-1], index=dataframe.index)
    y_dataset = pd.DataFrame(y_prepared, columns=[label_name], index=dataframe.index)
    make_predictions(x_prepared, y_dataset)


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
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_att)

    return scaled_df


def standard_label_preprocessing(y, dataframe, label_name):
    lc = LabelEncoder().fit(y)
    y_encoded = lc.transform(y)
    y_prepared = pd.DataFrame(y_encoded, columns=[label_name], index=dataframe.index)
    return y_prepared


def make_predictions(x_dataset, y_dataset):
    X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=test_size, random_state=42)
    if st.button("Run predictions"):
        st.write("Hyperparameters used: ", model_parameters)
        if selected_model == "XGBoost":
            model = XGBClassifier(**model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(**model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)
        elif selected_model == "Support Vector Machine":
            model = SVC(**model_parameters)
            model_train_predict(model, X_train, y_train, X_test, y_test)


def model_train_predict(model, x_train, y_train, x_test, y_test):
    # If there is only one class in the training set, return
    if y_train.nunique().values[0] == 1:
        st.error(":warning: There is only **one** class in the training set. " +
                 "Please increase the size of the training set or adjust the dataset accordingly.")
        return
    with st.spinner('Training the model..'):
        model.fit(x_train, y_train)
    with st.spinner('Predicting..'):
        y_pred = model.predict(x_test)
        # predictions = [round(value) for value in y_pred]

        # st.write("Train accuracy: %.2f%%" % (accuracy_score(y_train, model.predict(x_train)) * 100.0))
        # st.write("Test accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100.0))

        test_mse = mean_squared_error(y_test, y_pred)
        st.write("Test rmse: %.4f" % (np.sqrt(test_mse)))


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

    return hyperparameters


def get_full_pipeline(numerical_att, categorical_att):
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

    return full_pipeline