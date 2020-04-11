import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd


POSSIBLE_MODEL = ["XGBoost (classifier)",
                  "XGBoost (regressor)",
                  "Random Forest",
                  "Support Vector Machine",
                  "K-nearest neighbors"]
KERNEL_OPTIONS = ['Rbf', 'Linear', 'Poly', 'Sigmoid']
EVALUATION_METRICS = ["Accuracy", "RMSE", "F1", "Precision", "Recall", "MSE"]
WEIGHT_FUNCTION_OPTION = ["Uniform", "Distance"]
ALGORITHM = ["Auto", "Ball tree", "Kd tree", "Brute"]
metrics2string = {"MSE": "neg_mean_squared_error",
                  "RMSE": "neg_mean_squared_error"}


def load_page(dataframe):
    """Loading the initial page, displaying the experiment parameters and the model's hyperparameters."""

    st.sidebar.subheader("Experiment parameters:")
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    evaluation_metrics = st.sidebar.multiselect("Select the evaluation metrics", EVALUATION_METRICS)
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    cross_val = st.sidebar.checkbox("Cross validation")
    if cross_val:
        cv_k = st.sidebar.number_input("Please select the cross-validation K:",
                                       min_value=1,
                                       value=10,
                                       max_value=dataframe.shape[0])

    model_parameters = display_hyperparameters(selected_model)
    display_experiment_stats(dataframe, test_size, selected_model)
    if st.button("Run predictions"):
        if len(evaluation_metrics) == 0:
            st.error("Please select at least one evaluation metric!")
        else:
            run_predictions(dataframe,
                            test_size,
                            selected_model,
                            model_parameters,
                            evaluation_metrics,
                            cross_val,
                            cv_k if cross_val else None)


def run_predictions(dataframe, test_size, selected_model, parameters, metrics, cross_val, cv_k):
    """Puts together preprocessing, training and testing."""

    st.markdown(":chart_with_upwards_trend: Hyperparameters used: ")
    st.write(parameters)

    if cross_val:
        st.warning("Warning, only the first metric is selected when using Cross Validation.")

    # Preprocessing data
    x, y = preprocessing.preprocess(dataframe)
    st.success("Preprocessing completed!")

    model = get_model(selected_model, parameters)

    if cross_val:
        # model = get_model(selected_model, parameters)
        cross_validation(model, x, y, cv_k, metrics[0])

    else:
        # Training the model
        train_status = st.warning("Training model..")
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        # model = get_model(selected_model, parameters)
        model.fit(X_train, y_train)
        train_status.success("Training completed!")

        # Testing the model
        test_status = st.warning("Testing model..")
        test_model(model, X_train, y_train, X_test, y_test, metrics)
        test_status.success("Testing completed!")


def cross_validation(model, x, y, cv_k, metric):
    """Training and testing using cross validation."""

    current_status = st.warning("Training and testing model..")
    right_metric = metrics2string.get(metric, metric.lower())
    results = cross_validate(model, x, y, cv=cv_k, scoring=right_metric)
    scores = results.get("test_score")

    # When MSE or RMSE are used as metrics, the results are negative
    if metric == "MSE":
        scores = scores * -1

    if metric == "RMSE":
        scores = scores * -1
        scores = np.sqrt(scores)

    current_status.success("Training and testing completed!")
    evaluation = pd.DataFrame([[scores.mean(), scores.std()]],
                              columns=[[metric, metric], ["Mean", "Standard deviation"]],
                              index=["Dataset"])
    st.dataframe(evaluation)


def display_experiment_stats(dataframe, test_size, selected_model):
    """Displays the experiment input, e.g. test set size"""

    num_instances = dataframe.shape[0]
    training_instances = round(num_instances * (1 - test_size), 0)
    test_instances = round(num_instances * test_size, 0)
    st.write("Running **%s** with a test set size of **%d%%**." % (selected_model, round(test_size * 100, 0)))
    st.write("There are **%d** instances in the training set and **%d** instances in the test set." %
             (training_instances, test_instances))


def display_hyperparameters(selected_model):
    """Display the possible hyperparameters of the model chosen by the user."""

    hyperparameters = {}
    st.sidebar.subheader("Model parameters:")

    if selected_model == "XGBoost (classifier)":
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 20, 3)

    if selected_model == "XGBoost (regressor)":
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 100, 6)

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


def get_model(model, parameters):
    """Creates and trains a new model based on the user's input."""

    if model == "XGBoost (classifier)":
        model = XGBClassifier(**parameters)

    elif model == "Random Forest":
        model = RandomForestClassifier(**parameters)

    elif model == "Support Vector Machine":
        model = SVC(**parameters)

    elif model == "K-nearest neighbors":
        model = KNeighborsClassifier(**parameters)

    elif model == "XGBoost (regressor)":
        model = XGBRegressor(**parameters)

    return model


def test_model(model, X_train, y_train, X_test, y_test, metrics):
    """Tests the model predictions based on the chosen metrics."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics_data = {}

    if "RMSE" in metrics:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        metrics_data["RMSE"] = [np.sqrt(train_mse), np.sqrt(test_mse)]

    if "Accuracy" in metrics:
        train_accuracy = accuracy_score(y_train, y_train_pred) * 100.0
        test_accuracy = accuracy_score(y_test, y_test_pred) * 100.0
        metrics_data["Accuracy"] = [train_accuracy, test_accuracy]

    if "F1" in metrics:
        f1_train = f1_score(y_train, y_train_pred, average='micro') * 100.0
        f1_test = f1_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["F1-Score"] = [f1_train, f1_test]

    if "Precision" in metrics:
        precision_train = precision_score(y_train, y_train_pred, average='micro') * 100.0
        precision_test = precision_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["Precision"] = [precision_train, precision_test]

    if "Recall" in metrics:
        recall_train = recall_score(y_train, y_train_pred, average='micro') * 100.0
        recall_test = recall_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["Recall"] = [recall_train, recall_test]

    if "MSE" in metrics:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        metrics_data["MSE"] = [train_mse, test_mse]

    evaluation = pd.DataFrame(metrics_data, index=["Train", "Test"])
    st.write(evaluation)
