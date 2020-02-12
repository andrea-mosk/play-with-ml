from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import preprocessing


POSSIBLE_MODEL = ["XGBoost (classifier)",
                  "XGBoost (regressor)",
                  "Random Forest",
                  "Support Vector Machine",
                  "K-nearest neighbors"]
KERNEL_OPTIONS = ['Rbf', 'Linear', 'Poly', 'Sigmoid']
EVALUATION_METRICS = ["Accuracy", "RMSE", "F1", "Precision", "Recall", "MSE"]
WEIGHT_FUNCTION_OPTION = ["Uniform", "Distance"]
ALGORITHM = ["Auto", "Ball tree", "Kd tree", "Brute"]


def load_page(dataframe):
    st.sidebar.subheader("Experiment parameters:")
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    evaluation_metrics = st.sidebar.multiselect("Select the evaluation metrics",
                                                EVALUATION_METRICS)
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    model_parameters = display_hyperparameters(selected_model)
    display_experiment_stats(dataframe, test_size, selected_model)
    if st.button("Run predictions"):
        if len(evaluation_metrics) == 0:
            st.error("Please select at least one evaluation metric!")
        else:
            run_predictions(dataframe, test_size, selected_model, model_parameters, set(evaluation_metrics))


def run_predictions(dataframe, test_size, selected_model, parameters, metrics):
    """Puts together preprocessing, training and testing."""

    st.markdown(":chart_with_upwards_trend: Hyperparameters used: ")
    st.write(parameters)

    # Preprocessing data
    with st.spinner("Preprocessing data.."):
        x, y = preprocessing.preprocess(dataframe)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Training the model
    with st.spinner("Training model.."):
        model = get_trained_model(X_train, y_train, selected_model, parameters)
        if model is None:
            return

    # Testing the model
    with st.spinner("Testing model.."):
        test_model(model, X_train, y_train, X_test, y_test, metrics)


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
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 20, 3)
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)

    if selected_model == "XGBoost (regressor)":
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
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

    return model.fit(x, y)


def test_model(model, X_train, y_train, X_test, y_test, metrics):
    """Tests the model predictions based on the chosen metrics."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if "RMSE" in metrics:
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
