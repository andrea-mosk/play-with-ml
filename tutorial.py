import streamlit as st
from PIL import Image


def display_info(uploaded):
    st.write("# Welcome to Play With ML!")

    if not uploaded:
        st.write("**To start, please upload your dataset into the uploader in the sidebar.**"
                 " After uploading, you can go into one of these sections:")

    else:
        st.write("You have uploaded your dataset! Now, you can go into one of these two sections:")

    st.write("**1. Data exploration.** This section allows you to take a first look at the dataset \
                 and discover the main statistics, such as number of instances, number of features, and missing data. \
                 Furthermore, you can easily make plots to discover potential correlations between features very quickly.")
    st.write("For example, the figure below shows the relationship between the ground living area square feet \
                 and the price of the houses in the training set of Kaggle's public competition \
                  [House Prices: Advanced Regression Techniques] \
                 (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).")
    plot_image = Image.open("Images/house_train_plot2.PNG")
    st.image(plot_image, use_column_width=True)
    st.write(" ")
    st.write("**2. Run Predictions.** This section allows you to train and test, on your dataset, \
             one of the available machine learning models. The available models include XGBoost, Random forests, \
             Support vector machines, and so on. During the experiment, you can also adjust the experiment parameters \
             (such as the size of the test set) and the hyperparameters of the model selected (such as the number \
             of estimators in XGBoost). The dataset will automatically be preprocessed in order to handle missing \
             data, drop useless rows, scale features and so on.")
    video_file = open('Videos/RunPredictions_recording.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def display_warnings():
    st.markdown("## **Warning  :warning:** ##")
    st.markdown("To make the app work, there are some constraints on the dataset's structure:")
    st.markdown("* The last attribute of the dataset must be the **label** and should not contain missing values.")
