import streamlit as st
import dataframefunctions
import dataexploration
import runpredictions
from PIL import Image


SIDEBAR_WHAT_TODO = ["Tutorial & info", "Data exploration", "Run predictions"]


def main():
    """Load the dataset and initializes the app if the dataset is not none."""

    uploaded_file = st.sidebar.file_uploader("Please upload your dataset (CSV format):", type='csv')
    is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
    if uploaded_file is not None:
        is_loaded_dataset.success("Dataset uploaded successfully!")
        dataframe = dataframefunctions.get_dataframe(uploaded_file)
        st.sidebar.markdown("## Navigation ##")
        selected_option = st.sidebar.radio("Go to:", SIDEBAR_WHAT_TODO)
        if selected_option == "Tutorial & info":
            display_info(uploaded=True)
            display_warnings()
        elif selected_option == "Data exploration":
            dataexploration.load_page(dataframe)
        elif selected_option == "Run predictions":
            runpredictions.load_page(dataframe)

    else:
        display_info(uploaded=False)


def display_info(uploaded):
    st.write("# Welcome to Play With ML!")

    if not uploaded:
        st.write("**To start, please upload your dataset into the uploader in the sidebar.**")

    else:
        st.write("You have uploaded your dataset! Now, you can go into one of these two sections:")

    st.write("**1. Data exploration.** This section allows you to take a first look at the dataset \
                 and discover the main statistics, such as number of instances, number of features, and missing data. \
                 Furthermore, you can easily make plots to discover potential correlations between the features.")
    st.write("For example, the figure below shows the relationship between the ground living area square feet \
                 and the price of the houses in the training set of Kaggle's public competition \
                  [House Prices: Advanced Regression Techniques] \
                 (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).")
    plot_image = Image.open("Images/house_train_plot2.PNG")
    st.image(plot_image,
             use_column_width=True)
    st.write(" ")
    st.write("**2. Run Predictions.** This section allows you to train and test, on your dataset, \
                 one of the available model. The available models include XGBoost, Random forests, \
                 Support vector machines, and so on. During the experiment, you can also adjust the experiment \
                 (such as the size of the test set) and the hyperparameters of the model selected (such as the number \
                 of estimators in XGBoost). The dataset will automatically be preprocessed in order to handle missing \
                 data, drop useless rows, scale the features and so on.")
    video_file = open('Videos/RunPredictions_recording.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def display_warnings():
    st.markdown("## Warning  :warning: ##")
    st.markdown("To make the app work, there are some constraints on the dataset's structure:")
    st.markdown("* The last attribute of the dataset must be the **label** and should not contain missing values.")
    

if __name__ == "__main__":
    main()










