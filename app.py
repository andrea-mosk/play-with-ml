import streamlit as st
import dataframefunctions
import dataexploration
import runpredictions
from PIL import Image
import tutorial

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
            tutorial.display_info(uploaded=True)
            tutorial.display_warnings()
        elif selected_option == "Data exploration":
            dataexploration.load_page(dataframe)
        elif selected_option == "Run predictions":
            runpredictions.load_page(dataframe)

    else:
        tutorial.display_info(uploaded=False)
        tutorial.display_warnings()


if __name__ == "__main__":
    main()










