import streamlit as st
import pandas as pd
import renderfunctions
import dataframefunctions
import dataexploration
import runpredictions
from InteractiveWrangling import Wrangler
from dumbClass import DumbClass
import time


SIDEBAR_WHAT_TODO = ["Data exploration", "Run predictions", "Interactive wrangling"]


def main():
    uploaded_file = st.sidebar.file_uploader("Please upload your dataset:", type='csv')
    is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
    if uploaded_file is not None:
        is_loaded_dataset.success("Dataset uploaded successfully!")
        try:
            dataframe = dataframefunctions.get_dataframe(uploaded_file)
            dataframe_upl = dataframe.copy()
        except:
            is_loaded_dataset.error("The imported dataset can't be read from pandas, please try again.")
        selected_option = st.sidebar.selectbox("What to do?", SIDEBAR_WHAT_TODO)
        if selected_option == "Data exploration":
            dataexploration.render_data_explorations(dataframe)
        elif selected_option == "Run predictions":
            runpredictions.load_page(dataframe)
        elif selected_option == "Interactive wrangling":
            dataexploration.wrangl(dataframe)



if __name__ == "__main__":
    main()










