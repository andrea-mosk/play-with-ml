import streamlit as st
import pandas as pd
import renderfunctions
import dataframefunctions
import dataexploration
import runpredictions


SIDEBAR_WHAT_TODO = ["Data exploration", "Run predictions"]


def main():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset and see all the statistics!", type='csv')
    if uploaded_file is not None:
        dataframe = dataframefunctions.get_dataframe(uploaded_file)
        selected_option = st.sidebar.selectbox("What to do?", SIDEBAR_WHAT_TODO)
        if selected_option == "Data exploration":
            dataexploration.render_data_explorations(dataframe)
        elif selected_option == "Run predictions":
            runpredictions.render_run_predictions(dataframe)


if __name__ == "__main__":
    main()










