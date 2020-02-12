import streamlit as st
import dataframefunctions
import dataexploration
import runpredictions
import PIL


SIDEBAR_WHAT_TODO = ["Tutorial", "Data exploration", "Run predictions"]


def main():
    """Load the dataset and initializes the app if the dataset is not none."""

    uploaded_file = st.sidebar.file_uploader("Please upload your dataset:", type='csv')
    is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
    if uploaded_file is not None:
        is_loaded_dataset.success("Dataset uploaded successfully!")
        try:
            dataframe = dataframefunctions.get_dataframe(uploaded_file)
            selected_option = st.sidebar.selectbox("What to do?", SIDEBAR_WHAT_TODO)
            if selected_option == "Data exploration":
                dataexploration.load_page(dataframe)
            elif selected_option == "Run predictions":
                runpredictions.load_page(dataframe)
            elif selected_option == "Tutorial":
                display_info(uploaded=True)
        except:
            is_loaded_dataset.error("The uploaded dataset can't be read from pandas, please try again.")
    else:
        display_info(uploaded=False)


def display_info(uploaded):
    st.write("# Welcome to PLAY WITH ML!")

    if not uploaded:
        st.write("To start, please upload your dataset into the uploader in the sidebar.")

    else:
        st.write("You have uploaded your dataset! Now, you can go into one of these two sections:")
        st.write("**1. Data exploration** This section allows you to take a first look at the dataset \
                 and discover the main statistics, such as number of instances, number of features, and missing data. \
                 Furthermore, you can easily make plots to discover potential correlations between the features.")
        st.write("**2. Run Predictions.** This section allows you to tr")
    

if __name__ == "__main__":
    main()










