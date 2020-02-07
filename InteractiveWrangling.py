import streamlit as st
import time

class Wrangler:

    dataframe = None
    POSSIBLE_COMMANDS = ["Drop", "Encode"]
    data_placeholder = None

    def __init__(self, input_dataframe):
        st.write(self.dataframe)
        self.dataframe = input_dataframe.copy()
        self.dataframe_placeholder = st.empty()
        self.modify_dataframe(input_dataframe)

    def display_dataset(self, input_dataframe):
        self.dataframe_placeholder.dataframe(input_dataframe.head(10))


    def modify_dataframe(self, input_dataset):
        current_dataset = input_dataset.copy()
        self.display_dataset(current_dataset)
        selected_command = st.selectbox("What to do?", ["Drop", "Encode"])
        if st.button("Drop ID"):
            time.sleep(5)
            current_dataset.drop("Id", axis=1, inplace=True)
            self.display_dataset(current_dataset)


    def display_commands(self):
        selected_command = st.selectbox("What to do?", self.POSSIBLE_COMMANDS)
        options = self.display_options(selected_command)
        if st.button(selected_command):
            if selected_command == "Drop":
                self.dataframe.drop(options["features"], axis=1, inplace=True)
                self.display_dataset()
            else:
                st.write(selected_command)

    def drop(self):
        self.dataframe.drop("Id", axis=1, inplace=True)
        self.display_dataset()

    def display_options(self, selected_command):
        options = {}
        if selected_command == "Drop":
            options["features"] = st.multiselect("Select the feature(s) to drop", self.dataframe.columns)

        if selected_command == "Encode":
            options["feature"] = st.selectbox("Select the feature to encode", self.dataframe.columns)

        return options


