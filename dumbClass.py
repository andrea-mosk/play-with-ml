import random
import streamlit as st


class DumbClass:

    value = 0

    def __init__(self):
        self.value = random.random()
        st.balloons()

    def useless_box(self):
        new_value = st.slider("Select a value", 0, 100)
        st.write("the selected value is %d" % new_value)
