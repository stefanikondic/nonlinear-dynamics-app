import streamlit as st

from utils.helpers import parse_parameter_value


def get_parameter_value(param_name, default="1.0"):
    value_str = st.text_input(
        f"{param_name}",
        value=default,
        key=f"{param_name}_text",
    )
    return parse_parameter_value(value_str, param_name)
