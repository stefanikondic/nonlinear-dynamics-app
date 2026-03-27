import streamlit as st


def apply_app_styles():
    st.markdown(
        """
        <style>
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1050px;
        }

        h1 {
            margin-bottom: 0.2rem;
        }

        h2, h3 {
            margin-top: 0.6rem;
            margin-bottom: 0.3rem;
        }

        div[data-testid="stTextInput"] {
            margin-bottom: 0.35rem;
        }

        div[data-testid="stNumberInput"] {
            margin-bottom: 0.35rem;
        }

        div[data-testid="stTextArea"] {
            margin-bottom: 0.35rem;
        }

        div[data-testid="stSlider"] {
            margin-bottom: 0.2rem;
        }

        div[data-testid="stRadio"] {
            margin-bottom: 0.2rem;
        }

        div[data-testid="stCheckbox"] {
            margin-bottom: -0.2rem;
        }

        .stButton > button {
            width: 180px;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.55rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
