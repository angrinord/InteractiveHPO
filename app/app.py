import streamlit as st

from utils.strings import S
from .sidebar import sidebar
from .experiment_form import new_form
from .experiment import experiment


def start_app(models, optimizers, metrics):
    st.set_page_config(page_title=S("page_title"), layout="wide")
    st.session_state.setdefault("_models", models)
    st.session_state.setdefault("_optimizers", optimizers)
    st.session_state.setdefault("_metrics", metrics)
    sidebar()

    if st.session_state.creating:
        new_form(models, optimizers, metrics)
    elif st.session_state.active in st.session_state.experiments:
        experiment(st.session_state.active)
    else:
        st.title(S("home_title"))
        st.write(S("home_body"))
