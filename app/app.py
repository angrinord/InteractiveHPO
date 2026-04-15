import streamlit as st

from app.sidebar import sidebar
from app.experiment_form import new_form
from app.experiment import experiment


def start_app(S, models, optimizers, metrics):
    st.set_page_config(page_title=S["page_title"], layout="wide")
    sidebar(S)

    if st.session_state.creating:
        new_form(S, models, optimizers, metrics)
    elif st.session_state.active in st.session_state.experiments:
        experiment(st.session_state.active, S)
    else:
        st.title(S["home_title"])
        st.write(S["home_body"])
