import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.random_forest import RandomForestModel
from optimizers.smac_optimizer import SMACOptimizer
from strings import get_strings
from app.app import start_app

MODELS = {
    "Random Forest": RandomForestModel(),
}

OPTIMIZERS = {
    "SMAC": SMACOptimizer(),
}

METRICS = {
    "accuracy":         lambda y, yp: accuracy_score(y, yp),
    "f1":               lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
    "precision":        lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
    "recall(macro)":    lambda y, yp: recall_score(y, yp, average="macro", zero_division=0),
}

def _init_state():
    st.session_state.setdefault("experiments", {})
    st.session_state.setdefault("active", None)
    st.session_state.setdefault("creating", False)
    st.session_state.setdefault("locale", "en")

def run():
    _init_state()
    S = get_strings(st.session_state.locale)
    start_app(S, MODELS, OPTIMIZERS, METRICS)

if __name__ == "__main__":
    run()
