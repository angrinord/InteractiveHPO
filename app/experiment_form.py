import random

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


def new_form(S, models, optimizers, metrics):
    st.header(S["new_experiment_header"])

    with st.form("new_exp"):
        name      = st.text_input(S["field_experiment_name"])
        csv_file  = st.file_uploader(S["field_dataset"], type="csv")
        model_key  = st.selectbox(S["field_model"], list(models.keys()))
        opt_key    = st.selectbox(S["field_optimizer"], list(optimizers.keys()))
        seed_input = st.number_input(S["field_seed"], value=0, step=1)
        submitted  = st.form_submit_button(S["btn_create"])

    if not submitted:
        return

    if not name:
        st.error(S["err_name_required"])
        return
    if name in st.session_state.experiments:
        st.error(S["err_name_exists"].format(name=name))
        return
    if csv_file is None:
        st.error(S["err_dataset_required"])
        return

    # Negative seed_input means random; sample once so every run of this
    # experiment is reproducible against itself (including SMAC resume hashing).
    seed = random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)

    sample = csv_file.read(2048).decode("utf-8")
    csv_file.seek(0)
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(csv_file, sep=sep)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

    st.session_state.experiments[name] = {
        "model":            models[model_key],
        "optimizer":        optimizers[opt_key],
        "primary_metric":   None,
        "original_metric":  None,
        "metrics":          metrics,
        "seed":            seed,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "result":  None,
    }
    st.session_state.active = name
    st.session_state.creating = False
    st.rerun()
