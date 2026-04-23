import random
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.strings import S
from utils.io import pick_file, load_model_from_path, has_display, demo_datasets, mounted_models


def _render_param(p):
    if p.type == "int":
        return int(st.number_input(p.label, min_value=p.min, max_value=p.max, value=p.default, step=1))
    if p.type == "float":
        return float(st.number_input(p.label, min_value=p.min, max_value=p.max, value=float(p.default)))
    if p.type == "select":
        return st.selectbox(p.label, p.choices, index=p.choices.index(p.default))
    return p.default


def new_form(models, optimizers, metrics):
    st.header(S("new_experiment_header"))

    # Lives outside the form so switching it immediately refreshes parameter fields without a submit.
    opt_key = st.selectbox(
        S("field_optimizer"),
        list(optimizers.keys()),
        key="_new_exp_optimizer",
    )
    schema = optimizers[opt_key].params_schema

    # Dataset — demo selectbox (forced when no display) or manual path + Browse.
    _display_ok = has_display()
    _demos = demo_datasets()
    use_demo_ds = st.checkbox(
        S("checkbox_use_demo_datasets"),
        value=not _display_ok,
        disabled=not _display_ok,
        key="_new_exp_use_demo_ds",
    )

    if use_demo_ds:
        selected_demo_ds = st.selectbox(
            S("field_demo_dataset"),
            list(_demos.keys()),
            index=None,
            placeholder="Select a dataset…",
            key="_new_exp_demo_ds",
        )
        dataset_path = _demos[selected_demo_ds] if selected_demo_ds else ""
    else:
        col_path, col_browse = st.columns([5, 1], vertical_alignment="bottom")
        with col_path:
            dataset_path = st.text_input(
                S("field_dataset"),
                value=st.session_state.get("_new_exp_dataset_path", ""),
                placeholder="/path/to/data.csv",
                key="_new_exp_dataset_input",
            )
        with col_browse:
            if st.button(S("btn_browse"), key="_new_exp_dataset_browse",
                         use_container_width=True, disabled=not _display_ok):
                picked = pick_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                if picked:
                    st.session_state["_new_exp_dataset_path"] = picked
                    st.rerun()

    # Model — demo registry selectbox or custom file picker; falls back to mounted .py files without a display.
    use_demo = st.checkbox(S("checkbox_use_demo_models"), key="_new_exp_use_demo",
                           value=not _display_ok, disabled=not _display_ok)

    custom_model_path = ""
    selected_model_key = None
    if use_demo:
        selected_model_key = st.selectbox(
            S("field_model"),
            list(models.keys()),
            index=None,
            key="_new_exp_model",
            placeholder="Select a model…",
        )
    elif _display_ok:
        col_custom, col_browse_mdl = st.columns([5, 1], vertical_alignment="bottom")
        with col_custom:
            custom_model_path = st.text_input(
                S("field_custom_model"),
                value=st.session_state.get("_new_exp_model_path", ""),
                placeholder="/path/to/model.py",
                key="_new_exp_model_input",
            )
        with col_browse_mdl:
            if st.button(S("btn_browse"), key="_new_exp_model_browse",
                         use_container_width=True):
                picked = pick_file(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
                if picked:
                    st.session_state["_new_exp_model_path"] = picked
                    st.rerun()
        st.session_state["_new_exp_model_path"] = custom_model_path
    else:
        _mounts = mounted_models()
        selected_mounted_mdl = st.selectbox(
            S("field_custom_model"),
            list(_mounts.keys()),
            index=None,
            placeholder="Select a model…",
            key="_new_exp_mounted_model",
        )
        custom_model_path = _mounts[selected_mounted_mdl] if selected_mounted_mdl else ""

    custom_model = None
    if not use_demo and custom_model_path:
        custom_model, model_err = load_model_from_path(custom_model_path)
        if model_err:
            st.error(model_err)
            custom_model = None
        else:
            st.success(S("info_custom_model_loaded").format(name=custom_model.name))

    with st.form("new_exp"):
        name       = st.text_input(S("field_experiment_name"))
        seed_input = st.number_input(S("field_seed"), value=0, step=1)

        if schema:
            st.subheader(S("field_optimizer_params"))
            opt_param_values = {p.name: _render_param(p) for p in schema}
        else:
            opt_param_values = {}

        submitted = st.form_submit_button(S("btn_create"))

    if not submitted:
        return

    if not name:
        st.error(S("err_name_required"))
        return
    if name in st.session_state.experiments:
        st.error(S("err_name_exists").format(name=name))
        return
    if not dataset_path:
        st.error(S("err_dataset_required"))
        return
    path = Path(dataset_path)
    if not path.is_file():
        st.error(S("err_dataset_not_found").format(path=dataset_path))
        return
    if use_demo:
        if selected_model_key is None:
            st.error(S("err_model_required"))
            return
    else:
        if not custom_model_path:
            st.error(S("err_model_required"))
            return
        if custom_model is None:
            # Path entered but failed validation — error already shown above.
            return

    seed = random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)

    raw = path.read_bytes()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(path, sep=sep)
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

    optimizer = type(optimizers[opt_key])(**opt_param_values)

    if use_demo:
        model      = models[selected_model_key]
        model_name = selected_model_key
        model_path = ""
    else:
        model      = custom_model
        model_name = custom_model.name
        model_path = str(Path(custom_model_path).resolve())

    st.session_state.pop("_new_exp_dataset_path", None)
    st.session_state.pop("_new_exp_model_path", None)

    st.session_state.experiments[name] = {
        "model":           model,
        "model_name":      model_name,
        "model_path":      model_path,
        "optimizer":       optimizer,
        "primary_metric":  None,
        "original_metric": None,
        "metrics":         metrics,
        "seed":            seed,
        "dataset_path":    str(path.resolve()),
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "result":  None,
    }
    st.session_state.active = name
    st.session_state.creating = False
    st.rerun()
