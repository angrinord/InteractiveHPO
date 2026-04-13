import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from models.random_forest import RandomForestModel
from optimizers.smac_optimizer import SMACOptimizer
from optimizers.base import OptimizationResult
from strings import get_strings

MODELS = {
    "Random Forest": RandomForestModel(),
}

OPTIMIZERS = {
    "SMAC": SMACOptimizer(),
}

METRICS = {
    "accuracy": lambda y, yp: accuracy_score(y, yp),
    "f1":       lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
}

LOCALES = {"English": "en", "Deutsch": "de", "Español": "es"}

# Resolved strings for the current run — updated at the top of run().
S: dict = get_strings()

# Session state
def _init_state():
    st.session_state.setdefault("experiments", {})
    st.session_state.setdefault("active", None)
    st.session_state.setdefault("creating", False)
    st.session_state.setdefault("locale", "en")

# Sidebar
def _sidebar():
    with st.sidebar:
        locale_labels = list(LOCALES.keys())
        current_label = next(k for k, v in LOCALES.items() if v == st.session_state.locale)
        col_icon, col_sel, _ = st.columns([1, 2, 4], gap="small")
        with col_icon:
            st.markdown("🌐")
        with col_sel:
            chosen_label = st.selectbox(
                S["label_language"],
                locale_labels,
                index=locale_labels.index(current_label),
                label_visibility="collapsed",
            )
        chosen_locale = LOCALES[chosen_label]
        if chosen_locale != st.session_state.locale:
            st.session_state.locale = chosen_locale
            st.rerun()

        st.title(S["sidebar_title"])

        if st.button(S["btn_new_experiment"], use_container_width=True):
            st.session_state.creating = True
            st.session_state.active = None

        st.divider()

        for name in st.session_state.experiments:
            active = name == st.session_state.active
            label = f"{S['experiment_active_prefix'] if active else ''}{name}"
            if st.button(label, key=f"exp_{name}", use_container_width=True):
                st.session_state.active = name
                st.session_state.creating = False

# New experiment form
def _new_experiment_form():
    st.header(S["new_experiment_header"])

    with st.form("new_exp"):
        name      = st.text_input(S["field_experiment_name"])
        csv_file  = st.file_uploader(S["field_dataset"], type="csv")
        model_key = st.selectbox(S["field_model"],     list(MODELS.keys()))
        opt_key   = st.selectbox(S["field_optimizer"], list(OPTIMIZERS.keys()))
        metric     = st.selectbox(S["field_metric"],    list(METRICS.keys()))
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
        "model":      MODELS[model_key],
        "optimizer":  OPTIMIZERS[opt_key],
        "metric":     metric,
        "metric_fn":  METRICS[metric],
        "seed":       seed,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "result":  None,
    }
    st.session_state.active = name
    st.session_state.creating = False
    st.rerun()

# Experiment page
def _experiment_page(name: str):
    exp = st.session_state.experiments[name]

    col_title, col_del = st.columns([5, 1])
    with col_title:
        st.header(S["experiment_header"].format(name=name))
        st.caption(S["experiment_caption"].format(
            model=exp["model"].name,
            optimizer=exp["optimizer"].name,
            metric=exp["metric"],
            seed=exp["seed"],
        ))
    with col_del:
        if st.button(S["btn_delete"]):
            del st.session_state.experiments[name]
            st.session_state.active = None
            st.rerun()

    st.divider()

    with st.form("run_form"):
        n_trials = st.number_input(S["field_n_trials"], min_value=1, max_value=1000, value=30, step=5)
        run = st.form_submit_button(S["btn_run"])

    if run:
        with st.spinner(S["spinner_optimizing"]):
            result = exp["optimizer"].optimize(
                model=exp["model"],
                X_train=exp["X_train"], y_train=exp["y_train"],
                X_val=exp["X_val"],     y_val=exp["y_val"],
                metric_fn=exp["metric_fn"],
                n_trials=int(n_trials),
                previous_result=exp["result"],
                seed=exp["seed"],
            )
        exp["result"] = result
        best_idx = max(range(len(result.trials)), key=lambda i: result.trials[i].score)
        st.session_state[f"{name}_selected_trial"] = best_idx
        st.rerun()

    result: OptimizationResult | None = exp["result"]
    if result is None:
        st.info(S["info_no_result"])
        return

    sel_key = f"{name}_selected_trial"
    if sel_key not in st.session_state:
        best_idx = max(range(len(result.trials)), key=lambda i: result.trials[i].score)
        st.session_state[sel_key] = best_idx
    selected_idx = st.session_state[sel_key]
    selected_trial = result.trials[selected_idx]

    # 1. Best config + selected trial config
    best_trial_num = max(result.trials, key=lambda t: t.score).trial
    st.subheader(S["subheader_best_config"])
    col_best, col_sel = st.columns(2)
    with col_best:
        st.caption(S["caption_best_overall"].format(trial=best_trial_num))
        st.metric(S["label_score"], f"{result.best_score:.4f}")
        st.dataframe(
            pd.DataFrame(result.best_config.items(), columns=[S["col_hyperparameter"], S["col_value"]])
              .set_index(S["col_hyperparameter"]),
            use_container_width=True,
        )
    with col_sel:
        delta = selected_trial.score - result.best_score
        st.caption(S["caption_selected_trial"].format(trial=selected_trial.trial))
        st.metric(
            S["label_score"], f"{selected_trial.score:.4f}",
            delta=f"{delta:+.4f}" if delta != 0.0 else None,
        )
        st.dataframe(
            pd.DataFrame(selected_trial.config.items(), columns=[S["col_hyperparameter"], S["col_value"]])
              .set_index(S["col_hyperparameter"]),
            use_container_width=True,
        )

    st.divider()

    col_pie, col_perf = st.columns(2)

    # 2. HyperSHAP pie
    with col_pie:
        st.subheader(S["subheader_hp_importance"])
        if result.hyperparameter_importance_warning:
            st.warning(result.hyperparameter_importance_warning)
        imp = result.hyperparameter_importance or {}
        if imp:
            fig = go.Figure(go.Pie(
                labels=list(imp.keys()), values=list(imp.values()),
                hole=0.35, textinfo="label+percent",
            ))
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(S["info_no_importance"])

    # 3. Trial performance graph
    with col_perf:
        st.subheader(S["subheader_performance"])
        trials = result.trials
        params = list(trials[0].config.keys()) if trials else []

        # One row of customdata per trial: the HP values in params order
        customdata = [[t.config.get(p) for p in params] for t in trials]

        # Build a hover template with one line per hyperparameter
        hp_rows = "".join(
            S["hover_hp_row"].format(name=p, index=i) for i, p in enumerate(params)
        )
        hover_trial = S["hover_trial"].format(hp_rows=hp_rows)

        df_trials = pd.DataFrame([
            {"Trial": r.trial, "Score": r.score, "Incumbent": r.incumbent_score}
            for r in trials
        ])
        marker_colors = ["#636EFA"] * len(trials)
        marker_sizes = [6] * len(trials)
        if 0 <= selected_idx < len(trials):
            marker_colors[selected_idx] = "#EF553B"
            marker_sizes[selected_idx] = 13

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials["Trial"], y=df_trials["Score"],
            mode="markers", name=S["trace_trial_score"],
            marker=dict(size=marker_sizes, color=marker_colors, opacity=0.7),
            customdata=customdata,
            hovertemplate=hover_trial,
        ))
        fig.add_trace(go.Scatter(
            x=df_trials["Trial"], y=df_trials["Incumbent"],
            mode="lines", name=S["trace_incumbent"],
            line=dict(width=2),
            hovertemplate=S["hover_incumbent"],
        ))
        fig.update_layout(
            xaxis_title=S["axis_trial"],
            yaxis_title=exp["metric"].capitalize(),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=40, l=40, r=20),
        )

        event = st.plotly_chart(
            fig, use_container_width=True,
            on_select="rerun", selection_mode="points",
            key=f"perf_chart_{name}",
        )
        pts = getattr(getattr(event, "selection", None), "points", None) or []
        for pt in pts:
            if pt.get("curve_number", 0) == 0:
                new_idx = pt["point_index"]
                if st.session_state.get(sel_key) != new_idx:
                    st.session_state[sel_key] = new_idx
                    st.rerun()

def run():
    global S
    _init_state()
    S = get_strings(st.session_state.locale)
    st.set_page_config(page_title=S["page_title"], layout="wide")
    _sidebar()

    if st.session_state.creating:
        _new_experiment_form()
    elif st.session_state.active in st.session_state.experiments:
        _experiment_page(st.session_state.active)
    else:
        st.title(S["home_title"])
        st.write(S["home_body"])

if __name__ == "__main__":
    run()
