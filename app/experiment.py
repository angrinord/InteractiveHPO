import threading

import streamlit as st

from optimizers.base import OptimizationResult
from app.analytics.best_config import best_config
from app.analytics.selected_config import selected_config
from app.analytics.hp_importance import hp_importance
from app.analytics.performance import performance


def experiment(name: str, S: dict):
    exp = st.session_state.experiments[name]

    if exp["original_metric"] is None:
        metric_label = "~"
    elif exp["primary_metric"] != exp["original_metric"]:
        metric_label = S["metric_inconsistent"]
    else:
        metric_label = exp["primary_metric"]

    col_title, col_del = st.columns([11, 1], vertical_alignment="center")
    with col_title:
        st.header(S["experiment_header"].format(name=name))
        st.caption(S["experiment_caption"].format(
            model=exp["model"].name,
            optimizer=exp["optimizer"].name,
            metric=metric_label,
            seed=exp["seed"],
        ))
    with col_del:
        if st.button("🗑", key=f"{name}_delete"):
            st.session_state["_pending_delete"] = True
            st.rerun()

    metric_names  = list(exp["metrics"].keys())
    display_key   = f"{name}_display_metric"
    sel_key       = f"{name}_selected_trial"
    pending_key   = f"{name}_pending_run"
    pending_n_key = f"{name}_pending_n_trials"
    decision_key  = f"{name}_run_decision"

    first_metric = next(iter(exp["metrics"]))
    if display_key not in st.session_state:
        st.session_state[display_key] = exp["primary_metric"] or first_metric
    display_metric = st.session_state[display_key]

    result: OptimizationResult | None = exp["result"]
    is_running = exp.get("run_state") == "running"

    # ── Commit completed background run ─────────────────────────────────────
    if exp.get("run_state") == "done":
        pending   = exp.pop("pending_result")
        exp["result"]     = pending
        exp["run_state"]  = None
        result            = pending
        pending_display   = exp.pop("pending_display_metric", display_metric)
        st.session_state[display_key] = pending_display
        display_metric = pending_display
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[display_metric])
        st.session_state[sel_key] = best_idx
        st.rerun()
    elif exp.get("run_state") == "error":
        st.error(f"Run failed: {exp.get('run_error', 'Unknown error')}")
        exp["run_state"] = None

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns(2)

    with ctrl_left:
        with st.container(border=True):
            with st.form("run_form"):
                n_trials = st.number_input(
                    S["field_n_trials"], min_value=1, max_value=1000, value=30, step=5
                )
                run = st.form_submit_button(
                    S["btn_run"], use_container_width=True, disabled=is_running
                )

    with ctrl_right:
        with st.container(border=True):
            display_metric_input = st.selectbox(
                S["field_display_metric"],
                metric_names,
                index=metric_names.index(display_metric),
                key=f"{name}_metric_select",
                disabled=is_running,
            )
            if result is not None:
                reevaluate = st.button(
                    S["btn_reevaluate"], use_container_width=True, disabled=is_running
                )
            else:
                reevaluate = False

    # ── Dialogs ──────────────────────────────────────────────────────────────
    @st.dialog(S["dialog_delete_title"])
    def _confirm_delete():
        st.write(S["warn_delete"].format(name=name))
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button(S["btn_confirm_delete"], use_container_width=True, type="primary"):
                del st.session_state.experiments[name]
                st.session_state.active = None
                st.session_state["_pending_delete"] = False
                st.rerun()
        with col_cancel:
            if st.button(S["btn_cancel"], use_container_width=True):
                st.session_state["_pending_delete"] = False
                st.rerun()

    @st.dialog(S["dialog_metric_warning_title"])
    def metric_change_dialog():
        st.write(S["warn_metric_changed"].format(
            original=exp["original_metric"],
            new=display_metric,
        ))
        st.divider()
        btn_new, btn_old, btn_cancel = st.columns(3)
        with btn_new:
            if st.button(S["btn_eval_with"].format(metric=display_metric),
                         use_container_width=True):
                st.session_state[decision_key] = "new"
                st.session_state[pending_key]  = False
                st.rerun()
        with btn_old:
            if st.button(S["btn_eval_with"].format(metric=exp["original_metric"]),
                         use_container_width=True):
                st.session_state[decision_key] = "old"
                st.session_state[pending_key]  = False
                st.rerun()
        with btn_cancel:
            if st.button(S["btn_cancel"], use_container_width=True):
                st.session_state[pending_key] = False
                st.rerun()

    # ── Run pressed ───────────────────────────────────────────────────────────
    if run and not is_running:
        if exp["original_metric"] is None:
            chosen = display_metric_input
            exp["primary_metric"]  = chosen
            exp["original_metric"] = chosen
            _start_run(exp, chosen, int(n_trials), chosen)
            st.session_state[display_key] = chosen
            st.rerun()
        else:
            needs_warning = (
                display_metric != exp["primary_metric"]
                and exp["primary_metric"] == exp["original_metric"]
            )
            if needs_warning:
                st.session_state[pending_key]  = True
                st.session_state[pending_n_key] = int(n_trials)
                st.rerun()
            else:
                _start_run(exp, exp["primary_metric"], int(n_trials), display_metric)
                st.session_state[display_key] = exp["primary_metric"]
                st.rerun()

    if st.session_state.get(pending_key):
        metric_change_dialog()

    if st.session_state.get("_pending_delete"):
        _confirm_delete()

    # ── Post-dialog run ───────────────────────────────────────────────────────
    decision = st.session_state.pop(decision_key, None)
    if decision == "new":
        exp["primary_metric"] = display_metric
        _start_run(exp, display_metric, st.session_state[pending_n_key], display_metric)
        st.session_state[display_key] = display_metric
        st.rerun()
    elif decision == "old":
        _start_run(exp, exp["original_metric"], st.session_state[pending_n_key],
                   exp["original_metric"])
        st.session_state[display_key] = exp["original_metric"]
        st.rerun()

    # ── Reevaluate pressed ────────────────────────────────────────────────────
    if reevaluate and result is not None:
        st.session_state[display_key] = display_metric_input
        display_metric = display_metric_input
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[display_metric])
        st.session_state[sel_key] = best_idx
        st.rerun()

    # ── Running indicator (auto-refreshes; sidebar stays interactive) ─────────
    if is_running:
        @st.fragment(run_every=2)
        def _status():
            if exp.get("run_state") == "running":
                st.info(S["spinner_optimizing"])
            else:
                st.rerun()  # triggers full rerun to commit the result
        _status()
        if result is None:
            return  # nothing to show yet

    if result is None:
        st.info(S["info_no_result"])
        return

    if sel_key not in st.session_state:
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[display_metric])
        st.session_state[sel_key] = best_idx
    selected_idx   = st.session_state[sel_key]
    selected_trial = result.trials[selected_idx]

    st.divider()

    # ── Analytics ─────────────────────────────────────────────────────────────
    analytics_left, analytics_right = st.columns(2)

    with analytics_left:
        best_config(result, S, display_metric)
        st.divider()
        hp_importance(result, S, display_metric)
    with analytics_right:
        selected_config(selected_trial, result, S, display_metric)
        st.divider()
        performance(name, result, selected_idx, sel_key, S, display_metric)


def _start_run(exp, primary_metric, n_trials, display_metric):
    """Launch the optimizer in a daemon thread; write result back into exp when done."""
    previous_result = exp["result"]
    exp["run_state"]             = "running"
    exp["pending_result"]        = None
    exp["pending_display_metric"] = display_metric
    exp["run_error"]             = None

    def _thread():
        try:
            result = exp["optimizer"].optimize(
                model=exp["model"],
                X_train=exp["X_train"], y_train=exp["y_train"],
                X_val=exp["X_val"],     y_val=exp["y_val"],
                metrics=exp["metrics"],
                primary_metric=primary_metric,
                n_trials=n_trials,
                previous_result=previous_result,
                seed=exp["seed"],
            )
            exp["pending_result"] = result
            exp["run_state"]      = "done"
        except Exception as e:
            exp["run_error"]  = str(e)
            exp["run_state"]  = "error"

    threading.Thread(target=_thread, daemon=True).start()
