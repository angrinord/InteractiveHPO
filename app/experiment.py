import threading
from pathlib import Path
import streamlit as st
from utils.strings import S
from optimizers.base import OptimizationResult
from utils.io import save as save_experiment, attach_dataset, attach_model, pick_file, load_model_from_path, has_display, mounted_models
import app.analytics as analytics
import app.dialogs as dialogs


def experiment(name: str):
    exp = st.session_state.experiments[name]

    if exp["original_metric"] is None:
        metric_label = "~"
    elif exp["primary_metric"] != exp["original_metric"]:
        metric_label = S("metric_inconsistent")
    else:
        metric_label = exp["primary_metric"]

    col_title, col_btns = st.columns([40, 2], vertical_alignment="center")
    with col_title:
        st.header(S("experiment_header").format(name=name))
        st.caption(S("experiment_caption").format(
            model=exp["model"].name if exp.get("model") is not None else exp.get("model_name", "~"),
            optimizer=exp["optimizer"].name,
            metric=metric_label,
            seed=exp["seed"],
        ))
    with col_btns:
        col_save, col_del = st.columns(2)
        with col_save:
            st.download_button(
                "💾",
                data=save_experiment(name, exp),
                file_name=f"{name}.ihpo",
                mime="application/octet-stream",
                key=f"{name}_save",
                use_container_width=True,
            )
        with col_del:
            if st.button("🗑", key=f"{name}_delete", use_container_width=True):
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
    dataset_available = exp.get("X_train") is not None
    model_available   = exp.get("model") is not None
    can_run           = dataset_available and model_available
    ctrl_left, ctrl_right = st.columns(2)

    with ctrl_left:
        with st.container(border=True):
            if can_run:
                with st.form("run_form", border=False):
                    n_trials = st.number_input(
                        S("field_n_trials"), min_value=1, max_value=1000, value=30, step=5
                    )
                    run = st.form_submit_button(
                        S("btn_run"), use_container_width=True, disabled=is_running
                    )
            else:
                run = False

                # ── Model picker (shown when custom model file is unavailable) ──
                if not model_available:
                    st.caption(S("info_model_readonly_mode"))
                    pending_mdl_key = f"{name}_pending_model_path"
                    if has_display():
                        col_mp, col_mb = st.columns([5, 1], vertical_alignment="bottom")
                        with col_mp:
                            new_mdl_path = st.text_input(
                                S("field_custom_model"),
                                value=st.session_state.get(pending_mdl_key,
                                                            exp.get("model_path", "")),
                                key=f"{name}_model_path_input",
                            )
                        with col_mb:
                            if st.button(S("btn_browse"), key=f"{name}_browse_model"):
                                picked = pick_file(filetypes=[("Python files", "*.py"),
                                                              ("All files", "*.*")])
                                if picked:
                                    st.session_state[pending_mdl_key] = picked
                                    st.rerun()
                        st.session_state[pending_mdl_key] = new_mdl_path
                    else:
                        _mounts = mounted_models()
                        sel = st.selectbox(
                            S("field_custom_model"),
                            list(_mounts.keys()),
                            index=None,
                            placeholder="Select a model…",
                            key=f"{name}_model_path_input",
                        )
                        new_mdl_path = _mounts[sel] if sel else ""
                        st.session_state[pending_mdl_key] = new_mdl_path
                    mdl_file_exists = bool(new_mdl_path) and Path(new_mdl_path).is_file()
                    mdl_valid = False
                    if new_mdl_path and mdl_file_exists:
                        _, mdl_err = load_model_from_path(new_mdl_path)
                        if mdl_err:
                            st.error(mdl_err)
                        else:
                            mdl_valid = True
                    elif new_mdl_path:
                        st.error(S("err_dataset_not_found").format(path=new_mdl_path))
                    if st.button(S("btn_load_model"), use_container_width=True,
                                 type="primary", disabled=not mdl_valid,
                                 key=f"{name}_load_model"):
                        try:
                            attach_model(exp, new_mdl_path)
                            st.session_state.pop(pending_mdl_key, None)
                        except ValueError as e:
                            st.error(str(e))
                        st.rerun()

                # ── Dataset picker (shown when dataset file is unavailable) ─────
                if not dataset_available:
                    st.caption(S("info_readonly_mode"))
                    pending_ds_key = f"{name}_pending_dataset_path"
                    col_path, col_browse = st.columns([5, 1], vertical_alignment="bottom")
                    with col_path:
                        new_ds_path = st.text_input(
                            S("field_dataset"),
                            value=st.session_state.get(pending_ds_key,
                                                        exp.get("dataset_path", "")),
                            key=f"{name}_dataset_path_input",
                        )
                    with col_browse:
                        if st.button(S("btn_browse"), key=f"{name}_browse_dataset",
                                     disabled=not has_display()):
                            picked = pick_file(filetypes=[("CSV files", "*.csv"),
                                                          ("All files", "*.*")])
                            if picked:
                                st.session_state[pending_ds_key] = picked
                                st.rerun()
                    st.session_state[pending_ds_key] = new_ds_path
                    path_valid = bool(new_ds_path) and Path(new_ds_path).is_file()
                    if new_ds_path and not path_valid:
                        st.error(S("err_dataset_not_found").format(path=new_ds_path))
                    if st.button(S("btn_load_dataset"), use_container_width=True,
                                 type="primary", disabled=not path_valid,
                                 key=f"{name}_load_dataset"):
                        try:
                            attach_dataset(exp, new_ds_path)
                            st.session_state.pop(pending_ds_key, None)
                        except ValueError as e:
                            st.error(S("err_dataset_not_found").format(path=new_ds_path)
                                     + f" ({e})")
                        st.rerun()

    with ctrl_right:
        with st.container(border=True):
            display_metric_input = st.selectbox(
                S("field_display_metric"),
                metric_names,
                index=metric_names.index(display_metric),
                key=f"{name}_metric_select",
                disabled=is_running,
            )
            if result is not None:
                reevaluate = st.button(
                    S("btn_reevaluate"), use_container_width=True, disabled=is_running
                )
            else:
                reevaluate = False

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
        dialogs.open_metric_change_dialog(exp["original_metric"], display_metric, decision_key, pending_key)

    if st.session_state.get("_pending_delete"):
        dialogs.open_confirm_delete(name, exp)

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
                st.info(S("spinner_optimizing"))
            else:
                st.rerun()  # triggers full rerun to commit the result
        _status()
        if result is None:
            return  # nothing to show yet

    if result is None:
        st.info(S("info_no_result"))
        return

    if result.trials_limit is not None and len(result.trials) >= result.trials_limit:
        st.warning(S("warn_trials_exhausted").format(n=result.trials_limit))

    if sel_key not in st.session_state:
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[display_metric])
        st.session_state[sel_key] = best_idx
    selected_idx   = st.session_state[sel_key]
    selected_trial = result.trials[selected_idx]

    st.divider()

    # ── Analytics ─────────────────────────────────────────────────────────────
    panels = [
        (analytics.best_config,     (result, display_metric)),
        (analytics.selected_config, (selected_trial, result, display_metric)),
        (analytics.hp_importance,   (result, display_metric)),
        (analytics.performance,     (name, result, selected_idx, sel_key, display_metric)),
    ]

    cols = st.columns(2)
    for i, (panel, panel_args) in enumerate(panels):
        with cols[i % 2]:
            if i >= 2:
                st.divider()
            panel(*panel_args)


def _start_run(exp, primary_metric, n_trials, display_metric):
    """Launch the optimizer in a daemon thread; write result back into exp when done."""
    previous_result = exp["result"]
    cancel_event = threading.Event()
    exp["run_state"]              = "running"
    exp["cancel_event"]           = cancel_event
    exp["pending_result"]         = None
    exp["pending_display_metric"] = display_metric
    exp["run_error"]              = None

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
                cancel_event=cancel_event,
            )
            if not cancel_event.is_set():
                exp["pending_result"] = result
                exp["run_state"]      = "done"
        except Exception as e:
            if not cancel_event.is_set():
                exp["run_error"]  = str(e)
                exp["run_state"]  = "error"

    threading.Thread(target=_thread, daemon=True).start()
