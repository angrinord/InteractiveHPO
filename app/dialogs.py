"""Streamlit dialogs for InteractiveHPO.

Each public function opens the corresponding dialog by defining the decorated
inner function at call time.  This defers the ``@st.dialog(S(...))`` evaluation
to render time so that the locale is always resolved correctly.
"""

from pathlib import Path

import streamlit as st

from utils.strings import S
from utils.io import (
    parse as parse_experiment,
    build_experiment,
    dataset_path_ok,
    model_path_ok,
    load_model_from_path,
    pick_file,
    has_display,
    mounted_models,
)


def open_confirm_delete(name: str, exp: dict) -> None:
    """Open the delete-confirmation dialog for *name*."""
    @st.dialog(S("dialog_delete_title"))
    def _dialog():
        st.write(S("warn_delete").format(name=name))
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button(S("btn_confirm_delete"), use_container_width=True, type="primary"):
                if cancel_event := exp.get("cancel_event"):
                    cancel_event.set()
                del st.session_state.experiments[name]
                st.session_state.active = None
                st.session_state["_pending_delete"] = False
                st.rerun()
        with col_cancel:
            if st.button(S("btn_cancel"), use_container_width=True):
                st.session_state["_pending_delete"] = False
                st.rerun()

    _dialog()


def open_metric_change_dialog(
    original_metric: str,
    display_metric: str,
    decision_key: str,
    pending_key: str,
) -> None:
    """Open the metric-change warning dialog."""
    @st.dialog(S("dialog_metric_warning_title"))
    def _dialog():
        st.write(S("warn_metric_changed").format(
            original=original_metric,
            new=display_metric,
        ))
        st.divider()
        btn_new, btn_old, btn_cancel = st.columns(3)
        with btn_new:
            if st.button(S("btn_eval_with").format(metric=display_metric),
                         use_container_width=True):
                st.session_state[decision_key] = "new"
                st.session_state[pending_key]  = False
                st.rerun()
        with btn_old:
            if st.button(S("btn_eval_with").format(metric=original_metric),
                         use_container_width=True):
                st.session_state[decision_key] = "old"
                st.session_state[pending_key]  = False
                st.rerun()
        with btn_cancel:
            if st.button(S("btn_cancel"), use_container_width=True):
                st.session_state[pending_key] = False
                st.rerun()

    _dialog()


def open_load_dialog() -> None:
    """Open the load-experiment dialog."""
    @st.dialog(S("dialog_load_title"))
    def _dialog():
        # Pre-seed text inputs from Browse picks made during the last render.
        # Must happen before any widget is instantiated (Streamlit constraint).
        if "_load_pending_path" in st.session_state:
            st.session_state["_load_dialog_dataset_input"] = st.session_state.pop("_load_pending_path")
        if "_load_pending_model_path" in st.session_state:
            st.session_state["_load_dialog_model_input"] = st.session_state.pop("_load_pending_model_path")

        uploaded_ihpo = st.file_uploader(S("field_load_file"), type="ihpo")
        if uploaded_ihpo is None:
            return

        try:
            snapshot = parse_experiment(uploaded_ihpo.read())
        except ValueError as exc:
            st.error(S("err_load_invalid") + f" ({exc})")
            return

        available_models     = st.session_state["_models"]
        available_optimizers = st.session_state["_optimizers"]
        available_metrics    = st.session_state["_metrics"]

        # ── Dataset ───────────────────────────────────────────────────────────
        path_ok = dataset_path_ok(snapshot)
        if not path_ok:
            st.warning(S("warn_load_dataset_missing").format(name=snapshot["dataset_path"]))
            col_path, col_browse = st.columns([5, 1], vertical_alignment="bottom")
            with col_path:
                override_path = st.text_input(
                    S("field_dataset"),
                    value=st.session_state.get("_load_dialog_dataset_path",
                                               snapshot["dataset_path"]),
                    key="_load_dialog_dataset_input",
                )
            with col_browse:
                just_browsed = False
                if st.button(S("btn_browse"), key="_load_dialog_browse",
                             disabled=not has_display()):
                    picked = pick_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    if picked:
                        st.session_state["_load_dialog_dataset_path"] = picked
                        st.session_state["_load_pending_path"] = picked
                        just_browsed = True

            if not just_browsed:
                st.session_state["_load_dialog_dataset_path"] = override_path

            effective_path = st.session_state.get("_load_dialog_dataset_path", "")
            path_ok = bool(effective_path) and Path(effective_path).is_file()
            if effective_path and not path_ok:
                st.error(S("err_dataset_not_found").format(path=effective_path))
            snapshot = {**snapshot, "dataset_path": effective_path}

        # ── Custom model path ─────────────────────────────────────────────────
        stored_model_path = snapshot.get("model_path", "")
        custom_model_ok = True  # True when no custom model, or path is valid

        if stored_model_path:
            if not model_path_ok(snapshot):
                st.warning(S("warn_load_model_path_missing").format(path=stored_model_path))
                if has_display():
                    col_mp, col_mb = st.columns([5, 1], vertical_alignment="bottom")
                    with col_mp:
                        override_model_path = st.text_input(
                            S("field_custom_model"),
                            value=st.session_state.get("_load_dialog_model_path",
                                                        stored_model_path),
                            key="_load_dialog_model_input",
                        )
                    with col_mb:
                        just_browsed_model = False
                        if st.button(S("btn_browse"), key="_load_dialog_model_browse"):
                            picked = pick_file(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
                            if picked:
                                st.session_state["_load_dialog_model_path"] = picked
                                st.session_state["_load_pending_model_path"] = picked
                                just_browsed_model = True

                    if not just_browsed_model:
                        st.session_state["_load_dialog_model_path"] = override_model_path
                else:
                    _mounts = mounted_models()
                    sel = st.selectbox(
                        S("field_custom_model"),
                        list(_mounts.keys()),
                        index=None,
                        placeholder="Select a model…",
                        key="_load_dialog_model_input",
                    )
                    override_model_path = _mounts[sel] if sel else ""
                    st.session_state["_load_dialog_model_path"] = override_model_path

                effective_model_path = st.session_state.get("_load_dialog_model_path", "")
                if effective_model_path and Path(effective_model_path).is_file():
                    _, model_err = load_model_from_path(effective_model_path)
                    if model_err:
                        st.error(model_err)
                        custom_model_ok = False
                    else:
                        custom_model_ok = True
                        snapshot = {**snapshot, "model_path": effective_model_path}
                elif effective_model_path:
                    st.error(S("err_dataset_not_found").format(path=effective_model_path))
                    custom_model_ok = False
                else:
                    custom_model_ok = False
            else:
                # Path exists — validate the file content up front.
                _, model_err = load_model_from_path(stored_model_path)
                if model_err:
                    st.error(model_err)
                    custom_model_ok = False

        # ── Built-in model selection (registry models only) ───────────────────
        model_name = snapshot["model_name"]
        if not stored_model_path:
            if model_name not in available_models:
                st.warning(S("warn_load_model_missing").format(model=model_name))
                model_name = st.selectbox(S("field_model"), list(available_models.keys()))

        # ── Experiment name ───────────────────────────────────────────────────
        name = st.text_input(S("field_experiment_name"), value=snapshot["name"])
        name_taken = name in st.session_state.experiments
        if name_taken:
            st.error(S("err_name_exists").format(name=name))

        def _do_load(read_only: bool) -> None:
            try:
                _, exp = build_experiment(
                    snapshot, available_metrics, available_models, available_optimizers,
                    model_name=model_name,
                    read_only=read_only,
                )
            except ValueError as exc:
                st.error(S("err_load_invalid") + f" ({exc})")
                return
            st.session_state.experiments[name] = exp
            st.session_state.active = name
            st.session_state.creating = False
            st.session_state.pop("_load_dialog_dataset_path", None)
            st.session_state.pop("_load_dialog_model_path", None)
            st.rerun()

        col_load, col_readonly = st.columns(2)
        with col_load:
            if st.button(
                S("btn_load"), use_container_width=True, type="primary",
                disabled=(not name or name_taken or not path_ok or not custom_model_ok),
            ):
                _do_load(read_only=False)
        with col_readonly:
            if st.button(
                S("btn_load_readonly"), use_container_width=True,
                disabled=(not name or name_taken),
            ):
                _do_load(read_only=True)

    _dialog()
