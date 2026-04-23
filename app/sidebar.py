import streamlit as st

from utils.strings import S, LOCALES
import app.dialogs as dialogs

# CSS spinner; keyframe prefixed to avoid collisions with other page styles.
_SPINNER = (
    '<style>@keyframes _sb_spin{to{transform:rotate(360deg)}}'
    '._sb_dot{width:14px;height:14px;border-radius:50%;'
    'border:2px solid rgba(150,150,150,0.25);'
    'border-top-color:rgba(150,150,150,0.85);'
    'animation:_sb_spin 0.75s linear infinite}</style>'
    '<div style="display:flex;align-items:center;height:38px">'
    '<div class="_sb_dot"></div></div>'
)


def sidebar():
    with st.sidebar:
        locale_labels = list(LOCALES.keys())
        current_label = next(k for k, v in LOCALES.items() if v == st.session_state.locale)
        col_icon, col_sel, _ = st.columns([1, 2, 4], gap="small")
        with col_icon:
            st.markdown("🌐")
        with col_sel:
            chosen_label = st.selectbox(
                S("label_language"),
                locale_labels,
                index=locale_labels.index(current_label),
                label_visibility="collapsed",
            )
        chosen_locale = LOCALES[chosen_label]
        if chosen_locale != st.session_state.locale:
            st.session_state.locale = chosen_locale
            st.rerun()

        st.title(S("sidebar_title"))

        col_new, col_load = st.columns(2)
        with col_new:
            if st.button(S("btn_new_experiment"), use_container_width=True):
                st.session_state.creating = True
                st.session_state.active = None
                st.rerun()
        with col_load:
            if st.button(S("btn_load_experiment"), use_container_width=True):
                dialogs.open_load_dialog()

        st.divider()

        for name, exp in st.session_state.experiments.items():
            active     = name == st.session_state.active
            is_running = exp.get("run_state") == "running"

            if is_running:
                col_btn, col_spin = st.columns([5, 1], vertical_alignment="center")
                with col_btn:
                    clicked = st.button(
                        name, key=f"exp_{name}", use_container_width=True,
                        type="primary" if active else "secondary",
                    )
                with col_spin:
                    st.markdown(_SPINNER, unsafe_allow_html=True)
            else:
                clicked = st.button(
                    name, key=f"exp_{name}", use_container_width=True,
                    type="primary" if active else "secondary",
                )

            if clicked:
                st.session_state.active = name
                st.session_state.creating = False
                st.rerun()
