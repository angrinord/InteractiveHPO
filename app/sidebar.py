import streamlit as st

from strings import LOCALES


def sidebar(S):
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
            st.rerun()

        st.divider()

        for name in st.session_state.experiments:
            active = name == st.session_state.active
            if st.button(
                name,
                key=f"exp_{name}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.active = name
                st.session_state.creating = False
                st.rerun()
