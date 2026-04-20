import plotly.graph_objects as go
import streamlit as st

from utils.strings import S


def hp_importance(result, display_metric):
    st.subheader(S("subheader_hp_importance"))
    warning = result.hyperparameter_importance_warning.get(display_metric)
    if warning:
        st.warning(warning)
    imp = result.hyperparameter_importance.get(display_metric, {})
    if imp:
        fig = go.Figure(go.Pie(
            labels=list(imp.keys()), values=list(imp.values()),
            hole=0.35, textinfo="label+percent",
        ))
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(S("info_no_importance"))
