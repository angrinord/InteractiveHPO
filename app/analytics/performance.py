import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.strings import S


def performance(name, result, selected_idx, sel_key, display_metric):
    st.subheader(S("subheader_performance"))
    trials = result.trials
    params = list(trials[0].config.keys()) if trials else []

    # Recompute incumbents for the display metric
    incumbent = float("-inf")
    incumbent_scores = []
    for t in trials:
        s = t.scores[display_metric]
        if s > incumbent:
            incumbent = s
        incumbent_scores.append(incumbent)

    # One row of customdata per trial: the HP values in params order
    customdata = [[t.config.get(p) for p in params] for t in trials]

    # Build a hover template with one line per hyperparameter
    hp_rows = "".join(
        S("hover_hp_row").format(name=p, index=i) for i, p in enumerate(params)
    )
    hover_trial = S("hover_trial").format(hp_rows=hp_rows)

    df_trials = pd.DataFrame([
        {"Trial": t.trial, "Score": t.scores[display_metric], "Incumbent": inc}
        for t, inc in zip(trials, incumbent_scores)
    ])
    marker_colors = ["#636EFA"] * len(trials)
    marker_sizes = [6] * len(trials)
    if 0 <= selected_idx < len(trials):
        marker_colors[selected_idx] = "#EF553B"
        marker_sizes[selected_idx] = 13

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trials["Trial"], y=df_trials["Score"],
        mode="markers", name=S("trace_trial_score"),
        marker=dict(size=marker_sizes, color=marker_colors, opacity=0.7),
        customdata=customdata,
        hovertemplate=hover_trial,
    ))
    fig.add_trace(go.Scatter(
        x=df_trials["Trial"], y=df_trials["Incumbent"],
        mode="lines", name=S("trace_incumbent"),
        line=dict(width=2),
        hovertemplate=S("hover_incumbent"),
    ))
    fig.update_layout(
        xaxis_title=S("axis_trial"),
        yaxis_title=display_metric.capitalize(),
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
