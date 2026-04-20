import pandas as pd
import streamlit as st

from utils.strings import S


def selected_config(selected_trial, result, display_metric):
    best_score = max(t.scores[display_metric] for t in result.trials)
    delta = selected_trial.scores[display_metric] - best_score
    st.subheader(S("subheader_selected_config"))
    st.caption(S("caption_selected_trial").format(trial=selected_trial.trial))
    st.metric(
        f"{display_metric} {S('label_score')}", f"{selected_trial.scores[display_metric]:.4f}",
        delta=f"{delta:+.4f}" if delta != 0.0 else None,
    )
    st.dataframe(
        pd.DataFrame(selected_trial.config.items(), columns=[S("col_hyperparameter"), S("col_value")])
          .set_index(S("col_hyperparameter")),
        use_container_width=True,
    )
