import pandas as pd
import streamlit as st

from utils.strings import S


def best_config(result, display_metric):
    best_trial = max(result.trials, key=lambda t: t.scores[display_metric])
    st.subheader(S("subheader_best_config"))
    st.caption(S("caption_best_overall").format(trial=best_trial.trial))
    st.metric(f"{display_metric} {S('label_score')}", f"{best_trial.scores[display_metric]:.4f}")
    st.dataframe(
        pd.DataFrame(best_trial.config.items(), columns=[S("col_hyperparameter"), S("col_value")])
          .set_index(S("col_hyperparameter")),
        use_container_width=True,
    )
