"""
User-facing strings for InteractiveHPO.

HOW TO ADD A STRING
-------------------
1. Add an entry to _STRINGS:  "your_key": "English text"
2. Use it in code with S("your_key")
3. Run check_translations.py to find which .po files need updating
4. Add the msgid/msgstr pair to each locale's app.po and recompile:
       msgfmt locale/<lang>/LC_MESSAGES/app.po -o locale/<lang>/LC_MESSAGES/app.mo

Strings with {placeholders} use .format(**kwargs) at the call site.
"""

from __future__ import annotations

import functools
import gettext
from pathlib import Path

_LOCALE_DIR = Path(__file__).parent.parent / "locale"

# Registry of all user-facing strings.
# Keys are the symbolic identifiers used in code; values are the English text
# and serve as gettext msgids in the .po catalogs.
# Every entry here should have a corresponding msgid in each locale's app.po.
_STRINGS: dict[str, str] = {
    "page_title":   "Interactive HPO",
    "home_title":   "Interactive HPO",
    "home_body":    "Use the sidebar to create a new experiment.",

    "sidebar_title":      "Experiments",
    "label_language":     "Language",
    "btn_new_experiment": "＋ New",

    "new_experiment_header": "New experiment",
    "field_experiment_name": "Experiment name",
    "field_dataset":         "Dataset (CSV – last column = target)",
    "field_model":           "Demo Model",
    "field_optimizer":       "Optimizer",
    "field_metric":          "Metric",
    "field_seed":            "Seed (negative = random)",
    "btn_create":            "Create",
    "err_name_required":     "Please enter an experiment name.",
    "err_name_exists":       "'{name}' already exists.",
    "err_dataset_required":  "Please select a dataset.",
    "err_dataset_not_found": "Dataset not found: '{path}'",
    "btn_browse":            "Browse…",

    "experiment_header":       "Experiment: {name}",
    "experiment_caption":      "Model: **{model}** · Optimizer: **{optimizer}** · Metric: **{metric}** · Seed: **{seed}**",
    "btn_delete":              "🗑 Delete",
    "dialog_delete_title":     "Delete experiment",
    "warn_delete":             "Are you sure you want to delete '{name}'? This cannot be undone.",
    "btn_confirm_delete":      "Delete",

    "field_n_trials":              "Number of trials",
    "field_display_metric":        "Evaluation metric",
    "btn_run":                     "▶ Run",
    "btn_reevaluate":              "⚖️ Reevaluate",
    "btn_eval_with":               "Evaluate with {metric}",
    "btn_cancel":                  "Cancel",
    "dialog_metric_warning_title": "Optimization metric changed",
    "warn_metric_changed":         "The evaluation metric has been changed from the original ({original}). Do you want to optimize with respect to the selected metric? ({new})",
    "metric_inconsistent":         "Inconsistent",
    "spinner_optimizing":          "Optimizing…",
    "info_no_result":              "Set the number of trials and click **▶ Run**.",
    "warn_trials_exhausted":       "All {n} available configurations have been evaluated. No further trials are possible.",
    "field_optimizer_params":      "Optimizer settings",

    "subheader_best_config":     "Best configuration",
    "subheader_selected_config": "Selected configuration",
    "caption_best_overall":      "Best overall (Trial {trial})",
    "label_score":               "score",
    "caption_selected_trial":    "Trial {trial} — click any point on the graph to select",
    "col_hyperparameter":        "Hyperparameter",
    "col_value":                 "Value",

    "info_readonly_mode":        "Dataset unavailable — provide it below to run further trials.",
    "btn_load_dataset":          "Load dataset",
    "btn_load_readonly":         "Load without dataset",

    "checkbox_use_demo_models":        "Use demo models",
    "field_custom_model":              "Model path (.py)",
    "err_model_required":              "Please provide a model path or enable demo models.",
    "info_custom_model_loaded":        "Using custom model: '{name}'",
    "warn_load_model_path_missing":    "Custom model '{path}' could not be found — locate it or load read-only:",
    "info_model_readonly_mode":        "Custom model unavailable — provide it below to run further trials.",
    "btn_load_model":                  "Load model",

    "btn_load_experiment":       "📂 Load",
    "dialog_load_title":         "Load experiment",
    "field_load_file":           "Experiment file (.ihpo)",
    "btn_load":                  "Load",
    "err_load_invalid":          "Invalid or unreadable experiment file.",
    "err_load_metric":           "Unknown metric '{metric}'.",
    "warn_load_dataset_missing": "Original dataset '{name}' could not be found — please upload it:",
    "warn_load_model_missing":   "Model '{model}' is not available — please select one:",

    "subheader_hp_importance": "Hyperparameter importance (HyperSHAP)",
    "info_no_importance":      "No importance data available.",

    "subheader_performance": "Performance of Incumbent",
    "axis_trial":            "Trial",
    "trace_trial_score":     "Trial score",
    "trace_incumbent":       "Incumbent",
    "hover_trial":     "Trial: %{{x}}<br>Score: %{{y:.4f}}<br><b>Hyperparameters</b><br>{hp_rows}<extra></extra>",
    "hover_hp_row":    "{name}: %{{customdata[{index}]}}<br>",
    "hover_incumbent": "Trial: %{x}<br>Incumbent: %{y:.4f}<extra></extra>",
}

LOCALES: dict[str, str] = {"English": "en", "Deutsch": "de", "Español": "es"}


@functools.lru_cache(maxsize=None)
def _get_translation(locale: str) -> gettext.NullTranslations:
    return gettext.translation(
        "app", localedir=str(_LOCALE_DIR), languages=[locale], fallback=True
    )


def S(key: str) -> str:
    """Translate a string by its symbolic key.

    Raises KeyError if *key* is not registered in _STRINGS, which surfaces
    missing registrations immediately at development time.
    """
    import streamlit as st
    en_text = _STRINGS[key]
    locale = getattr(st.session_state, "locale", "en")
    return _get_translation(locale).gettext(en_text)
