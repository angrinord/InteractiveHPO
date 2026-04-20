import logging
import tempfile

from smac import Scenario, BlackBoxFacade
from smac.callback import Callback

from .base import BaseOptimizer, OptimizationResult, TrialCollector

logging.getLogger("smac").setLevel(logging.WARNING)

# Fixed trial budget used in every Scenario so the saved scenario hash never
# changes between runs.  Actual stopping is controlled by the callback, so
# SMAC always sees an identical scenario on resume and loads history cleanly.
_SMAC_MAX_TRIALS = 100_000


class _CollectCallback(TrialCollector, Callback):
    """SMAC Callback that delegates trial recording to TrialCollector.

    Adds only the SMAC-specific concerns: extracting the score from SMAC's
    cost value, looking up all metric scores from the cache written by
    target_fn, and signaling SMAC to stop once the trial target is reached.

    scores_cache   — dict populated by target_fn mapping config_key → scores dict.
    primary_metric — name of the primary metric (cost = 1 − scores[primary_metric]).
    """

    def __init__(
        self,
        scores_cache: dict | None = None,
        primary_metric: str = "",
        cancel_event=None,
        **kwargs,
    ):
        TrialCollector.__init__(self, **kwargs)
        self._scores_cache = scores_cache if scores_cache is not None else {}
        self._primary_metric = primary_metric
        self._cancel_event = cancel_event

    def on_tell_end(self, smbo, info, value):
        cost = float(value.cost) if not isinstance(value.cost, float) else value.cost
        config = dict(info.config)
        score = 1.0 - cost
        config_key = tuple(sorted(config.items()))
        all_scores = self._scores_cache.get(config_key, {self._primary_metric: score})
        self.record(config, score, all_scores)
        if self.done or (self._cancel_event and self._cancel_event.is_set()):
            smbo._stop = True


class SMACOptimizer(BaseOptimizer):
    name = "SMAC (BlackBox)"

    def _make_callback(
        self,
        target_new_trials: int,
        trial_offset: int = 0,
        initial_best_score: float = float("-inf"),
        initial_best_config: dict | None = None,
        scores_cache: dict | None = None,
        primary_metric: str = "",
        cancel_event=None,
        **_,
    ) -> TrialCollector:
        return _CollectCallback(
            target_new_trials=target_new_trials,
            trial_offset=trial_offset,
            initial_best_score=initial_best_score,
            initial_best_config=initial_best_config,
            scores_cache=scores_cache,
            primary_metric=primary_metric,
            cancel_event=cancel_event,
        )

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials, previous_result=None, seed: int = 0, cancel_event=None):
        scores_cache: dict = {}

        if previous_result is not None:
            output_dir = previous_result.metadata["smac_output_dir"]
            trial_offset = len(previous_result.trials)
            overwrite = False
            callback = self._make_callback(
                target_new_trials=n_trials,
                trial_offset=trial_offset,
                initial_best_score=previous_result.best_score,
                initial_best_config=previous_result.best_config,
                scores_cache=scores_cache,
                primary_metric=primary_metric,
                cancel_event=cancel_event,
            )
        else:
            output_dir = tempfile.mkdtemp()
            overwrite = True
            callback = self._make_callback(
                target_new_trials=n_trials,
                scores_cache=scores_cache,
                primary_metric=primary_metric,
                cancel_event=cancel_event,
            )

        _seed = seed  # capture before SMAC overwrites the `seed` kwarg

        def target_fn(config, seed: int = 0) -> float:
            config_dict = dict(config)
            all_scores = model.train_evaluate(
                config_dict, X_train, y_train, X_val, y_val, metrics, seed=_seed
            )
            scores_cache[tuple(sorted(config_dict.items()))] = all_scores
            return 1.0 - all_scores[primary_metric]

        scenario = Scenario(
            model.get_config_space(seed=seed),
            n_trials=_SMAC_MAX_TRIALS,
            deterministic=True,
            seed=seed,
            output_directory=output_dir,
        )
        smac = BlackBoxFacade(
            scenario,
            target_fn,
            callbacks=[callback],
            overwrite=overwrite,
        )
        smac.optimize()
        incumbent = smac.intensifier.get_incumbent()

        all_trials = (previous_result.trials if previous_result else []) + callback.results

        config_space = model.get_config_space(seed=seed)
        hp_importance = {}
        hp_warning = {}
        for metric_name in metrics:
            imp, warn = self.compute_hp_importance(config_space, all_trials, metric_name, seed=seed)
            hp_importance[metric_name] = imp
            hp_warning[metric_name] = warn

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=dict(incumbent),
            best_score=max((r.score for r in all_trials), default=0.0),
            hyperparameter_importance=hp_importance,
            hyperparameter_importance_warning=hp_warning,
            metadata={"smac_output_dir": output_dir},
        )


# Module-level sentinel — the loader looks for this name.
OPTIMIZER = SMACOptimizer()
