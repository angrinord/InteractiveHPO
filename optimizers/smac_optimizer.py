import logging
import tempfile

from smac import Scenario, BlackBoxFacade
from smac.callback import Callback

from .base import BaseOptimizer, OptimizationResult, TrialResult

logging.getLogger("smac").setLevel(logging.WARNING)

# Fixed trial budget used in every Scenario so the saved scenario hash never
# changes between runs.  Actual stopping is controlled by the callback, so
# SMAC always sees an identical scenario on resume and loads history cleanly.
_SMAC_MAX_TRIALS = 100_000


class _CollectCallback(Callback):
    """Captures per-trial results and stops SMAC after a fixed number of NEW trials.

    trial_offset        — trials already completed in previous runs, used to
                          produce globally-sequential trial numbers.
    target_new_trials   — stop SMAC (via smbo._stop) once this many NEW trials
                          have been recorded in this run.
    initial_best_cost   — best cost seen before this run, so incumbent_score is
                          correct relative to the full history.
    initial_best_config — config that produced initial_best_cost.
    """

    def __init__(
        self,
        target_new_trials: int,
        trial_offset: int = 0,
        initial_best_cost: float = float("inf"),
        initial_best_config: dict | None = None,
    ):
        self.results: list[TrialResult] = []
        self._target_new_trials = target_new_trials
        self._trial_offset = trial_offset
        self._incumbent_cost = initial_best_cost
        self._incumbent_config = initial_best_config

    def on_tell_end(self, smbo, info, value):
        cost = float(value.cost) if not isinstance(value.cost, float) else value.cost
        config = dict(info.config)
        score = 1.0 - cost

        if cost < self._incumbent_cost:
            self._incumbent_cost = cost
            self._incumbent_config = config

        self.results.append(TrialResult(
            trial=self._trial_offset + len(self.results) + 1,
            config=config,
            score=score,
            incumbent_score=1.0 - self._incumbent_cost,
            incumbent_config=self._incumbent_config or config,
        ))

        if len(self.results) >= self._target_new_trials:
            smbo._stop = True


class SMACOptimizer(BaseOptimizer):
    name = "SMAC (BlackBox)"

    def optimize(self, model, X_train, y_train, X_val, y_val, metric_fn, n_trials,
                 previous_result=None, seed: int = 0):
        if previous_result is not None:
            output_dir = previous_result.metadata["smac_output_dir"]
            trial_offset = len(previous_result.trials)
            overwrite = False
            callback = _CollectCallback(
                target_new_trials=n_trials,
                trial_offset=trial_offset,
                initial_best_cost=1.0 - previous_result.best_score,
                initial_best_config=previous_result.best_config,
            )
        else:
            output_dir = tempfile.mkdtemp()
            overwrite = True
            callback = _CollectCallback(target_new_trials=n_trials)

        _seed = seed  # capture before SMAC overwrites the `seed` kwarg

        def target_fn(config, seed: int = 0) -> float:
            score = model.train_evaluate(
                dict(config), X_train, y_train, X_val, y_val, metric_fn, seed=_seed
            )
            return 1.0 - score  # SMAC minimises cost

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
        hp_importance, hp_warning = self.compute_hp_importance(
            model.get_config_space(seed=seed), all_trials, seed=seed
        )

        return OptimizationResult(
            trials=all_trials,
            best_config=dict(incumbent),
            best_score=max((r.score for r in all_trials), default=0.0),
            hyperparameter_importance=hp_importance,
            hyperparameter_importance_warning=hp_warning,
            metadata={"smac_output_dir": output_dir},
        )


# Module-level sentinel — the loader looks for this name.
OPTIMIZER = SMACOptimizer()
