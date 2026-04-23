from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

@dataclass
class OptimizerParam:
    """Describes one user-configurable parameter of an optimizer, for form rendering."""
    name: str                               # kwarg name passed to __init__
    label: str                              # human-readable label shown in the form
    type: str                               # "int", "float", or "select"
    default: Any
    min: Any = None                         # lower bound for int / float
    max: Any = None                         # upper bound for int / float
    choices: List[Any] = field(default_factory=list)  # options for select


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    scores: Dict[str, float]    # score for every metric
    score: float                # primary metric score (used internally)
    incumbent_score: float      # running incumbent score
    incumbent_config: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Accumulated results of a completed or partial optimization run."""

    trials: List[TrialResult]
    primary_metric: str
    best_config: Dict[str, Any]
    best_score: float
    hyperparameter_importance: Dict[str, Dict[str, float]]          # metric → {hp: importance}
    hyperparameter_importance_warning: Dict[str, Optional[str]]     # metric → warning or None
    trials_limit: Optional[int] = None    # None = unlimited; set by optimizers with a finite search space
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrialCollector:
    """Reusable bookkeeper for optimizer trial results.

    Tracks the per-trial results list and the running incumbent as trials
    complete.  Optimizer-specific callbacks should inherit from (or delegate
    to) this class and call ``record()`` once per evaluated trial.

    Parameters
    ----------
    target_new_trials:
        Stop collecting (``done`` becomes True) after this many new trials.
    trial_offset:
        Number of trials already recorded in a previous run; used to produce
        globally-sequential trial numbers when resuming.
    initial_best_score:
        Best primary-metric score seen before this run (``-inf`` for a fresh
        run).  Ensures the incumbent is correct relative to full history.
    initial_best_config:
        Config that produced ``initial_best_score``.
    """

    def __init__(
        self,
        target_new_trials: int,
        trial_offset: int = 0,
        initial_best_score: float = float("-inf"),
        initial_best_config: Optional[Dict[str, Any]] = None,
    ):
        self.results: List[TrialResult] = []
        self._target_new_trials = target_new_trials
        self._trial_offset = trial_offset
        self._incumbent_score = initial_best_score
        self._incumbent_config = initial_best_config

    @property
    def done(self) -> bool:
        """True once the target number of new trials has been collected."""
        return len(self.results) >= self._target_new_trials

    def record(
        self,
        config: Dict[str, Any],
        score: float,
        all_scores: Dict[str, float],
    ) -> TrialResult:
        """Record one completed trial, update the incumbent, return the TrialResult."""
        if score > self._incumbent_score:
            self._incumbent_score = score
            self._incumbent_config = config

        trial = TrialResult(
            trial=self._trial_offset + len(self.results) + 1,
            config=config,
            scores=all_scores,
            score=score,
            incumbent_score=self._incumbent_score,
            incumbent_config=self._incumbent_config or config,
        )
        self.results.append(trial)
        return trial


class BaseOptimizer(ABC):
    """Base class for all hyperparameter optimizers."""

    params_schema: List[OptimizerParam] = []

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _make_callback(
        self,
        target_new_trials: int,
        trial_offset: int = 0,
        initial_best_score: float = float("-inf"),
        initial_best_config: Optional[Dict[str, Any]] = None,
        cancel_event=None,
        **kwargs,
    ) -> TrialCollector:
        """Return an optimizer-framework callback that is also a TrialCollector.

        The returned object must:
          - Be accepted as a callback by the underlying optimizer framework.
          - Inherit from TrialCollector so that record(), done, and results work.
          - Call self.record() each time a trial evaluation completes.
          - Stop the optimizer once self.done is True.

        The four named parameters map directly to TrialCollector.__init__.
        Optimizer-specific constructor arguments (e.g. a scores_cache or a
        primary metric name) should be passed via **kwargs and handled by the
        concrete implementation.
        """
        ...

    @abstractmethod
    def optimize(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        metrics: dict,
        primary_metric: str,
        n_trials: int,
        previous_result: Optional["OptimizationResult"] = None,
        seed: int = 0,
        cancel_event=None,
    ) -> OptimizationResult: ...

    def get_params(self) -> dict:
        """Return current optimizer parameters as ``{name: value}`` for each schema entry.

        Uses the naming convention that a schema param named ``foo`` is stored
        on the instance as ``self._foo``.  Optimizers with no ``params_schema``
        return an empty dict.
        """
        return {p.name: getattr(self, f"_{p.name}") for p in self.params_schema}

    def compute_hp_importance(
        self,
        config_space,
        trials: List[TrialResult],
        metric_name: str,
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Estimate hyperparameter importance from a completed list of trials.

        Returns (importance_dict, warning_message).  warning_message is None
        when HyperSHAP succeeds.
        """
        import numpy as np
        from ConfigSpace import Configuration

        params = list(config_space.keys())

        data: list[tuple] = []
        for t in trials:
            try:
                cfg = Configuration(config_space, values=t.config)
                data.append((cfg, t.scores[metric_name]))
            except Exception:
                continue

        if len(data) < 2:
            uniform = 1.0 / len(params) if params else 0.0
            return (
                {p: uniform for p in params},
                "Not enough trials for importance estimation; showing uniform weights.",
            )

        try:
            task = ExplanationTask.from_data(config_space, data)
            hs = HyperSHAP(task)
            iv = hs.tunability()
            order1 = iv.get_n_order(order=1).dict_values
            raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
            total = sum(raw.values()) or 1.0
            return {k: v / total for k, v in raw.items()}, None
        except Exception as e:
            warning = f"HyperSHAP failed ({e}) — falling back to surrogate feature importances."

        try:
            from sklearn.ensemble import RandomForestRegressor
            X = np.array([cfg.get_array() for cfg, _ in data])
            y = np.array([score for _, score in data])
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(X, y)
            importances = rf.feature_importances_
            total = importances.sum() or 1.0
            return {p: float(importances[i] / total) for i, p in enumerate(params)}, warning
        except Exception as e2:
            warning += f" Surrogate fallback also failed ({e2}); showing uniform weights."
            uniform = 1.0 / len(params) if params else 0.0
            return {p: uniform for p in params}, warning
