from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hypershap import ExplanationTask, HyperSHAP

@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    score: float            # metric value for this config
    incumbent_score: float  # best score seen so far
    incumbent_config: Dict[str, Any]


@dataclass
class OptimizationResult:
    trials: List[TrialResult]
    best_config: Dict[str, Any]
    best_score: float
    hyperparameter_importance: Optional[Dict[str, float]] = None
    hyperparameter_importance_warning: Optional[str] = None
    # Optimizer-specific state needed to resume a run (e.g. output directory path).
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def optimize(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        metric_fn,
        n_trials: int,
        previous_result: Optional["OptimizationResult"] = None,
        seed: int = 0,
    ) -> OptimizationResult: ...

    def compute_hp_importance(
        self,
        config_space,
        trials: List[TrialResult],
        seed: int = 0,
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Estimate hyperparameter importance from a completed list of trials.

        Optimizer-agnostic: reconstructs ConfigSpace Configuration objects
        from the plain config dicts stored in each TrialResult, then tries
        HyperSHAP followed by an RF-based fallback.

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
                data.append((cfg, t.score))
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
            # Extract per-feature (order-1) Shapley values; keys are (index,) tuples.
            order1 = iv.get_n_order(order=1).dict_values
            raw = {params[idx]: abs(val) for (idx,), val in order1.items()}
            total = sum(raw.values()) or 1.0
            return {k: v / total for k, v in raw.items()}, None
        except Exception as e:
            warning = f"HyperSHAP failed ({e}) — falling back to surrogate feature importances."

        # Fallback: train an RF on the trial data.
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
