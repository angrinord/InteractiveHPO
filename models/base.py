from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace


class BaseModel(ABC):
    # Interface for models
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_config_space(self, seed: int = 0) -> ConfigurationSpace: ...

    @abstractmethod
    def train_evaluate(
        self,
        config: dict,
        X_train, y_train,
        X_val, y_val,
        metric_fn,
        seed: int = 0,
    ) -> float: ...