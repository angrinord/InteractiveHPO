from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace


class BaseModel(ABC):
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
        metrics: dict,   # {metric_name: callable}
        seed: int = 0,
    ) -> dict: ...       # {metric_name: score}
