from jax.random import PRNGKey
from typing import Any
from abc import ABC, abstractmethod
from flax.struct import dataclass

@dataclass
class StepOut:
    state:Any
    observation:Any
    reward:float
    info:Any
    terminated:bool
    truncated:bool

@dataclass
class ResetOut:
    state:Any
    observation:Any

@dataclass
class RewardOut:
    reward:float
    terminated:bool

class BaseEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, state:Any, action:Any, rng_key:PRNGKey) -> StepOut:
        pass

    @abstractmethod
    def reset(self, rng_key:PRNGKey) -> ResetOut:
        pass

    @abstractmethod
    def reward(self, state:Any, action:Any, new_state:Any) -> RewardOut:
        pass