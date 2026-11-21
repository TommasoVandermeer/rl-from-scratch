import jax.numpy as jnp
from jax import Array
from typing import Any
from abc import ABC, abstractmethod
from flax.struct import dataclass, field

@dataclass
class StepData:
    state: Any
    new_state: Any
    observation: Any
    reward: Any
    action: Any
    info: Any
    terminated: Array
    truncated: Array

@dataclass
class StepOut:
    state: Any
    observation: Any
    reward: Array
    info: Any
    terminated: Array 
    truncated: Array = field(default_factory=lambda: jnp.array(False))

@dataclass
class ResetOut:
    state: Any
    observation: Any
    parameters: Any

@dataclass
class RewardOut:
    reward: Array
    terminated: Array
    reward_info: Any

class BaseEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _get_obs(self, state: Any, parameters: Any) -> Any:
        pass

    @abstractmethod
    def step(self, state: Any, action: Any, parameters: Any, rng_key: Array) -> StepOut:
        pass

    @abstractmethod
    def reset(self, rng_key: Array) -> ResetOut:
        pass

    @abstractmethod
    def reward(self, state: Any, action: Any, new_state: Any, parameters: Any) -> RewardOut:
        pass