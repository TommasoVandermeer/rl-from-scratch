from typing import Any
from abc import ABC, abstractmethod
from flax.struct import dataclass

@dataclass
class UpdateOut:
    agent_state: Any
    info: Any

class Agent(ABC):
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    @abstractmethod
    def act(self, agent_state: Any, observation: Any) -> Any:
        pass

    @abstractmethod
    def update(self, agent_state: Any, update_data: Any) -> UpdateOut:
        pass