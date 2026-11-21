from jax import jit
from functools import partial
from typing import Any

from rl_from_scratch.agents.agent import Agent, UpdateOut

class DP(Agent):
    def __init__(self, gamma: float = 0.99):
        super().__init__(gamma)

    @partial(jit, static_argnames=('self'))
    def act(self, agent_state: Any, observation: Any) -> Any:
        pass

    @partial(jit, static_argnames=('self'))
    def update(self, update_data: Any) -> UpdateOut:
        pass