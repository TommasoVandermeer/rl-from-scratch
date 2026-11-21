from jax import Array, random, lax
import jax.numpy as jnp
from typing import Any

from rl_from_scratch.envs.base_env import StepData
from rl_from_scratch.envs.grid_world_env import GridWorldEnv
from rl_from_scratch.dynamics.grid_world import ACTION_LABELS, GridWorldAction

seed = 0
n_steps = 100

env = GridWorldEnv()
reset_out = env.reset(random.PRNGKey(seed))
parameters = reset_out.parameters
def scan_body(carry: tuple[Any, Any], key: Array) -> tuple[tuple[Any, Any], StepData]:
    state, observation = carry
    action_key, step_key = random.split(key)
    action = GridWorldAction(
        move = random.categorical(action_key, jnp.log(jnp.ones((4,))/4))
    )
    step_out = env.step(state, action, parameters, step_key)
    step_data = StepData(
        state = state,
        new_state = step_out.state,
        observation = step_out.observation,
        reward = step_out.reward,
        info = step_out.info,
        terminated = step_out.terminated,
        truncated = step_out.truncated,
        action = action,
    )
    return (step_out.state, step_out.observation), step_data
_, trajectory = lax.scan(
    scan_body, 
    (reset_out.state, reset_out.observation),
    random.split(random.PRNGKey(seed), n_steps),
    n_steps,
)
print(trajectory.state.position)
env.render(trajectory, parameters)