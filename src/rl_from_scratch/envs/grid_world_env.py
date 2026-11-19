import jax.numpy as jnp
from jax import Array, jit, vmap, random
from flax.struct import dataclass
from functools import partial

from rl_from_scratch.envs.base_env import BaseEnv, StepOut, ResetOut, RewardOut
from rl_from_scratch.dynamics.grid_world import GridWorldState, GridWorldAction, dynamics, LIMITS

SUCCESS_REWARD = 1.
FAILURE_PENALTY = 1.

WIDTH = LIMITS[0,1] - LIMITS[0,0]
HEIGHT = LIMITS[1,1] - LIMITS[1,0]

@dataclass
class GridWorldParameters:
    goal_position: Array
    obstacles_position: Array

@dataclass
class GridWorldObservations:
    position: Array
    task_parameters: GridWorldParameters

class GridWorldEnv(BaseEnv):
    def __init__(self):
        pass

    # Private functions
    def _sample_2d_categorical(self, key: Array, logits: Array) -> tuple:
        W, H = logits.shape
        flat_idx = random.categorical(key, logits.ravel())
        row, col = jnp.divmod(flat_idx, W)
        return row, col, flat_idx

    def _get_obs(self, state: GridWorldState, parameters: GridWorldParameters) -> GridWorldObservations:
        return GridWorldObservations(
            position = state.position,
            task_parameters = parameters,
        )

    # Public functions
    def reward(self, state: GridWorldState, action: GridWorldAction, new_state: GridWorldState, parameters: GridWorldParameters) -> RewardOut:
        success = jnp.all(new_state.position == parameters.goal_position)
        obs = jnp.atleast_2d(parameters.obstacles_position)
        collision = jnp.any(jnp.all(new_state.position == obs, axis=-1))
        return RewardOut(
            reward = jnp.asarray(success, dtype=jnp.float32) * jnp.asarray(SUCCESS_REWARD, dtype=jnp.float32) - jnp.asarray(collision, dtype=jnp.float32) * jnp.asarray(FAILURE_PENALTY, dtype=jnp.float32),
            terminated = jnp.logical_or(success, collision),
            reward_info = {
                "success": success,
                "collision": collision,
            }
        )

    @partial(jit, static_argnames=('self',))
    def step(self, state: GridWorldState, action: GridWorldAction, parameters: GridWorldParameters, rng_key: Array) -> StepOut:
        new_state = dynamics(state, action)
        reward_out = self.reward(state, action, new_state, parameters)
        observation = self._get_obs(new_state, parameters)
        return StepOut(
            state = new_state,
            observation = observation,
            reward = reward_out.reward,
            terminated = reward_out.terminated,
            truncated = jnp.zeros((1,), dtype=jnp.bool),
            info = reward_out.reward_info,
        )

    @partial(jit, static_argnames=('self',))
    def reset(self, rng_key: Array) -> ResetOut:
        pos_key, goal_key, obs_key = random.split(rng_key, 3)
        # SAMPLE POSITION
        pos_logits = jnp.log(jnp.ones((WIDTH, HEIGHT))/(WIDTH*HEIGHT))
        pos_x_idx, pos_y_idx, pos_flat_idx = self._sample_2d_categorical(pos_key, pos_logits)
        position = jnp.array([
            LIMITS[0,0] + pos_x_idx,
            LIMITS[1,0] + pos_y_idx,
        ])
        # SAMPLE GOAL
        goal_logits = jnp.log(jnp.ones((WIDTH, HEIGHT))/(WIDTH*HEIGHT -1))
        goal_logits = goal_logits.at[pos_flat_idx].set(-jnp.inf)
        goal_x_idx, goal_y_idx, goal_flat_idx = self._sample_2d_categorical(goal_key, goal_logits)
        goal_position = jnp.array([
            LIMITS[0,0] + goal_x_idx,
            LIMITS[1,0] + goal_y_idx,
        ])
        # OBSTACLES
        obstacle_logits = jnp.log(jnp.ones((WIDTH, HEIGHT))/(WIDTH*HEIGHT -2))
        obstacle_logits = obstacle_logits.at[pos_flat_idx].set(-jnp.inf)
        obstacle_logits = obstacle_logits.at[goal_flat_idx].set(-jnp.inf)
        obs_x_idx, obs_y_idx, _ = self._sample_2d_categorical(obs_key, obstacle_logits)
        obstacles_position = jnp.array([
            LIMITS[0,0] + obs_x_idx,
            LIMITS[1,0] + obs_y_idx,
        ])
        # STATE
        state = GridWorldState(position=position)
        # PARAMETERS
        parameters = GridWorldParameters(
            goal_position = goal_position,
            obstacles_position = obstacles_position,
        )
        # OBSERVATION
        observation = self._get_obs(state, parameters)
        return ResetOut(
            state = state,
            observation = observation,
            parameters = parameters
        )


# DEBUG
# from rl_from_scratch.dynamics.grid_world import ACTION_LABELS
# env = GridWorldEnv()
# reset_data = env.reset(random.PRNGKey(0))
# state = reset_data.state
# observation = reset_data.observation
# parameters = reset_data.parameters
# action = 0
# step_data = env.step(
#     state,
#     GridWorldAction(move=action),
#     parameters,
#     random.PRNGKey(0)
# )
# print(
#     f"Initial state: {state.position}\n",
#     f"Goal: {parameters.goal_position}\n",
#     f"Obstacle: {parameters.obstacles_position}\n",
#     f"Action: {ACTION_LABELS[action]}\n",
#     f"New state: {step_data.state.position}"
# )