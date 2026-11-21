import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap, random, nn
from flax.struct import dataclass
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import patches, animation

from rl_from_scratch.envs.env import Env, StepOut, ResetOut, RewardOut, StepData
from rl_from_scratch.dynamics.grid_world import GridWorldState, GridWorldAction, dynamics, LIMITS
from rl_from_scratch.spaces.discrete import Discrete

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

class GridWorldEnv(Env):
    def __init__(self):
        self._action_space = Discrete(4)
        self._limits = LIMITS
        self._width = LIMITS[0,1] - LIMITS[0,0] + 1
        self._height = LIMITS[1,1] - LIMITS[1,0] + 1
        self._n_cells = self._width * self._height
        self.transitions = vmap(vmap(dynamics, in_axes=(None,0)), in_axes=(0,None))(
            vmap(self._idx_to_state)(jnp.arange(self._n_cells)),
            vmap(lambda idx: GridWorldAction(idx))(jnp.arange(4))
        )
        next_idx = vmap(self._state_to_idx)(self.transitions)
        self.p = nn.one_hot(next_idx, num_classes=int(self._n_cells))

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    # Private functions
    def _state_to_idx(self, state: GridWorldState):
        position = state.position
        return self._width * (position[0] - self._limits[0,0]) + (position[1] - self._limits[1,0])
    
    def _idx_to_state(self, idx: Array):
        y = idx % self._width + self._limits[1,0]
        x = idx // self._width + self._limits[0,0]
        return GridWorldState(jnp.array([x, y]))

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

    def _reward(self, state: GridWorldState, action: GridWorldAction, new_state: GridWorldState, parameters: GridWorldParameters) -> RewardOut:
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

    # Public functions
    def rewards_terminations(self, parameters: GridWorldParameters):
        rewards_out = vmap(vmap(self._reward, in_axes=(None,0,0,None)), in_axes=(0,None,0,None))(
            vmap(self._idx_to_state)(jnp.arange(self._n_cells)),
            vmap(lambda idx: GridWorldAction(idx))(jnp.arange(4)),
            self.transitions,
            parameters
        )
        return rewards_out.reward, rewards_out.terminated

    @partial(jit, static_argnames=('self',))
    def reward(self, state: GridWorldState, action: GridWorldAction, new_state: GridWorldState, parameters: GridWorldParameters) -> RewardOut:
        return self._reward(state, action, new_state, parameters)

    @partial(jit, static_argnames=('self',))
    def sample_action(self, key):
        move = self.action_space.sample(key)
        return GridWorldAction(move)

    @partial(jit, static_argnames=('self',))
    def step(self, state: GridWorldState, action: GridWorldAction, parameters: GridWorldParameters, rng_key: Array) -> StepOut:
        new_state = dynamics(state, action)
        reward_out = self._reward(state, action, new_state, parameters)
        observation = self._get_obs(new_state, parameters)
        return StepOut(
            state = new_state,
            observation = observation,
            reward = reward_out.reward,
            terminated = reward_out.terminated,
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
        state = GridWorldState(position)
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

    def render(self, trajectory: StepData, parameters: GridWorldParameters) -> None:
        positions = np.array(trajectory.state.position)
        obstacle_position = np.array(parameters.obstacles_position)
        goal_position = np.array(parameters.goal_position)
        # Plot specs
        fig, ax = plt.subplots(1,1)
        ax.set_aspect("equal")
        x_edges = jnp.arange(LIMITS[0, 0], LIMITS[0, 1] + 2)
        y_edges = jnp.arange(LIMITS[1, 0], LIMITS[1, 1] + 2)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_labels = [str(int(x)) for x in jnp.arange(LIMITS[0, 0], LIMITS[0, 1] + 1)]
        y_labels = [str(int(y)) for y in jnp.arange(LIMITS[1, 0], LIMITS[1, 1] + 1)]
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(y_edges, minor=True)
        ax.grid(True, which="minor", linestyle="--", alpha=0.7)
        ax.set_xticks(x_centers, labels=x_labels, minor=False)
        ax.set_yticks(y_centers, labels=y_labels, minor=False)
        ax.tick_params(axis="both", which="both", length=0)
        # Static objects
        ax.scatter(goal_position[0] + .5, goal_position[1] + .5, s=100, c='red', marker='*')
        obstacle = patches.Rectangle((obstacle_position[0], obstacle_position[1]), 1., 1., color='black')
        ax.add_patch(obstacle)
        # Dynamic objects
        robot = patches.Circle((positions[0,0] + .5, positions[0,1] + .5), radius=0.5)
        ax.add_patch(robot)
        # Animation
        def update(frame):
            robot.center = (positions[frame,0] + .5, positions[frame,1] + .5)
            return (robot,)
        animation.FuncAnimation(fig, update, frames=len(positions), interval=200, blit=False)
        plt.show()