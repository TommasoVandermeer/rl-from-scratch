import jax.numpy as jnp
from jax import Array
from flax.struct import dataclass

ACTION_LABELS = {
    0: "UP",
    1: "LEFT",
    2: "DOWN",
    3: "RIGHT",
}
DISPLACEMENTS = jnp.array([
    [0, 1],     # UP
    [-1, 0],    # LEFT
    [0, -1],    # DOWN
    [1, 0],     # RIGHT
], dtype=jnp.int32)
LIMITS = jnp.array([
    [0, 5],
    [0, 5]
], dtype=jnp.int32)

## STATE 
@dataclass
class GridWorldState:
    position: Array # [x, y] \in [LIMITS[0,0], LIMITS[0,1]] x [LIMITS[1,0], LIMITS[1,1]]

## ACTION
@dataclass
class GridWorldAction:
    move: Array # a \in {0, 1, 2, 3}

def dynamics(state: GridWorldState, action: GridWorldAction) -> GridWorldState:
    new_state = GridWorldState(
        position=jnp.clip(
            state.position + DISPLACEMENTS[action.move],
            LIMITS[:,0],
            LIMITS[:,1]
        )
    )
    return new_state