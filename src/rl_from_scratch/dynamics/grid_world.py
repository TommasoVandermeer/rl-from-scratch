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
    [-10, 10],
    [-10, 10]
], dtype=jnp.int32)

## STATE 
@dataclass
class GridWorldState:
    position: Array # [x, y] \in [LIMITS[0,0], LIMITS[0,1]] x [LIMITS[1,0], LIMITS[1,1]]

## ACTION
@dataclass
class GridWorldAction:
    move: int # a \in {0, 1, 2, 3}

def dynamics(state:GridWorldState, action: GridWorldAction) -> GridWorldState:
    new_state = GridWorldState(
        position=jnp.clip(
            state.position + DISPLACEMENTS[action.move],
            LIMITS[:,0],
            LIMITS[:,1]
        )
    )
    return new_state

# DEBUG
# for action in range(len(DISPLACEMENTS)):
#     print(
#         f"From [{0},{0}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(
#             GridWorldState(position=jnp.zeros((2,))), 
#             GridWorldAction(move=action),
#         ).position
#     )
#     print(
#         f"From [{LIMITS[0,0]},{LIMITS[1,0]}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(
#             GridWorldState(position=jnp.array([LIMITS[0,0],LIMITS[1,0]])), 
#             GridWorldAction(move=action)
#         ).position
#     )
#     print(
#         f"From [{LIMITS[0,1]},{LIMITS[1,1]}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(
#             GridWorldState(position=jnp.array([LIMITS[0,1],LIMITS[1,1]])), 
#             GridWorldAction(move=action)
#         ).position
# )