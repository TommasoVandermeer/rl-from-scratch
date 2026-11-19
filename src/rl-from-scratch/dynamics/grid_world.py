import jax.numpy as jnp
from jax import Array

ACTION_LABELS = {
    0: "up",
    1: "left",
    2: "down",
    3: "right",
}
DISPLACEMENTS = {
    0: jnp.array([0, 1], dtype=jnp.int32),  # UP
    1: jnp.array([-1, 0], dtype=jnp.int32), # LEFT
    2: jnp.array([0, -1], dtype=jnp.int32), # DOWN
    3: jnp.array([1, 0], dtype=jnp.int32),  # RIGHT
}
LIMITS = jnp.array([
    [-10, 10],
    [-10, 10]
], dtype=jnp.int32)

## STATE [x, y] \in [LIMITS[0,0], LIMITS[0,1]] x [LIMITS[1,0], LIMITS[1,1]]
## ACTION a \in {0, 1, 2, 3}

def dynamics(state:Array, action: int) -> Array:
    new_state = state + DISPLACEMENTS[action]
    new_state = new_state.at[0].set(jnp.clip(new_state[0], LIMITS[0,0], LIMITS[0,1]))
    new_state = new_state.at[1].set(jnp.clip(new_state[1], LIMITS[1,0], LIMITS[1,1]))
    return new_state

# DEBUG
# for action in range(len(DISPLACEMENTS)):
#     print(
#         f"From [{0},{0}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(jnp.zeros((2,)), action)
#     )
#     print(
#         f"From [{LIMITS[0,0]},{LIMITS[1,0]}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(jnp.array([LIMITS[0,0],LIMITS[1,0]]), action)
#     )
#     print(
#         f"From [{LIMITS[0,1]},{LIMITS[1,1]}] applying {ACTION_LABELS[action]}: ", 
#         dynamics(jnp.array([LIMITS[0,1],LIMITS[1,1]]), action)
#     )