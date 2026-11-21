from jax import Array, random, jit
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from rl_from_scratch.spaces.space import Space

@dataclass
class Box(Space):
    low: Array
    high: Array
    shape: tuple = ()

    @partial(jit, static_argnames=('self',))
    def sample(self, key: Array) -> Array:
        u = random.uniform(key, shape=self.low.shape)
        return self.low + u * (self.high - self.low)

    @partial(jit, static_argnames=('self',))
    def contains(self, x: Array) -> Array:
        return jnp.all(x >= self.low) & jnp.all(x <= self.high)