from jax import Array, random, jit
from flax.struct import dataclass
from functools import partial

from rl_from_scratch.spaces.space import Space

@dataclass
class Discrete(Space):
    n: int # Number of available actions

    @partial(jit, static_argnames=('self',))
    def sample(self, key: Array) -> Array:
        return random.randint(key, (), minval=0, maxval=self.n)

    @partial(jit, static_argnames=('self',))
    def contains(self, x: Array) -> Array:
        return (x >= 0) & (x < self.n)