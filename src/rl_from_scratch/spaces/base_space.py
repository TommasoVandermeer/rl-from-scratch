from jax import Array
from flax.struct import dataclass

@dataclass
class Space:
    def sample(self, key: Array) -> Array:
        raise NotImplementedError

    def contains(self, x: Array) -> Array:
        raise NotImplementedError