from jax import jit, Array, vmap, nn, lax
from functools import partial
from typing import Any
from flax.struct import dataclass
import jax.numpy as jnp

from rl_from_scratch.agents.agent import Agent, UpdateOut

@dataclass
class DPState:
    V: Array # State value function (states,) \in [-inf,+inf]
    policy: Array # Policy (states, actions) \in [0,1]

@dataclass
class DPData:
    p: Array # Transition probabilities (states, actions, states) \in [0,1]
    r: Array # Rewards (states, actions) \in [-inf,+inf]
    t: Array # Terminations (states, actions) \in {0,1}

class DP(Agent):
    def __init__(
            self, 
            n_states: int, 
            n_actions: int, 
            gamma: float = 0.99,
            eval_threshold: float = 1e-6,
            eval_max_iterations: int = 500,
        ):
        super().__init__(gamma)
        self.n_states = n_states
        self.n_actions = n_actions
        self._eval_threshold = eval_threshold
        self._eval_max_iterations = eval_max_iterations

    def _Q(self, p: Array, r: Array, t: Array, V: Array) -> Array:
        return jnp.sum(
            vmap(
                lambda p, r, t, V: p * (r + self.gamma * (1-t) * V), 
                in_axes=(2,None,None,0)
            )(p, r, t, V), # (states', states, actions)
            axis=0,
        )

    def _policy_evaluation_step(self, policy: Array, p: Array, r: Array, t: Array, V: Array) -> tuple[Array, Array]:
        Q = self._Q(p, r, t, V)
        V_new = jnp.sum(policy * Q, axis=1)
        delta = jnp.max(jnp.abs(V_new - V))
        return V_new, delta

    def policy_improvement(self, p: Array, r: Array, t: Array, V: Array) -> Array:
        Q = self._Q(V, p, r, t)
        greedy_actions = jnp.argmax(Q, axis=1)
        return nn.one_hot(greedy_actions, num_classes=self.n_actions, dtype=V.dtype)

    def policy_evaluation(self, V: Array, policy: Array, p: Array, r: Array, t: Array):
        init_carry = (V, jnp.array(jnp.inf, dtype=V.dtype), jnp.array(0, dtype=jnp.int32))
        def cond_fn(carry):
            _, delta, iters = carry
            return jnp.logical_and(delta >= self._eval_threshold, iters < self._eval_max_iterations)
        def body_fn(carry):
            V_curr, _, iters = carry
            V_next, delta = self._policy_evaluation_step(policy, p, r, t, V_curr)
            return (V_next, delta, iters + 1)
        V_eval, final_delta, eval_iters = lax.while_loop(cond_fn, body_fn, init_carry)
        return V_eval, final_delta, eval_iters

    @partial(jit, static_argnames=('self',))
    def act(self, agent_state: DPState, observation: Any) -> Any:
        pass

    @partial(jit, static_argnames=('self',))
    def update(self, agent_state: DPState, update_data: DPData) -> UpdateOut:
        V = agent_state.V
        policy = agent_state.policy
        p = update_data.p
        r = update_data.r
        t = update_data.t
        # Policy evaluation
        V_eval, delta, iterations = self.policy_evaluation(V, policy, p, r, t)
        # Policy improvement
        policy_new = self.policy_improvement(p, r, t, V_eval)
        changed = jnp.any(jnp.argmax(policy_new, axis=1) != jnp.argmax(policy, axis=1))
        return UpdateOut(
            DPState(V_eval, policy_new),
            {
                "eval_delta": delta,
                "eval_iterations": iterations,
                "policy_changed": changed,
            }
        )