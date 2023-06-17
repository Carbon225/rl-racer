from brax import geometry
from brax import kinematics
from brax.base import System
from brax.generalized import constraint
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax.generalized.base import State
import jax.numpy as jnp


def generalized_direct_step(
    sys: System, state: State, tau: jnp.ndarray, debug: bool = False
) -> State:
    state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
    state = state.replace(qf_constraint=constraint.force(sys, state))
    state = integrator.integrate(sys, state)
    x, xd = kinematics.forward(sys, state.q, state.qd)
    state = state.replace(x=x, xd=xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = constraint.jacobian(sys, state)

    if debug:
        state = state.replace(contact=geometry.contact(sys, state.x))

    return state
