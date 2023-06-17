from brax import base
from brax.base import Transform, Force, Motion
from brax.envs import PipelineEnv, State
from brax.io import mjcf

import jax
import jax.numpy as jnp

from pathlib import Path

from rl_racer.direct_step import generalized_direct_step


class HoverV1(PipelineEnv):
    def __init__(
        self,
        n_frames: int = 5,
        reset_noise_scale=1.0,
        max_ang_vel=jnp.array([13.962634, 13.962634, 10.471976]),
        max_ang_acc=jnp.array([13.962634, 13.962634, 10.471976]),
        gravity: float = 9.81,
        hover_throttle: float = 0.25,
        hover_target=jnp.array([0.0, 0.0, 1.0]),
        **kwargs,
    ):
        path = Path(__file__).parent / 'scene.xml'
        sys = mjcf.load(path)

        super().__init__(sys=sys, backend='generalized', n_frames=n_frames, **kwargs)

        inertia = sys.link.inertia.i[0].diagonal()
        mass = sys.link.inertia.mass[0]

        self._reset_noise_scale = reset_noise_scale
        self._hover_target = hover_target
        self._max_ang_vel = max_ang_vel
        self._torque = inertia * max_ang_acc
        self._max_thrust = mass * gravity / hover_throttle

    @property
    def action_size(self) -> int:
        return 4

    def reset(self, rng: jnp.ndarray) -> State:
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q.at[:2].add(jax.random.uniform(rng, (2,), minval=low, maxval=hi))
        qd = jnp.zeros((self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        distance = jnp.linalg.norm(pipeline_state.q[:3] - self._hover_target)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward': zero,
            'distance': distance,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        # pipeline

        pipeline_state0 = state.pipeline_state
        drone_rot = Transform.create(rot=pipeline_state0.q[3:])

        throttle = action[0]
        thrust = (throttle + 1.0) * 0.5 * self._max_thrust
        force_vec = jnp.array([0.0, 0.0, thrust])
        force_vec = drone_rot.do(Force.create(vel=force_vec)).vel
        
        target_ang_vel = action[1:] * self._max_ang_vel
        current_ang_vel = pipeline_state0.qd[3:]
        torque_direction = target_ang_vel - current_ang_vel
        torque_vec = torque_direction * self._torque

        tau = jnp.concatenate([force_vec, torque_vec])
        pipeline_state = self.pipeline_step(pipeline_state0, tau)

        # reward

        drone_xyz = pipeline_state.q[:3]
        distance = jnp.linalg.norm(drone_xyz - self._hover_target)
        reward = state.metrics['distance'] - distance

        # observation

        obs = self._get_obs(pipeline_state)

        # --

        state.metrics.update(
            reward=reward,
            distance=distance,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward
        )

    def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
        drone_rot = pipeline_state.q[3:]
        drone_rot = Transform.create(rot=drone_rot)

        drone_vel = pipeline_state.qd[:3]
        drone_vel = Motion.create(vel=drone_vel)
        drone_vel = drone_rot.do(drone_vel).vel

        drone_ang = pipeline_state.qd[3:]

        target_vec = self._hover_target - pipeline_state.q[:3]
        target_vec = target_vec / jnp.linalg.norm(target_vec)
        target_vec = drone_rot.do(Transform.create(pos=target_vec)).pos

        return jnp.concatenate([
            drone_vel,
            drone_ang,
            target_vec,
        ])

    def pipeline_step(
        self, pipeline_state, tau: jnp.ndarray
    ) -> base.State:
        def f(state, _):
            return (
                generalized_direct_step(self.sys, state, tau, self._debug),
                None,
            )
        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]
