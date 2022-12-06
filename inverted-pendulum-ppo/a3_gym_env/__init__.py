from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

_load_env_plugins()

register(
    id="Pendulum-v1-custom",
    entry_point="gym_set_state.envs:UpdatedPendulumEnv",
    reward_threshold=200,
)
