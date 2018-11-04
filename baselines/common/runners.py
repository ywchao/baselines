import numpy as np
from abc import ABC, abstractmethod

from gym import spaces

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            self.batch_ob_shape = None
            self.obs = { k: np.zeros((nenv,) + v.shape, dtype=v.dtype.name) for k, v in obs_space.spaces.items() }
            obs = env.reset()
            for k in self.obs:
                self.obs[k][:] = obs[k]
        else:
            self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
            self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError
