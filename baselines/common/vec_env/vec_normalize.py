from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

from gym import spaces

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, skip_rms=None):
        VecEnvWrapper.__init__(self, venv)
        obs_space = venv.observation_space
        if isinstance(obs_space, spaces.Dict):
            self.ob_rms = { k: RunningMeanStd(shape=v.shape) if skip_rms is None or k not in skip_rms else v for k, v in obs_space.spaces.items() } if ob else None
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        if not np.any(np.isnan(rews)):
            self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if not np.any(np.isnan(rews)) and self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if isinstance(self.ob_rms, dict):
                for k in self.ob_rms:
                    if isinstance(self.ob_rms[k], RunningMeanStd):
                        self.ob_rms[k].update(obs[k])
                        obs[k] = np.clip((obs[k] - self.ob_rms[k].mean) / np.sqrt(self.ob_rms[k].var + self.epsilon), -self.clipob, self.clipob)
            else:
                self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)
