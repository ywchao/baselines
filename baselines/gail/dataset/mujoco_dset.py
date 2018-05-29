'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize, flattened=True):
        # `flattened` makes a difference only when `randomize` is set to True.
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.flattened = flattened
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(len(self.inputs))
            np.random.shuffle(idx)
            self._inputs = self.inputs[idx]
            self._labels = self.labels[idx]
        if not self.flattened:
            self._inputs = flatten(self._inputs)
            self._labels = flatten(self._labels)

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all

        if batch_size < 0:
            return self._inputs, self._labels
        if self.pointer + batch_size >= len(self._inputs):
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self._inputs[self.pointer:end, :]
        labels = self._labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True, obs_only=False):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])

        # TODO: Assume qpos initialization and randomizing at the sequence level
        # if using obs_only. Can untie these features later.
        self.obs_only = obs_only
        if self.obs_only:
            self.obs = traj_data['obs'][:traj_limitation]
            self.qpos = traj_data['qpos'][:traj_limitation]
            self.num_transition = sum([len(i) for i in self.obs])
            # TODO: Ensure a pre-fixed minimum `num_transition`. Should change
            # according to `timesteps_per_batch` in run_mujoco.py. 
            if self.num_transition < 1000:
                factor = int(np.ceil(1000 / self.num_transition))
                self.obs = np.tile(self.obs, factor)
                self.qpos = np.tile(self.qpos, factor)
                self.num_transition = sum([len(i) for i in self.obs])
        else:
            obs = traj_data['obs'][:traj_limitation]
            acs = traj_data['acs'][:traj_limitation]
            self.obs = np.array(flatten(obs))
            self.acs = np.array(flatten(acs))
            if len(self.acs) > 2:
                self.acs = np.squeeze(self.acs)
            assert len(self.obs) == len(self.acs)
            self.num_transition = len(self.obs)

        if 'ep_rets' in traj_data:
            self.rets = traj_data['ep_rets'][:traj_limitation]
        else:
            self.rets = np.array([0])
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))

        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.randomize = randomize
        if self.obs_only:
            self.dset = Dset(self.obs, self.qpos, self.randomize, flattened=False)
        else:
            self.dset = Dset(self.obs, self.acs, self.randomize)
            # for behavior cloning
            self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                                  self.acs[:int(self.num_transition*train_fraction), :],
                                  self.randomize)
            self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                                self.acs[int(self.num_transition*train_fraction):, :],
                                self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        assert not self.obs_only or split is None
        assert not self.obs_only or batch_size <= self.num_transition
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def flatten(x):
    # x.shape = (E,), or (E, L, D)
    _, size = x[0].shape
    episode_length = [len(i) for i in x]
    y = np.zeros((sum(episode_length), size))
    start_idx = 0
    for l, x_i in zip(episode_length, x):
        y[start_idx:(start_idx+l)] = x_i
        start_idx += l
    return y

def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
