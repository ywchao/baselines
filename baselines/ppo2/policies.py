import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class HierarchicalMlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        """
        Example of ob_space and ac_space structures:

        ob_space = gym.spaces.Dict(dict(
          switch=gym.spaces.Box(-1, 1, shape=(), dtype=np.int32),
          meta=gym.spaces.Box(-np.inf, np.inf, shape=(57,)),
          task1=gym.spaces.Box(-np.inf, np.inf, shape=(52,)),
          task2=gym.spaces.Box(-np.inf, np.inf, shape=(52,)),
        ))

        ac_space = gym.spaces.Dict(dict(
          meta=gym.spaces.Dict(dict(
            switch=gym.spaces.Discrete(2),
            task1=gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
            )),
          task1=gym.spaces.Box(-1, 1, shape=(21,)),
          task2=gym.spaces.Box(-1, 1, shape=(21,)),
        ))
        """
        from gym import spaces
        from functools import partial
        assert isinstance(ob_space, spaces.Dict)
        assert isinstance(ac_space, spaces.Dict)
        assert isinstance(ac_space.spaces['meta'], spaces.Dict)
        assert len(ob_space.spaces) == len(ac_space.spaces) + 1

        def map_space_dict(space, fn, nout=1):
            if isinstance(space, spaces.Dict):
                if nout == 1:
                    out = {}
                else:
                    out = [{} for _ in range(nout)]
                for k in space.spaces:
                    ret = map_space_dict(space.spaces[k], fn, nout)
                    if nout == 1:
                        out[k] = ret
                    else:
                        assert len(ret) == nout
                        for i in range(len(ret)):
                            out[i][k] = ret[i]
            else:
                out = fn(space)
            return out
        self.pdtype = map_space_dict(ac_space, make_pdtype)
        ob_space_mlp = spaces.Dict({k: v for k, v in ob_space.spaces.items() if k != "switch"})
        X, processed_x = map_space_dict(ob_space_mlp, partial(observation_input, batch_size=nbatch), nout=2)

        classes = list(ac_space.spaces.keys())
        classes.remove('meta')
        assert all([ac_space.spaces[k].shape == ac_space.spaces[classes[0]].shape  for k in classes])

        with tf.variable_scope("model", reuse=reuse):
            def _set_pd_pi(pdtype, h):
                if isinstance(pdtype, dict):
                    pd, pi = {}, {}
                    for k in sorted(pdtype):
                        with tf.variable_scope(k, reuse=reuse):
                            pd[k], pi[k] = _set_pd_pi(pdtype[k], h)
                else:
                    pd, pi = pdtype.pdfromlatent(h, init_scale=0.01)
                return pd, pi

            self.pd, self.pi = {}, {}
            for k in ['meta'] + classes:
                with tf.variable_scope(k, reuse=reuse):
                    activ = tf.tanh
                    inp = tf.layers.flatten(processed_x[k])
                    pi_h1 = activ(fc(inp, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
                    pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
                    if k == "meta":
                        vf_h1 = activ(fc(inp, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
                        vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
                        vf = fc(vf_h2, 'vf', 1)[:,0]

                    self.pd[k], self.pi[k] = _set_pd_pi(self.pdtype[k], pi_h2)


        s0 = {k: self.pd['meta'][k].sample() for k in self.pd['meta']}
        s0['switch'] = tf.cast(s0['switch'], tf.int32)

        def neglogp(a):
            n0 = {k: self.pd['meta'][k].neglogp(a[k]) for k in self.pd['meta']}
            for k in classes:
                if k not in self.pd['meta']:
                    n0[k] = tf.zeros_like(n0['switch'])
            sg = [n0[k] for k in classes]
            ig = tf.stack([a['switch'], tf.range(tf.shape(a['switch'])[0])], axis=-1)
            ng = tf.gather_nd(sg, ig)
            return tf.reduce_sum([n0['switch'], ng], axis=0)

        a0 = s0
        neglogp0 = neglogp(a0)
        self.initial_state = None

        X['switch'] = processed_x['switch'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32, name='Ob')
        switch = tf.where(tf.equal(processed_x['switch'], -1), s0['switch'], processed_x['switch'])

        s1 = [self.pd[k].sample() for k in classes]
        i1 = tf.stack([switch, tf.range(tf.shape(switch)[0])], axis=-1)
        a1 = tf.gather_nd(s1, i1)

        aenv0 = (switch, a1) + tuple([s0[k] for k in classes if k in s0])

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, aenv = sess.run([a0, vf, neglogp0, aenv0], {X[k]: ob[k] for k in X})
            return a, v, self.initial_state, neglogp, aenv

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X[k]: ob[k] for k in X})

        def sample_placeholder(prepend_shape, name=None):
            return {k: self.pdtype['meta'][k].sample_placeholder(prepend_shape) for k in self.pd['meta']}

        def entropy():
            return tf.reduce_sum([self.pd['meta'][k].entropy() for k in self.pd['meta']], axis=0)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.classes = classes
        self.sample_placeholder = sample_placeholder
        self.neglogp = neglogp
        self.entropy = entropy
        self.adtypes = {k: a0[k].dtype.as_numpy_dtype for k in a0}
