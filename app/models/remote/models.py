import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Add, Dense, Multiply, Concatenate, Lambda
import tensorflow.keras.backend as K
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistribution

# Optionally configurable

DEPTH = 5
VALUE_DEPTH = 1
POLICY_DEPTH = 1


class CustomPolicy(ActorCriticPolicy):
    """
        Policy object that implements actor critic

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param reuse: (bool) If the policy is reusable or not
        :param scale: (bool) whether or not to scale the input
        """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        ACTIONS = ac_space.n
        FEATURE_SIZE = ob_space.shape[0] - ACTIONS

        with tf.variable_scope("model", reuse=reuse):

            obs, legal_actions = split_input(self.processed_obs, ACTIONS)

            extracted_features = resnet_extractor(obs, FEATURE_SIZE, **kwargs)

            self._policy = policy_head(extracted_features, legal_actions, FEATURE_SIZE, ACTIONS)
            self._value_fn, self.q_value = value_head(extracted_features, FEATURE_SIZE, ACTIONS)
            self._proba_distribution  = CategoricalProbabilityDistribution(self._policy)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        # value flat is self.value_fn[:, 0]
        # I should split obs in the various states, feed them to value_flat one by one, then apply one last layer to
        # produce a single value. Similar for the step and proba_step. But for sure this is problematic with batches, as
        # each sample will have different number of states!
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def split_input(obs, split):
    return   obs[:,:-split], obs[:,-split:]


def value_head(y, FEATURE_SIZE, ACTIONS):
    for _ in range(VALUE_DEPTH):
        y = dense(y, FEATURE_SIZE)
    vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    q = dense(y, ACTIONS, batch_norm = False, activation = 'tanh', name='q')
    return vf, q


def policy_head(y, legal_actions, FEATURE_SIZE, ACTIONS):

    for _ in range(POLICY_DEPTH):
        y = dense(y, FEATURE_SIZE)
    policy = dense(y, ACTIONS, batch_norm = False, activation = None, name='pi')

    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)

    policy = Add()([policy, mask])
    return policy


def resnet_extractor(y, FEATURE_SIZE, **kwargs):
    y = dense(y, FEATURE_SIZE)
    for _ in range(DEPTH):
        y = residual(y, FEATURE_SIZE)

    return y


def residual(y, filters):
    shortcut = y

    y = dense(y, filters)
    y = dense(y, filters, activation = None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y


def dense(y, filters, batch_norm = False, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)

    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)

    return y
