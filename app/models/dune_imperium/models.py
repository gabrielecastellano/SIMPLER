import types
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.layers import (BatchNormalization, Activation, Flatten, Add, Dense, Multiply, Concatenate, Lambda,
                                     Maximum, Reshape)
import tensorflow.keras.backend as K
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import (CategoricalProbabilityDistribution,
                                                   CategoricalProbabilityDistributionType)

# Optionally configurable
DEPTH = 1
VALUE_DEPTH = 1
POLICY_DEPTH = 1


class CustomPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic for value estimation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)
        # TODO remove super call and do everything here?

        print(f"ac_space.shape = {ac_space.shape} {ac_space}")

        # override BasePolicy to add placeholder
        with tf.variable_scope("input", reuse=False):
            self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=False, name="observation")
            self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch, None), name="action_ph")
            obs, legal_actions = split_input(self.processed_obs)

            print(f"self._obs_ph = {self._obs_ph}")
            print(f"self._processed_obs = {self._processed_obs}")
            print(f"self._action_ph = {self._action_ph}")

        with tf.variable_scope("model", reuse=reuse):

            # Determine the number of actions dynamically based on the first dimension of ob_space
            N_ACTIONS = tf.shape(obs)[1]
            FEATURE_SIZE = obs.shape[2]

            print(f"FEATURE_SIZE = {FEATURE_SIZE}")
            print(f"N_ACTIONS = {N_ACTIONS}")

            self._pdtype = CategoricalProbabilityDistributionType(N_ACTIONS)
            extracted_features = resnet_extractor(obs, FEATURE_SIZE)

            print(f"extracted_features  {extracted_features}")

            self._policy = policy_head(extracted_features, legal_actions, FEATURE_SIZE, N_ACTIONS)
            self._value_fn, self.q_value = value_head(extracted_features, FEATURE_SIZE, N_ACTIONS)

            self._proba_distribution = CategoricalProbabilityDistribution(self._policy)
            # override instance neglogp method as library one does not work with dynamic sized output
            self._proba_distribution.neglogp = types.MethodType(neglogp, self._proba_distribution)

            print(f"self._policy {self._policy}")
            print(f"self._value_fn {self._value_fn}")
            print(f"self._proba_distribution {self._proba_distribution}")


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


def split_input(input_):
    return input_[:, :, :-1], tf.squeeze(input_[:, :, -1:], axis=-1)


def value_head(y, FEATURE_SIZE, N_ACTIONS):

    for _ in range(VALUE_DEPTH):
        y = dense(y, FEATURE_SIZE)
    # value
    vf = dense(y, 1, batch_norm=False, activation='tanh')
    vf = Reshape(target_shape=(N_ACTIONS,))(vf)
    vf = K.max(vf, axis=-1, keepdims=True)
    vf._name = "/".join((lambda x: x[:1] + ["vf"] + x[1:])(vf.name.split("/")))

    # q values
    q = dense(y, 1, batch_norm=False, activation='tanh')
    q = Reshape(target_shape=(N_ACTIONS,), name='q')(q)

    return vf, q


def policy_head(y, legal_actions, FEATURE_SIZE, N_ACTIONS):

    for _ in range(POLICY_DEPTH):
        y = dense(y, FEATURE_SIZE)
    y = dense(y, 1, batch_norm=False, activation=None)
    policy = Reshape(target_shape=(N_ACTIONS,), name='pi')(y)
    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)
    policy = Add()([policy, mask])

    return policy


def resnet_extractor(y, FEATURE_SIZE):
    y = dense(y, FEATURE_SIZE)
    for _ in range(DEPTH):
        y = residual(y, FEATURE_SIZE)
    return y


def residual(y, filters):
    shortcut = y
    y = dense(y, filters)
    y = dense(y, filters, activation=None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)
    return y


def dense(y, filters, batch_norm=False, activation: object = 'relu', name=None):
    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name=name)(y)

    if batch_norm:
        if activation:
            y = BatchNormalization(momentum=0.9)(y)
        else:
            y = BatchNormalization(momentum=0.9, name=name)(y)

    if activation:
        y = Activation(activation, name=name)(y)

    return y


def observation_input(ob_space, batch_size=None, name='Ob', scale=False):
    # TODO check if shape is correctly defined
    observation_ph = tf.placeholder(shape=(batch_size, None, ob_space.shape[1]), dtype=ob_space.dtype, name=name)
    processed_observations = tf.cast(observation_ph, tf.float32)
    # rescale to [1, 0] if the bounds are defined
    if (scale and not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
            np.any((ob_space.high - ob_space.low) != 0)):
        # FIXME does not work (it stops being a placeholder)
        # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
        processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
    return observation_ph, processed_observations


def neglogp(self, x):
    one_hot_actions = tf.one_hot(x, tf.shape(self.logits)[-1])
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                      labels=tf.stop_gradient(one_hot_actions))
