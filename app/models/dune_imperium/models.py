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

CARD_INPUT_SIZE = 85
INTRIGUE_INPUT_SIZE = 52
TECH_INPUT_SIZE = 19
LEADER_INPUT_SIZE = 14
CONFLICT_INPUT_SIZE = 22

CONFLICT_OFFSET = 320
IMPERIUM_OFFSET = 342

PLAYER_INPUT_SIZE = 804
PLAYERS_OFFSET = 479
CURRENT_PLAYER_INPUT_SIZE = 3*INTRIGUE_INPUT_SIZE + CARD_INPUT_SIZE
CURRENT_PLAYER_OFFSET = 3695

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
            INPUT_SIZE = obs.shape[2]

            print(f"INPUT_SIZE = {INPUT_SIZE}")
            print(f"N_ACTIONS = {N_ACTIONS}")

            round_number = obs[:, 0:1]
            player = obs[:, 1:5]
            phase = obs[:, 5:11]
            spaces = obs[:, 11:22*14]
            mentat = obs[:, 22*14:22*14+1]
            reserve = obs[:, IMPERIUM_OFFSET:IMPERIUM_OFFSET+3]

            conflict = conflict_embedding_layer(obs[:, CONFLICT_OFFSET:CONFLICT_OFFSET+CONFLICT_INPUT_SIZE])
            imperium_row = card_embedding_layer(obs[:, IMPERIUM_OFFSET+3:IMPERIUM_OFFSET+3+CARD_INPUT_SIZE])
            techs = tech_embedding_layer(obs[:, IMPERIUM_OFFSET+3+CARD_INPUT_SIZE:IMPERIUM_OFFSET+3+CARD_INPUT_SIZE+TECH_INPUT_SIZE])
            trashed_intrigues = intrigue_embedding_layer(obs[:, IMPERIUM_OFFSET+3+CARD_INPUT_SIZE+TECH_INPUT_SIZE:IMPERIUM_OFFSET+3+CARD_INPUT_SIZE+TECH_INPUT_SIZE+INTRIGUE_INPUT_SIZE])

            # players
            player_1 = player_layer(obs[:, PLAYERS_OFFSET:PLAYERS_OFFSET + PLAYER_INPUT_SIZE])
            player_2 = player_layer(obs[:, PLAYERS_OFFSET + PLAYER_INPUT_SIZE:PLAYERS_OFFSET + 2 * PLAYER_INPUT_SIZE])
            player_3 = player_layer(obs[:, PLAYERS_OFFSET + 2 * PLAYER_INPUT_SIZE:PLAYERS_OFFSET + 3 * PLAYER_INPUT_SIZE])
            player_4 = player_layer(obs[:, PLAYERS_OFFSET + 3 * PLAYER_INPUT_SIZE:PLAYERS_OFFSET + 4 * PLAYER_INPUT_SIZE])

            # current player
            current_player = current_player_layer(obs[:, CURRENT_PLAYER_OFFSET:CURRENT_PLAYER_INPUT_SIZE])

            embedded_obs = tf.concat([
                round_number,
                player,
                phase,
                spaces,
                mentat,
                reserve,
                conflict,
                imperium_row,
                techs,
                trashed_intrigues,
                player_1,
                player_2,
                player_3,
                player_4,
                current_player,
            ], axis=-1)

            # feature extraction
            EMBEDDING_SIZE = embedded_obs.shape[2]

            self._pdtype = CategoricalProbabilityDistributionType(N_ACTIONS)
            extracted_features = resnet_extractor(embedded_obs, EMBEDDING_SIZE)

            print(f"extracted_features {extracted_features}")

            self._policy = policy_head(extracted_features, legal_actions, EMBEDDING_SIZE, N_ACTIONS)
            self._value_fn, self.q_value = value_head(extracted_features, EMBEDDING_SIZE, N_ACTIONS)

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


def resnet_extractor(y, n):
    y = dense(y, n)
    for _ in range(DEPTH):
        y = residual(y, n)
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


def card_embedding_layer(inputs):
    return tf.keras.layers.Embedding(85, 16, name='card_embedding')(inputs)

def intrigue_embedding_layer(inputs):
    return tf.keras.layers.Embedding(52, 16, name='intrigue_embedding')(inputs)

def tech_embedding_layer(inputs):
    return tf.keras.layers.Embedding(19, 8, name='tech_embedding')(inputs)

def conflict_embedding_layer(inputs):
    return tf.keras.layers.Embedding(22, 8, name='conflict_embedding')(inputs)

def leader_embedding_layer(inputs):
    return tf.keras.layers.Embedding(14, 8, name='card_embedding')(inputs)

def player_layer(inputs):
    return tf.concat([
        leader_embedding_layer(inputs[:, 0:LEADER_INPUT_SIZE]),
        inputs[:, LEADER_INPUT_SIZE:30],    # scalars
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE+30:LEADER_INPUT_SIZE+30+CARD_INPUT_SIZE]),                                 # discard
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 2 * CARD_INPUT_SIZE]),     # hand
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 2 * CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 3 * CARD_INPUT_SIZE]), # deck
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 3 * CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 4 * CARD_INPUT_SIZE]), # top 1
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 4 * CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 5 * CARD_INPUT_SIZE]), # top 2
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 5 * CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE]), # in play
        intrigue_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE]), # in play intrigues
        tech_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + TECH_INPUT_SIZE]), # techs
        tech_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + TECH_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2*TECH_INPUT_SIZE]), # activated techs
        inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2*TECH_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2*TECH_INPUT_SIZE + 5],   # baron
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5 + CARD_INPUT_SIZE]), # helena
        card_embedding_layer(inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5 + CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5 + 2 * CARD_INPUT_SIZE]), # ilesa
        inputs[:, LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5 + 2*CARD_INPUT_SIZE:LEADER_INPUT_SIZE + 30 + 6 * CARD_INPUT_SIZE + INTRIGUE_INPUT_SIZE + 2 * TECH_INPUT_SIZE + 5 + 2*CARD_INPUT_SIZE + 4], # tessia
    ], axis=-1)

def current_player_layer(inputs):
    return tf.concat([
        intrigue_embedding_layer(inputs[:, 0:INTRIGUE_INPUT_SIZE]),
        intrigue_embedding_layer(inputs[:, INTRIGUE_INPUT_SIZE:2*INTRIGUE_INPUT_SIZE]),
        intrigue_embedding_layer(inputs[:, 2*INTRIGUE_INPUT_SIZE:3*INTRIGUE_INPUT_SIZE]),
        card_embedding_layer(inputs[:, 3*INTRIGUE_INPUT_SIZE:3*INTRIGUE_INPUT_SIZE+CARD_INPUT_SIZE])
    ], axis=-1)
