import os
import gym
import numpy as np
import requests


BASE_URL = os.getenv('REMOTE_BASE_URL', 'http://localhost:8000')
ACTION_SPACE_SIZE_PLACEHOLDER = 2

class DuneImperiumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=False, manual=False):
        super(DuneImperiumEnv, self).__init__()
        self.current_observation = None
        self.current_player_num = None
        self.n_players = None
        self.remote_id = None
        self.name = 'dune_imperium'
        self.manual = manual
        self.session = requests.Session()
        self.new_game()

        #self.action_space = gym.spaces.Discrete(10)
        #self.observation_space = gym.spaces.Box(-1, 1, (self.observation_space_size,))
        self.verbose = verbose

    @property
    def observation(self):
        return self.current_observation

    @property
    def legal_actions(self):
        return np.ones(self.current_observation.shape[0], dtype=np.int32)

    def reset(self):
        self.new_game()
        return self.observation

    def render(self, mode='human', close=False):
        # skipping extra network calls for now - uncomment to debug
        # self.remote_call('render')
        pass

    def remote_call(self, endpoint, data=None):
        url = f'{BASE_URL}/{endpoint}/{self.remote_id}'
        if data is None:
            return self.session.get(url).json()
        else:
            return self.session.post(url, json=data).json()

    def new_game(self):
        response = self.session.get(f'{BASE_URL}/new_game').json()
        self.remote_id = response['id']
        # self.action_space_size = response['action_space_size']
        self.current_observation = self.get_observation_from_payload(response)
        self.n_players = response['player_count']
        self.current_player_num = response['current_player']
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE_PLACEHOLDER)
        self.observation_space = gym.spaces.Box(-1, 1, (ACTION_SPACE_SIZE_PLACEHOLDER, self.current_observation.shape[1]))

    def step(self, action):
        try:
            response = self.remote_call('step', data={"action": int(action)})
        except:
            print(f'>>> Non-200 response! Potentially invalid move.\n'
                  f'observation: {self.observation}\n'
                  f'action: {action}\n')
            raise
        self.current_player_num = response['next_player']
        self.current_observation = self.get_observation_from_payload(response)
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE_PLACEHOLDER)
        self.observation_space = gym.spaces.Box(-1, 1, (ACTION_SPACE_SIZE_PLACEHOLDER, self.current_observation.shape[1]))

        reward = response['reward']
        if response['done']:
            reward = []
            for x in response['reward']:
                if x == 0:
                    reward.append(-1)
                else:
                    reward.append(x)
        return self.observation, reward, response['done'], {}

    def get_observation_from_payload(self, response):
        o = response['observation']
        o = np.array(o)
        # each state encoding also feature a 'validity bit' for the legal_actions mask
        o = np.append(o, np.ones((o.shape[0], 1), dtype=o.dtype), axis=-1)
        return np.array(o)

    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Sushi Go!')

    def render(self, mode='human', close=False):
        # skipping extra network calls for now - uncomment to debug
        # self.remote_call('render')
        pass
