from gym import core, spaces
from dmc2gym.distracting_control import suite
from dm_env import specs
import numpy as np
from dm_control import manipulation
from collections import OrderedDict

def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
            self,
            domain_name,
            task_name,
            task_kwargs=None,
            visualize_reward={},
            from_pixels=False,
            height=84,
            width=84,
            camera_id=0,
            frame_skip=1,
            environment_kwargs=None,
            channels_first=True,
            difficulty=None,
            dynamic=False,
            background_dataset_path=None,
            background_dataset_videos="train",
            background_kwargs=None,
            camera_kwargs=None,
            color_kwargs=None,
            default_background=False,
            default_camera=False,
            default_color=False,
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            pixels_only=False,
            environment_kwargs=environment_kwargs,
            difficulty=difficulty,
            dynamic=dynamic,
            background_dataset_path=background_dataset_path,
            background_dataset_videos=background_dataset_videos,
            background_kwargs=background_kwargs,
            camera_kwargs=camera_kwargs,
            color_kwargs=color_kwargs,
            default_background=default_background,
            default_camera=default_camera,
            default_color=default_color,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )
            
        self._state_space = _spec_to_box(
                self._env.observation_spec().values()
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        skipped_obses, skipped_acts = [], []
        intermediate_rewards = []
        for i in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()

            # Don't save last frame
            if i < self._frame_skip - 1:
                _obs = self._get_obs(time_step)
                skipped_obses.append(_obs)
                skipped_acts.append(action)
                intermediate_rewards.append(time_step.reward or 0)

            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        extra['skipped_obses'] = skipped_obses
        extra['skipped_acts'] = skipped_acts
        extra['intermediate_rewards'] = intermediate_rewards
        extra_info = extra.copy()
        return obs, reward, done, extra_info

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )


class DMCWrapperManipulation(core.Env):
    def __init__(
            self,
            domain_name,
            task_name,
            task_kwargs=None,
            visualize_reward={},
            from_pixels=False,
            height=84,
            width=84,
            camera_id=0,
            frame_skip=1,
            environment_kwargs=None,
            channels_first=True,
            difficulty=None,
            dynamic=False,
            background_dataset_path=None,
            background_dataset_videos="train",
            background_kwargs=None,
            camera_kwargs=None,
            color_kwargs=None,
            default_background=False,
            default_camera=False,
            default_color=False,
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        assert domain_name == 'manipulation'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        if from_pixels:
            task_name += '_vision'
        else:
            task_name += '_features'

        # create task
        self._env = manipulation.load(
            environment_name=task_name,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        excluded_keys = ['front_close']

        spec = OrderedDict()
        for k, v in self._env.observation_spec().items():
            if k in excluded_keys:
                continue
            spec[k] = v

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                spec.values()
            )

        self._state_space = _spec_to_box(
            spec.values()
        )

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        skipped_obses, skipped_acts = [], []
        intermediate_rewards = []
        for i in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()

            # Don't save last frame
            if i < self._frame_skip - 1:
                _obs = self._get_obs(time_step)
                skipped_obses.append(_obs)
                skipped_acts.append(action)
                intermediate_rewards.append(time_step.reward or 0)

            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        extra['skipped_obses'] = skipped_obses
        extra['skipped_acts'] = skipped_acts
        extra['intermediate_rewards'] = intermediate_rewards
        extra_info = extra.copy()
        return obs, reward, done, extra_info

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )