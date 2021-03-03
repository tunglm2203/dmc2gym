import dmc2gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gym
from collections import deque


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

def imshow(obs):
    if obs.shape[2] == 9:
        plt.subplot(131)
        plt.imshow(obs[:, :, :3])
        plt.subplot(132)
        plt.imshow(obs[:, :, 3:6])
        plt.subplot(133)
        plt.imshow(obs[:, :, 6:])

        # plt.subplot(231)
        # plt.imshow(obs[:, :, :3])
        # plt.subplot(232)
        # plt.imshow(obs[:, :, 3:6])
        # plt.subplot(233)
        # plt.imshow(obs[:, :, 6:])

        # plt.subplot(234)
        # plt.imshow(np.abs(obs[:, :, 3:6] - obs[:, :, :3]))
        # plt.subplot(235)
        # plt.imshow(np.abs(obs[:, :, 6:] - obs[:, :, 3:6]))
    else:
        plt.imshow(obs)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=False)


def main_dmc2gym():
    action_repeat = dict(
        cartpole=8,
        walker=2,
        cheetah=4,
        finger=2,
        reacher=4,
        ball_in_cup=4,
        hopper=4,
        fish=4,
        pendulum=4,
        quadruped=4
    )
    camera_id = dict(
        cartpole=0,
        walker=0,
        cheetah=0,
        finger=0,
        reacher=0,
        ball_in_cup=0,
        hopper=0,
        fish=0,
        pendulum=0,
        quadruped=2
    )

    img_size = 84
    n_steps = 50

    # env_name = ['quadruped', 'walk']
    # env_name = ['quadruped', 'run']
    # env_name = ['dog', 'run']
    # env_name = ['cheetah', 'run']
    # env_name = ['walker', 'stand']
    env_name = ['walker', 'walk']
    # env_name = ['walker', 'run']
    # env_name = ['finger', 'spin']            # Sparse
    # env_name = ['finger', 'turn_easy']     # Sparse
    # env_name = ['finger', 'turn_hard']     # Sparse
    # env_name = ['reacher', 'easy']           # Sparse
    # env_name = ['reacher', 'hard']         # Sparse
    # env_name = ['hopper', 'stand']
    # env_name = ['hopper', 'hop']
    # env_name = ['cartpole', 'swingup']
    # env_name = ['cartpole', 'balance']
    # env_name = ['cartpole', 'balance_sparse']
    # env_name = ['cartpole', 'swingup_sparse']
    # env_name = ['ball_in_cup', 'catch']    # Sparse
    # env_name = ['fish', 'upright']
    # env_name = ['fish', 'swim']
    # env_name = ['pendulum', 'swingup']
    from_image = True

    if from_image:
        env = dmc2gym.make(
            domain_name=env_name[0],
            task_name=env_name[1],
            difficulty='easy',
            background_dataset_path='../dmc2gym/dmc2gym/videos/DAVIS/JPEGImages/480p',
            dynamic=False,
            default_background=False,
            default_camera=True,
            default_color=True,
            seed=1,
            visualize_reward=False,
            from_pixels=from_image,
            height=img_size,
            width=img_size,
            frame_skip=action_repeat[env_name[0]]
        )
        env = FrameStack(env, k=3)
    else:
        env = dmc2gym.make(
            domain_name=env_name[0],
            task_name=env_name[1],
            seed=1,
            visualize_reward=False,
            from_pixels=False,
            frame_skip=1,
            camera_id=camera_id[env_name[0]],
        )
    print('[INFO] Observation space: ', env.observation_space)
    print('[INFO] Action space: ', env.action_space)
    o = env.reset()

    reset_step = 10
    for i in tqdm(range(n_steps)):
        a = env.action_space.sample()
        o, r, done, _ = env.step(a)
        print('Reward: ', r)
        if from_image:
            imshow(o.transpose(1, 2, 0))
        else:
            im = env.render(mode='rgb_array')
            imshow(im)


        if done or (i != 0 and i % reset_step == 0):
            env.reset()

if __name__ == '__main__':
    main_dmc2gym()
