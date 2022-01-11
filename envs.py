import cv2
import gym
import numpy as np
from gym.spaces.box import Box
from nes_py.wrappers import JoypadSpace
from actions import ACTIONS, ACTIONS_MASK
import matplotlib.pyplot as plt
from collections import deque
import Contra
import time


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = JoypadSpace(env, ACTIONS)
    env = AtariRescale84x84(env)
    env = NormalizedEnv(env)
    env = ControllerStepEnv(env)
    return env


def _process_frame42(frame):
    # frame = frame[34:]
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale84x84, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [4, 84, 84])

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0

        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class ControllerStepEnv:
    def __init__(self, env):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.__before_score = 0
        self.__before_life = 2
        self.__before_status = 1

        self.__before_actions = deque(maxlen=128)

        self.__unchanged_score_steps = 0
        self.__unchanged_reward_steps = 0

        self.__killed_reward = -50
        self.__continue_score_zero = 128
        self.__continue_reward_zero = 128

        self.__right_actions = []
        for idx, a in enumerate(ACTIONS):
            if "right" in a:
                self.__right_actions.append(idx)

    def seed(self, seed):
        self.env.seed(seed)

    def step(self, action):
        states = []
        rewards = []

        for i in range(4):
            if i in (1, 3) and action in ACTIONS_MASK.keys():
                state, reward, done, info = self.env.step(ACTIONS_MASK[action])
            else:
                state, reward, done, info = self.env.step(action)

            states.append(state)
            rewards.append(reward)

        if self.__before_score == info["score"]:
            self.__unchanged_score_steps += 1
        else:
            self.__unchanged_score_steps = 0
        
        reward = info["score"] - self.__before_score
        self.__before_score = info["score"]

        # 没命
        if info["life"] != self.__before_life or info["status"] != self.__before_status:
            reward = self.__killed_reward
            done = True

        # score 连续为0
        if self.__unchanged_score_steps > self.__continue_score_zero:
            reward = self.__killed_reward
            done = True

        # 连续重复动作
        self.__before_actions.append(action)
        if self.__before_actions.count(action) == self.__before_actions.maxlen:
            reward = self.__killed_reward
            done = True

        # reward 连续为0
        if 0 == reward:
            self.__unchanged_reward_steps += 1
        else:
            self.__unchanged_reward_steps = 0
        if self.__unchanged_reward_steps > self.__continue_reward_zero:
            reward = self.__killed_reward
            done = True

        if action not in self.__right_actions:
            reward -= 0.05

        return np.concatenate(states, axis=0), reward, done, info

    def reset(self):
        self.__before_score = 0
        self.__before_life = 2
        self.__before_status = 1
        self.__before_actions.clear()
        self.__unchanged_score_steps = 0
        self.__unchanged_reward_steps = 0
        obs = self.env.reset()
        return np.concatenate([obs, obs, obs, obs], axis=0)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.__before_actions.clear()


def test():
    env = create_atari_env("Contra-v0")

    done = False
    env.reset()

    step = 0
    actions = []
    for _ in range(10000):
        if done:
            print("Over")
            break
        # time.sleep(1)

        action = env.action_space.sample()
        # action = 2

        actions.append(action)
        state, reward, done, info = env.step(action)
        step += 1
        print(reward, info, action, "==============>{0}<=============".format(step))
        env.render()

    env.close()


def test_pre():
    env = gym.make("Contra-v0")
    env = JoypadSpace(env, ACTIONS)
    obs = env.reset()

    plt.figure("Image")
    plt.imshow(obs)
    plt.show()

    plt.imshow(obs)
    plt.show()

    obs = cv2.resize(obs, (84, 84))
    plt.imshow(obs)
    plt.show()

    obs = obs.mean(2, keepdims=True)
    plt.imshow(obs)
    plt.show()

    obs = obs.astype(np.float32)
    obs *= (1.0 / 255.0)
    obs = np.moveaxis(obs, -1, 0)
    print(obs.shape)


if __name__ == "__main__":
    # env = create_atari_env("Contra-v0")
    # obs = env.reset()
    # env.render()
    # print(env.observation_space.shape)
    # print()

    # test_pre()
    test()
