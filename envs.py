import cv2
import gym
import numpy as np
from gym.spaces.box import Box
from nes_py.wrappers import JoypadSpace
from actions import ACTIONS
import matplotlib.pyplot as plt
from collections import deque


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = JoypadSpace(env, ACTIONS)
    env = AtariRescale84x84(env)
    env = NormalizedEnv(env)
    env = ControllerStepEnv(env, ACTIONS)
    return env


def _process_frame42(frame):
    frame = frame[34:]
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale84x84, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

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
    def __init__(self, env, acts):
        self.env = env

        # 包含动作A或者B的action
        self.__spe_act = []
        for i, a in enumerate(acts):
            if ("A" in a) or ("B" in a):
                self.__spe_act.append(i)

        # 包含right的action
        self.__right_act = []
        for i, a in enumerate(acts):
            if "right" in a:
                self.__right_act.append(i)

        # 同时包含 down 和 right 的action
        self.__down_right_action = []
        for i, a in enumerate(acts):
            if ("down" in a) and ("right" in a):
                self.__down_right_action.append(i)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.__before_action = -1
        self.__before_score = 0
        self.__before_life = 2
        self.__before_status = 1

        self.__before_2000_action = deque(maxlen=300)

        self.__unchanged_score_steps = 0
        self.__unchanged_reward_steps = 0

        self.__killed_reward = -50
        self.__continue_score_zero = 512
        self.__continue_reward_zero = 512

    def seed(self, seed):
        self.env.seed(seed)

    def step(self, action):
        if (self.__before_action == action) and (action in self.__spe_act):
            self.env.step(0)
            self.__before_action = -1
        else:
            self.__before_action = action
            
        # cal reward
        state, reward, done, info = self.env.step(action)

        if self.__before_score == info["score"]:
            self.__unchanged_score_steps += 1
        else:
            self.__unchanged_score_steps = 0
        
        reward = info["score"] - self.__before_score
        self.__before_score = info["score"]

        # if action in self.__right_act:
        #     reward += 0.05

        # if info["y_pos"] == 43 and action in self.__down_right_action:
        #     reward -= 0.1

        # 没命
        if info["life"] != self.__before_life or info["status"] != self.__before_status:
            reward = self.__killed_reward
            done = True

        # score 连续为0
        if self.__unchanged_score_steps > self.__continue_score_zero:
            reward = self.__killed_reward
            done = True

        # 连续重复动作
        self.__before_2000_action.append(action)
        if self.__before_2000_action.count(action) == self.__before_2000_action.maxlen:
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

        return state, reward, done, info

    def reset(self):
        self.__before_action = -1
        self.__before_score = 0
        self.__before_life = 2
        self.__before_status = 1
        self.__before_2000_action.clear()
        self.__unchanged_score_steps = 0
        self.__unchanged_reward_steps = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.__before_2000_action.clear()


def test():
    env = create_atari_env("Contra-v0")

    done = False
    env.reset()

    actions = []
    for _ in range(100000):
        if done:
            print("Over")
            break
        # time.sleep(0.02)

        action = env.action_space.sample()
        # action = 6

        actions.append(action)
        state, reward, done, info = env.step(action)
        print(reward, info, action)
        env.render()

    env.close()
    print(set(actions))


def test_pre():
    env = gym.make("Contra-v0")
    env = JoypadSpace(env, ACTIONS)
    obs = env.reset()

    plt.figure("Image")
    plt.imshow(obs)
    plt.show()

    obs = obs[35:]  # crop
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
