import time
import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic


def test(rank, args, shared_model, counter):
    """

    :param rank: worker数
    :param args:
    :param shared_model:
    :param counter: steps_counter
    :return:
    """
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0

    # 最高reward
    max_reward = 0

    # 每隔1min跑一次测试
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()[0, 0]

        state, reward, done, _ = env.step(action)

        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print("run_time {}, num_steps {}, FPS {:.0f}, current_episode_reward {}, current_episode_length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time), round(reward_sum, 2), episode_length))

            torch.save(model.state_dict(), "model.pkl")
            if reward_sum > max_reward:
                torch.save(model.state_dict(), "model-max-reward.pkl")
                print("===============> max reward model save, reward: {0} <================".format(round(reward_sum)))
                max_reward = reward_sum

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)


if __name__ == "__main__":
    torch.manual_seed(681)
    env = create_atari_env("Contra-v0")
    env.seed(681)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    # model.load_state_dict(torch.load('model.pkl'))
    # model.load_state_dict(torch.load('model-max-reward.pkl'))
    # model.load_state_dict(torch.load('data/model/model_01_12.pkl'))
    model.load_state_dict(torch.load('data/model/model_01_18.pkl'))
    # model.load_state_dict(torch.load('data/model/model.pkl'))

    model.eval()
    obs = env.reset()
    state = torch.from_numpy(obs)
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)

    reward_sum = 0
    steps = 0

    before_y_pos = 0

    while True:
        env.render()
        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        # action = prob.multinomial(num_samples=1).numpy()[0, 0]
        action = prob.max(1, keepdim=True)[1].numpy()[0, 0]
        state, reward, done, info = env.step(action)

        before_y_pos = info["y_pos"]
        time.sleep(0.02)

        if steps % 10 == 0:
            print("\n\n")

        print(reward, info, action, round(reward_sum, 2), steps, sep="\t")
        # print(prob)
        state = torch.from_numpy(state)
        reward_sum += reward
        steps += 1
        if done:
            print("total_reward", reward_sum)
            break