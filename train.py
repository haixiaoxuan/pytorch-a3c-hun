import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.utils.tensorboard import SummaryWriter
import os
import datetime


def ensure_shared_grads(model, shared_model):
    # 共享梯度
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    """
    :param rank: worker编号
    :param args:
    :param shared_model:
    :param counter: global_steps
    :param lock:
    :param optimizer:
    :return:
    """
    writer = SummaryWriter(log_dir=os.path.join("data/logs/{}".format(str(datetime.datetime.now())[:16])),
                           max_queue=100, flush_secs=10, filename_suffix=str(rank))
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    total_rewards = 0
    total_episode = 0
    total_update_times = 0
    episode_length = 0

    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []             # critic output state-value
        log_probs = []          # action log prob
        rewards = []            # 单步奖励
        entropies = []          # cross-entropy

        for i in range(args.num_steps):
            episode_length += 1

            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            # value, logit, (hx, cx) = model((state, (hx, cx)))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            action = action.numpy()[0][0]

            state, reward, done, info = env.step(action)
            total_rewards += reward

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                writer.add_scalar("total_reward/{0}".format(rank), round(total_rewards, 2), total_episode)
                writer.add_scalar("total_steps/{0}".format(rank), episode_length, total_episode)
                writer.add_scalar("total_score/{0}".format(rank), info["score"], total_episode)

                episode_length = 0
                total_rewards = 0
                total_episode += 1
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # Train
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t
            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        total_update_times += 1

        writer.add_scalar("policy_loss/{0}".format(rank), policy_loss, total_update_times)
        writer.add_scalar("value_loss/{0}".format(rank), value_loss, total_update_times)
        writer.add_scalar("entropy_loss/{0}".format(rank), -sum(entropies), total_update_times)
        # writer.flush()
