import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta, Normal
import pandas as pd
import argparse

def get_data():
    data = pd.read_csv('imp.csv', sep='\t')
    print("Train range: {} - {}".format(data[data['mask'] == 0].week_id.min(), data[data['mask'] == 0].week_id.max()))
    print("Test range: {} - {}".format(data[data['mask'] == 1].week_id.min(), data[data['mask'] == 1].week_id.max()))

    print("Train week range: {} - {}".format(data[data['mask'] == 0].week_id.min(), data[data['mask'] == 0].week_id.max()))
    print("Test week range: {} - {}".format(data[data['mask'] == 1].week_id.min(), data[data['mask'] == 1].week_id.max()))

    user_ids = data['iPinYou ID'].unique()
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    user_num = len(user_ids)

    train_data = data[data['mask'] == 0]
    test_data = data[data['mask'] == 1]

    bid_hist_train = np.zeros((15, user_num, 12)) 
    win_hist_train = np.zeros((15, user_num, 12))
    bid_hist_test = np.zeros((15, user_num, 12)) 
    win_hist_test = np.zeros((15, user_num, 12))

    for _, row in train_data.iterrows():
        user_idx = user_id_to_index[row['iPinYou ID']]
        T_idx = int((row['week_id'] - 1) * 7 + row['weekday'])
        hour_idx = int(row['hour'] // 2)
        if 0 <= T_idx < 15 and 0 <= user_idx < user_num and 0 <= hour_idx < 12:
            bid_hist_train[T_idx, user_idx, hour_idx] += row['random_bidding_price']
            win_hist_train[T_idx, user_idx, hour_idx] += row['Paying Price']

    for _, row in test_data.iterrows():
        user_idx = user_id_to_index[row['iPinYou ID']]
        T_idx = int((row['week_id'] - 1) * 7 + row['weekday'])
        hour_idx = int(row['hour'] // 2)
        if 15 <= T_idx < 30 and 0 <= user_idx < user_num and 0 <= hour_idx < 12:
            bid_hist_test[T_idx, user_idx, hour_idx] += row['random_bidding_price']
            win_hist_test[T_idx, user_idx, hour_idx] += row['Paying Price']

    return data, bid_hist_train, win_hist_train, bid_hist_test, win_hist_test

class BidGym:
    def __init__(self, init_w, budget, bid_hist, win_hist, convert_probs, T=23, eps_low=10, eps_high=21):
        self.budget = budget
        self.T = T
        self.rstar = convert_probs.sum() * self.T * 12
        self.bid_hist = bid_hist
        self.win_hist = win_hist
        self.convert_probs = convert_probs[:, np.newaxis].repeat(len(init_w), axis=1)
        self.eps_low, self.eps_high = eps_low, eps_high
        self.count = 0
        self.w = init_w
        self.r_cumulate = 0.
        self.budget_spend = 0.
        self.current_state = np.array([self.budget, 0., 0., 0.])
        self.current_state = np.concatenate([[self.current_state], np.zeros((T - 1, len(self.current_state)))])
        self.w_hist = np.zeros((self.T, len(self.w)))
        self.w_hist[0] = self.w
        self.budget_spend_hist = np.zeros(self.T)
        self.r_cumulate_hist = np.zeros(self.T)

    def step(self, a):
        a = a / 10
        step = self.count = self.count + 1
        self.w = (1 + a) * self.w
        bid_market = self.bid_hist[step] - self.win_hist[step] * \
            np.random.randint(low=self.eps_low, high=self.eps_high, size=self.bid_hist[step].shape) / 100

        reward, spend = self.cal_reward(self.w, step, bid_market, self.win_hist[step])
        self.r_cumulate += reward 
        self.budget_spend += spend

        self.current_state[step] = (
            self.budget - self.budget_spend,
            step / self.T,
            self.r_cumulate / self.rstar,
            self.r_cumulate / self.budget_spend if self.budget_spend > 0 else 0
        )
        self.w_hist[step] = self.w
        self.budget_spend_hist[step] = self.budget_spend
        self.r_cumulate_hist[step] = self.r_cumulate

        done = step == (self.T - 1) or self.budget <= self.budget_spend
        return self.current_state[step], reward, done


    def reset(self, step=0):
        self.count = step
        self.w = self.w_hist[step]
        self.r_cumulate = self.r_cumulate_hist[step]
        self.budget_spend = self.budget_spend_hist[step]
        return self.current_state[step]

    def cal_reward(self, w, step, bid_market, win_hist):
        bid = w * self.convert_probs
        win = (bid >= bid_market) * np.abs(win_hist)
        reward = (self.convert_probs * win).sum()
        spend = (bid_market * win).sum()
        return reward, spend
    
class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        return x

    def reset(self):
        self.R = np.zeros(self.shape)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist
    
    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)
        return mean

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, done
    
    class PPO_continuous:
        def __init__(self, args):
            self.policy_dist = args.policy_dist
            self.max_action = args.max_action
            self.batch_size = args.batch_size
            self.mini_batch_size = args.mini_batch_size
            self.max_train_steps = args.max_train_steps
            self.lr_a = args.lr_a  # Actor learning rate
            self.lr_c = args.lr_c  # Critic learning rate
            self.lamda = args.lamda  # GAE parameter
            self.gamma = args.gamma  # discount factor
            self.epsilon = args.epsilon  # PPO clip parameter
            self.K_epochs = args.K_epochs  # update policy for K epochs
            self.entropy_coef = args.entropy_coef  # entropy coefficient
            self.set_adam_eps = args.set_adam_eps  # set Adam optimizer epsilon

            # policy networks initialization
            if self.policy_dist == "Beta":
                self.actor = Actor_Beta(args)
            else:
                self.actor = Actor_Gaussian(args)

            # value network initialization
            self.critic = Critic(args)

            # optimizer initialization
            if self.set_adam_eps:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            else:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        def evaluate(self, s):
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            else:
                a = self.actor(s).detach().numpy().flatten()
            return a

        def choose_action(self, s):
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            with torch.no_grad():
                if self.policy_dist == "Beta":
                    dist = self.actor.get_dist(s)
                    a = dist.sample()
                    a_logprob = dist.log_prob(a)
                else:
                    dist = self.actor.get_dist(s)
                    a = dist.sample()
                    a = torch.clamp(a, -self.max_action, self.max_action)
                    a_logprob = dist.log_prob(a)
            return a.numpy().flatten(), a_logprob.numpy().flatten()

        def update(self, replay_buffer, total_steps, bs):
            s, a, a_logprob, V, done = replay_buffer.numpy_to_tensor()
            adv = []
            gae = 0

            # Advantage 
            with torch.no_grad():
                vs = self.critic(s)
                deltas = V - vs
                for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                    adv.insert(0, gae)
                adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
                v_target = adv + vs
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

            # policy and value networks update
            for _ in range(self.K_epochs):
                for index in BatchSampler(SubsetRandomSampler(range(bs)), self.mini_batch_size, False):
                    dist_now = self.actor.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                    a_logprob_now = dist_now.log_prob(a[index])
                    ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                    surr1 = ratios * adv[index]
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index], v_s)

                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

        def lr_decay(self, total_steps):
            lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
            lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
            for p in self.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in self.optimizer_critic.param_groups:
                p['lr'] = lr_c_now

def evaluate_policy(args, env, agent, state_norm):
    times = 100
    evaluate_reward = 0
    total_spend = 0
    days = 0

    for _ in range(times):
        s = env.reset()
        s = state_norm(s, update=False)
        episode_reward = 0
        episode_spend = 0
        done = False

        while not done:
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, done = env.step(action)
            episode_reward += r
            episode_spend += env.budget_spend - episode_spend
            s_ = state_norm(s_, update=False)
            s = s_

        evaluate_reward += episode_reward
        total_spend += episode_spend
        days += env.count

    avg_reward = evaluate_reward / times
    avg_spend = total_spend / times
    ROI = evaluate_reward / total_spend if total_spend > 0 else 0

    print(f"Evaluation Results: Average Reward: {avg_reward:.2f}, ROI: {ROI:.2f}")
    return avg_reward, ROI, int(days / times)

def test_policy(env, agent, state_norm, episodes=10):
    total_reward = 0
    total_spend = 0

    for _ in range(episodes):
        s = env.reset()
        s = state_norm(s, update=False)
        episode_reward = 0
        episode_spend = 0
        done = False

        while not done:
            a = agent.evaluate(s)
            if agent.policy_dist == "Beta":
                action = 2 * (a - 0.5) * agent.max_action
            else:
                action = a

            s_, r, done = env.step(action)
            episode_reward += r
            episode_spend += env.budget_spend - episode_spend
            s_ = state_norm(s_, update=False)
            s = s_

        total_reward += episode_reward
        total_spend += episode_spend

    avg_reward = total_reward / episodes
    avg_spend = total_spend / episodes
    ROI = total_reward / total_spend if total_spend > 0 else 0

    print(f"Test Results: Avg Reward: {avg_reward:.2f}, ROI: {ROI:.2f}")
    return avg_reward, ROI

def main():

    data, bid_hist_train, win_hist_train, bid_hist_test, win_hist_test = get_data()

    parser = argparse.ArgumentParser(description="RTB Optimization using PPO")
    
    parser.add_argument('--state_dim', type=int, default=4, help="State dimension")
    parser.add_argument('--action_dim', type=int, default=12, help="Action dimension")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch size for training")
    parser.add_argument('--mini_batch_size', type=int, default=30, help="Mini-batch size for PPO updates")
    parser.add_argument('--max_train_steps', type=int, default=int(3e6), help="Maximum training steps")
    parser.add_argument('--evaluate_freq', type=int, default=100, help="Frequency of evaluation during training")
    parser.add_argument('--policy_dist', type=str, default='Beta', help="Policy distribution (Beta or Gaussian)")
    parser.add_argument('--max_action', type=float, default=2.0, help="Maximum action value")
    parser.add_argument('--hidden_width', type=int, default=1024, help="Hidden layer width for networks")
    parser.add_argument('--lr_a', type=float, default=3e-4, help="Learning rate for actor network")
    parser.add_argument('--lr_c', type=float, default=3e-4, help="Learning rate for critic network")
    parser.add_argument('--lamda', type=float, default=1.0, help="GAE lambda parameter")
    parser.add_argument('--gamma', type=float, default=1.0, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument('--K_epochs', type=int, default=10, help="Number of epochs for PPO updates")
    parser.add_argument('--entropy_coef', type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument('--set_adam_eps', type=bool, default=True, help="Whether to set Adam optimizer epsilon")
    parser.add_argument('--budget', type=float, default=5e6, help="Total budget")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    init_w = np.array([10.] * 12)
    convert_probs = data.groupby('iPinYou ID')['convert_prob'].first().values
    env = BidGym(init_w, args.budget, bid_hist_train, win_hist_train, convert_probs, T=15, eps_low=10, eps_high=21)
    env_evaluate = BidGym(init_w, args.budget, bid_hist_train, win_hist_train, convert_probs, T=15, eps_low=10, eps_high=21)
    env_test = BidGym(init_w, args.budget, bid_hist_test, win_hist_test, convert_probs, T=15, eps_low=10, eps_high=21)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class ArgsWrapper:
        def __init__(self, args_dict):
            for key, value in args_dict.items():
                setattr(self, key, value)

    args_wrapper = ArgsWrapper(vars(args))

    evaluate_num = 0
    total_steps = 0

    replay_buffer = ReplayBuffer(args_wrapper)
    agent = replay_buffer.PPO_continuous(args_wrapper)
    state_norm = Normalization(shape=args_wrapper.state_dim)

    while total_steps < args.max_train_steps:
        replay_buffer.count = 0
        s = env.reset()
        s = state_norm(s)
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a
            s_, R, done = env.step(action)
            done_ = False
            while not done and not done_:
                _, r, done_ = env.step(0.)
                R += r
            env.reset(episode_steps)
            s_ = state_norm(s_)
            replay_buffer.store(s, a, a_logprob, R, done)
            s = s_
            if (replay_buffer.count == args.batch_size) or done:
                agent.update(replay_buffer, total_steps,replay_buffer.count)
                replay_buffer.count = 0

        total_steps += 1
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward, days = evaluate_policy(args, env_evaluate, agent, state_norm)
    
    test_policy(env_test, agent, state_norm, episodes=10)

if __name__ == '__main__':
    main()