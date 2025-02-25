import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
from env.ball_plate_env import BallBalanceEnv
import os


# set random seed
def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


# define the basic network frame
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.net(x)
        return self.output_layer(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.zeros_(m.bias)


# define actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.net = MLP(state_dim, action_dim * 2)  # out out mean and log_std

    def forward(self, state):
        x = self.net.net(state)
        mean_log_std = self.net.output_layer(x)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # limit the range of log_std
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6) # output the log possibility
        log_prob = log_prob.sum(dim=1, keepdim=True)
        scaled_action = action * self.action_bound
        return scaled_action, log_prob

    def select_action(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.action_bound
        return action

# define the Critic network，including double q-network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1_net = MLP(state_dim + action_dim, 1)
        self.q2_net = MLP(state_dim + action_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2


# define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity=1e6):
        self.buffer = collections.deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# define SAC algorithm
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.action_bound = action_bound

        # initialize the actor and critic network
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # define optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # define Automatically adjustable temperature parameters alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # replay buffer
        self.replay_buffer = ReplayBuffer()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                action = self.actor.select_action(state)
            return action.cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None

        # sample from the buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # update the critic network
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = (-(self.log_alpha * (log_prob + self.target_entropy).detach())).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # soft update target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item()
        }

# Training the SAC algorithm
def train_sac(num_episodes=400, batch_size=256):
    # create the environment
    env = BallBalanceEnv(render=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the random seed
    set_seed(42, env)

    # initialize the SAC agent
    agent = SACAgent(state_dim, action_dim, action_bound, device)

    # make dir to save the best model
    best_reward = -np.inf
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)
    # TensorBoard record
    writer = SummaryWriter('runs/sac_ball_balance_sb3')

    # Pre-populated experience playback buffer
    prefill_steps = 10000
    state, _ = env.reset()
    for _ in range(prefill_steps):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        is_done = done or truncated
        agent.replay_buffer.add(state, action, reward, next_state, is_done)
        state = next_state if not is_done else env.reset()[0]

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_ball_speeds = []
        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            is_done = done or truncated
            # get the velocity of the ball
            x_v = state[3]
            y_v = state[4]
            z_v = state[5]
            speed = np.sqrt(x_v**2 + y_v**2 + z_v**2)
            episode_ball_speeds.append(speed)
            agent.replay_buffer.add(state, action, reward, next_state, is_done)

            update_info = agent.update(batch_size)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if is_done:
                break
        avg_ball_speed = np.mean(episode_ball_speeds)
        writer.add_scalar('Ball Speed/Average', avg_ball_speed, episode + 1)
        # write down the metric
        writer.add_scalar('Reward', episode_reward, episode + 1)
        if update_info is not None:
            writer.add_scalar('Critic Loss', update_info['critic_loss'], episode + 1)
            writer.add_scalar('Actor Loss', update_info['actor_loss'], episode + 1)
            writer.add_scalar('Alpha Loss', update_info['alpha_loss'], episode + 1)
            writer.add_scalar('Alpha Value', update_info['alpha_value'], episode + 1)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'alpha': agent.alpha.item(),
                'log_alpha': agent.log_alpha.item()
            }, os.path.join(save_path, 'best_model.pth'))
            print(f"Training completed. Model saved at "
                  f"episode {episode + 1}.")
        # print the loss information
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, "
                  f"Actor Loss: {update_info['actor_loss']:.4f}, "
                  f"Critic Loss: {update_info['critic_loss']:.4f}, "
                  f"Alpha: {update_info['alpha_value']:.4f}， "
                  f"Alpha Loss: {update_info['alpha_loss']: 4f} ")

    writer.close()
    env.close()

if __name__ == '__main__':
    train_sac()
