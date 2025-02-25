import torch
import torch.nn as nn
import numpy as np
from env.ball_plate_env import BallBalanceEnv
import time
import matplotlib.pyplot as plt


# MLP network is the same as the before
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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.net = MLP(state_dim, action_dim * 2)  # output mean and log_std

    def forward(self, state):
        x = self.net.net(state)
        mean_log_std = self.net.output_layer(x)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # limit the range of log_std
        std = torch.exp(log_std)
        return mean, std

    def select_action(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.action_bound
        return action

def test_agent(render=True, num_episodes=100):
    # create the environment
    env = BallBalanceEnv(render=render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    # load the weight of the trained model
    model_path = 'models/best_model.pth'
    checkpoint = torch.load(model_path, map_location=device)

    # initialize the actor net
    actor = Actor(state_dim, action_dim, action_bound).to(
        device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # change to eval mode

    total_reward = 0
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(
                0).to(device)
            with torch.no_grad():
                action = actor.select_action(state_tensor)
            action = action.cpu().numpy()[0]

            next_state, reward, done, truncated, _ = env.step(
                action)
            is_done = done or truncated

            time.sleep(0.02)  # wait 0.02s every time step

            state = next_state
            episode_reward += reward

            if is_done:
                break
        episode_rewards.append(episode_reward)

        print(f"Test Episode {episode + 1}: Reward = "
              f"{episode_reward:.2f}")
        total_reward += episode_reward

    episodes = range(1, num_episodes + 1)
    plt.figure()
    plt.plot(episodes, episode_rewards, marker='o')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: "
          f"{avg_reward:.2f}")



    env.close()

if __name__ == '__main__':
    test_agent()