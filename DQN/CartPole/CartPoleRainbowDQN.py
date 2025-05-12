import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import math
from gymnasium.wrappers import RecordVideo


class NoisyLinear(nn.Module):
    """Noisy Linear Layer для улучшения исследования."""

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Обучаемые параметры
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)

        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def _scale_noise(self, size):
        noise = torch.randn(size)
        return noise.sign().mul_(noise.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return nn.functional.linear(x, weight, bias)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)


class RainbowDQN(nn.Module):
    """Сеть Rainbow DQN с Dueling Architecture и Noisy Layers."""

    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Dueling Network: разделение на Value и Advantage
        self.value_stream = NoisyLinear(128, n_atoms)
        self.advantage_stream = NoisyLinear(128, action_dim * n_atoms)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.n_atoms)
        q_dist = value + advantage - advantage.mean(1, keepdim=True)
        return nn.functional.softmax(q_dist, dim=2)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """Буфер воспроизведения с приоритетами."""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities.flatten()):
            self.priorities[idx] = priority


def train_rainbow(env_name="CartPole-v1", episodes=10000, batch_size=128, gamma=0.99, lr=3e-4):
    # Создаем среду с записью видео (сохраняем каждые 50 эпизодов)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        "videos",
        episode_trigger=lambda x: x % 50 == 0,  # Записывать каждые 50 эпизодов
        name_prefix="rainbow-dqn"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = RainbowDQN(state_dim, action_dim)
    target_model = RainbowDQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(capacity=1000000)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            # Epsilon-greedy (можно заменить на NoisyNet)
            if random.random() < 0.1:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_dist = model(torch.FloatTensor(state))
                    q_values = (q_dist * torch.linspace(-10, 10, 51)).sum(2)
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if len(buffer.buffer) > batch_size:
                samples, indices, weights = buffer.sample(batch_size)
                states = torch.FloatTensor(np.array([s[0] for s in samples]))
                actions = torch.LongTensor(np.array([s[1] for s in samples]))
                rewards = torch.FloatTensor(np.array([s[2] for s in samples]))
                next_states = torch.FloatTensor(np.array([s[3] for s in samples]))
                dones = torch.FloatTensor(np.array([s[4] for s in samples]))
                weights = torch.FloatTensor(weights)

                with torch.no_grad():
                    next_q_dist = target_model(next_states)
                    next_q_values = (next_q_dist * torch.linspace(-10, 10, 51)).sum(2)
                    best_actions = next_q_values.argmax(1)
                    target_dist = next_q_dist[range(batch_size), best_actions]

                    target_dist = rewards.view(-1, 1) + gamma * (1 - dones.view(-1, 1)) * target_dist

                current_dist = model(states)[range(batch_size), actions]
                loss = torch.sum(target_dist * (torch.log(target_dist) - torch.log(current_dist + 1e-8)), 1)
                loss = (weights * loss).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Обновление приоритетов
                new_priorities = np.abs(loss.detach().numpy().flatten()) + 1e-5
                buffer.update_priorities(indices, new_priorities)

            if done:
                break

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            model.reset_noise()
            target_model.reset_noise()
            print(f"Episode: {episode}, Reward: {total_reward}")

    env.close()  # Важно закрыть среду для сохранения видео


if __name__ == "__main__":
    train_rainbow()