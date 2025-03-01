import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
from gym import spaces

# **1️⃣ Define the Environment for Reinforcement Learning (RL)**
class NetworkResourceEnv(gym.Env):
    def __init__(self, num_packets=1000):
        super(NetworkResourceEnv, self).__init__()

        # Simulated network parameters
        self.num_packets = num_packets
        self.current_step = 0

        # Action Space: Allocate frequency slots (discrete values from 1 to 15)
        self.action_space = spaces.Discrete(15)

        # Observation Space: [Simulation Time, Source Node, Destination Node, Needed Slots]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)

        # Generate synthetic network traffic data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'simtime': np.sort(np.random.uniform(0, 100, num_packets)),
            'source': np.random.randint(1, 30, num_packets),
            'dest': np.random.randint(1, 30, num_packets),
            'needed_slots': np.random.randint(1, 15, num_packets),
            'holding_time': np.random.uniform(1, 20, num_packets)
        })

    def reset(self):
        """Resets the environment at the start of an episode."""
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        """Gets the next observation state."""
        obs = self.data.iloc[self.current_step][['simtime', 'source', 'dest', 'needed_slots']].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Executes an action (allocating frequency slots) and returns the new state and reward."""
        actual_needed = self.data.iloc[self.current_step]['needed_slots']
        holding_time = self.data.iloc[self.current_step]['holding_time']

        # Reward Function: Minimize excess slot allocation while avoiding under-allocation
        if action >= actual_needed:
            reward = 1 - (action - actual_needed) * 0.1  # Slight penalty for excess allocation
        else:
            reward = -1  # Heavy penalty for under-allocation leading to blocking

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.num_packets

        return self._next_observation(), reward, done, {}

# **2️⃣ Train a Reinforcement Learning Model (DQN) for Smart Resource Allocation**
env = DummyVecEnv([lambda: NetworkResourceEnv(num_packets=1000)])
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# **3️⃣ Evaluate Performance of the Trained Model**
obs = env.reset()
total_reward = 0

for _ in range(1000):  # Simulate 1000 packet allocations
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        break

# **4️⃣ Visualizing Results**
plt.figure(figsize=(8, 5))
plt.bar(["Before AI Optimization", "After RL Optimization"], [df_network['holding_time'].mean(), total_reward / 1000], color=['red', 'green'])
plt.xlabel("Scenario")
plt.ylabel("Holding Time Reduction Score")
plt.title("Comparison of Network Performance Before and After RL Optimization")
plt.show()

# **5️⃣ Returning Results**
total_reward / 1000  # Average RL optimization score
