#!/usr/bin/env python
# coding: utf-8

# In[1]:


# **Enhancing the RL Model with an Improved Reward Function**

class OptimizedNetworkResourceEnv(gym.Env):
    def __init__(self, num_packets=1000):
        super(OptimizedNetworkResourceEnv, self).__init__()

        self.num_packets = num_packets
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(15)  # Frequency slot allocation (1-15)
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)

        # Generate synthetic traffic data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'simtime': np.sort(np.random.uniform(0, 100, num_packets)),
            'source': np.random.randint(1, 30, num_packets),
            'dest': np.random.randint(1, 30, num_packets),
            'needed_slots': np.random.randint(1, 15, num_packets),
            'holding_time': np.random.uniform(1, 20, num_packets)
        })

    def reset(self):
        """Resets the environment."""
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        """Gets the next observation state."""
        obs = self.data.iloc[self.current_step][['simtime', 'source', 'dest', 'needed_slots']].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Executes an action (allocating frequency slots) and returns new state and reward."""
        actual_needed = self.data.iloc[self.current_step]['needed_slots']
        holding_time = self.data.iloc[self.current_step]['holding_time']

        # **Improved Reward Function:**
        # - Reward **efficient slot allocation**
        # - Penalize **under-allocation (blocking)**
        # - Penalize **over-allocation (resource waste)**

        if action == actual_needed:
            reward = 2  # Optimal allocation gets a high reward
        elif action > actual_needed:
            reward = 1 - (action - actual_needed) * 0.2  # Small penalty for over-allocation
        else:
            reward = -2  # Heavy penalty for blocking due to under-allocation

        self.current_step += 1
        done = self.current_step >= self.num_packets

        return self._next_observation(), reward, done, {}

# **Train the Enhanced RL Model**
env_optimized = DummyVecEnv([lambda: OptimizedNetworkResourceEnv(num_packets=1000)])
optimized_model = DQN("MlpPolicy", env_optimized, verbose=1)
optimized_model.learn(total_timesteps=10000)  # Training for 10,000 steps

# **Evaluate Performance of the Optimized RL Model**
obs = env_optimized.reset()
total_reward_optimized = 0

for _ in range(1000):  # Simulating 1000 packet allocations
    action, _states = optimized_model.predict(obs, deterministic=True)
    obs, reward, done, _ = env_optimized.step(action)
    total_reward_optimized += reward

    if done:
        break

# **Comparison Visualization**
plt.figure(figsize=(8, 5))
plt.bar(["Before AI Optimization", "After Basic RL", "After Improved RL"], 
        [df_network['holding_time'].mean(), total_reward / 1000, total_reward_optimized / 1000], 
        color=['red', 'blue', 'green'])
plt.xlabel("Scenario")
plt.ylabel("Holding Time Reduction Score")
plt.title("Comparison of Network Performance Before and After AI Optimization")
plt.show()

# **Returning Performance Metrics for Further Analysis**
total_reward / 1000, total_reward_optimized / 1000  # Comparing standard RL vs. optimized RL


# In[ ]:




