import gymnasium as gym

env = gym.make("HalfCheetah-v4", render_mode="human")  # Set render_mode to None for training
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Take random actions
    obs, reward, done, truncated, info = env.step(action)

env.close()
