import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data from training_log.txt
data = pd.read_csv('training_log.txt')

# Calculate the moving average for reward
# Window size of 5 (you can adjust this value as needed)
window_size = 5000
data['reward_moving_avg'] = data['reward'].rolling(window=window_size).mean()

# Plot the reward vs timestep
plt.figure(figsize=(10, 6))
plt.plot(data['timestep'], data['reward'], color='blue', linewidth=1.5, 
         label='Raw Reward', alpha=0.7)
plt.plot(data['timestep'], data['reward_moving_avg'], color='red', linewidth=2.5, 
         label=f'Moving Average (window={window_size})')

plt.title('Reward vs Timestep')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add a bit of padding to the x and y limits for better visualization
plt.xlim(min(data['timestep']) - 10, max(data['timestep']) + 10)
plt.tight_layout()

# Save the figure (optional)
plt.savefig('reward_vs_timestep.png', dpi=300)

# Show the plot
plt.show()