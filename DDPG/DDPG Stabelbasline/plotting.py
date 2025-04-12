import matplotlib.pyplot as plt
import numpy as np
import re


def extract_data(log_file_path):
    timesteps = []
    rewards = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
           
            match = re.match(r'Timestep: (\d+), Episode Reward: ([-\d\.]+)', line)
            if match:
                timestep = int(match.group(1))
                reward = float(match.group(2))
                timesteps.append(timestep)
                rewards.append(reward)
    
    return timesteps, rewards

log_file_path = 'training_log.txt'  # Change this to your log file path
timesteps, rewards = extract_data(log_file_path)

# Calculate the moving average for reward
window_size = 5  # Smaller window size since we might have fewer data points
rewards_array = np.array(rewards)
moving_avg = []

for i in range(len(rewards)):
    if i < window_size - 1:
        # For the first few points where we don't have enough previous data
        moving_avg.append(np.mean(rewards_array[:i+1]))
    else:
        moving_avg.append(np.mean(rewards_array[i-window_size+1:i+1]))

# Plot the reward vs timestep
plt.figure(figsize=(12, 6))
plt.plot(timesteps, rewards, 'o-', color='blue', linewidth=1.5, 
         label='Episode Reward', alpha=0.7)
plt.plot(timesteps, moving_avg, 'r-', linewidth=2.5, 
         label=f'Moving Average (window={window_size})')

plt.title('Episode Reward vs Timestep')
plt.xlabel('Timestep')
plt.ylabel('Episode Reward')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

x_padding = (max(timesteps) - min(timesteps)) * 0.05
plt.xlim(min(timesteps) - x_padding, max(timesteps) + x_padding)

# # Add annotations for some data points
# for i, (x, y) in enumerate(zip(timesteps, rewards)):
#     if i % 3 == 0:  # Annotate every third point to avoid clutter
#         plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
#                     xytext=(0,10), ha='center')

plt.tight_layout()

# Save the figure (optional)
plt.savefig('reward_vs_timestep.png', dpi=300)

# Show the plot
plt.show()