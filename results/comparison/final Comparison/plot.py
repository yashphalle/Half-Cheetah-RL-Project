import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Read the data files
ppo_data = pd.read_csv('PPO.txt')
ddpg_data = pd.read_csv('DDPG.txt')

# Plot the reward vs timestep for both algorithms
plt.plot(ppo_data['timestep'], ppo_data['reward'], label='PPO', color='blue', linewidth=2)
plt.plot(ddpg_data['timestep'], ddpg_data['reward'], label='DDPG', color='red', linewidth=2)

# Add labels and title
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('PPO vs DDPG Performance Comparison', fontsize=16)
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Improve appearance
plt.tight_layout()

# Save the figure
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Optional: Calculate moving averages for smoother visualization
window_size = 5000  # Adjust this value based on your data density

# Create a second plot with smoothed curves
plt.figure(figsize=(12, 6))

# Calculate and plot moving averages
ppo_data['smooth_reward'] = ppo_data['reward'].rolling(window=window_size).mean()
ddpg_data['smooth_reward'] = ddpg_data['reward'].rolling(window=window_size).mean()

plt.plot(ppo_data['timestep'], ppo_data['smooth_reward'], label='PPO (Smoothed)', color='blue', linewidth=2)
plt.plot(ddpg_data['timestep'], ddpg_data['smooth_reward'], label='DDPG (Smoothed)', color='red', linewidth=2)

# Add labels and title for the smoothed plot
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('Reward (Moving Average)', fontsize=14)
plt.title(f'PPO vs DDPG Performance Comparison (Moving Avg, Window={window_size})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the smoothed figure
plt.savefig('algorithm_comparison_smoothed.png', dpi=300, bbox_inches='tight')

# Show the smoothed plot
plt.show()

# Optional: Print some statistics
print("PPO Statistics:")
print(f"Min reward: {ppo_data['reward'].min()}")
print(f"Max reward: {ppo_data['reward'].max()}")
print(f"Mean reward: {ppo_data['reward'].mean()}")

print("\nDDPG Statistics:")
print(f"Min reward: {ddpg_data['reward'].min()}")
print(f"Max reward: {ddpg_data['reward'].max()}")
print(f"Mean reward: {ddpg_data['reward'].mean()}")