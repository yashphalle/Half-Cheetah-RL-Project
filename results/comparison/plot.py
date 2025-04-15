import matplotlib.pyplot as plt
import re
import numpy as np

# Hardcoded file path - update this to your log file location
log_file_path = "PPO.txt"

def parse_rewards(file_path):
    """Parse RL log file and extract reward values over timesteps."""
    timesteps = []
    rewards = []
    current_timestep = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Extract timestep
            timestep_match = re.search(r'total_timesteps\s+\|\s+(\d+)', line)
            if timestep_match:
                current_timestep = int(timestep_match.group(1))
                
            # Extract reward
            reward_match = re.search(r'\|\s+ep_rew_mean\s+\|\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+\|', line)
            if reward_match and current_timestep is not None:
                reward_value = float(reward_match.group(1))
                timesteps.append(current_timestep)
                rewards.append(reward_value)
    
    return timesteps, rewards

# Parse the rewards
timesteps, rewards = parse_rewards(log_file_path)

if not timesteps:
    print("No reward data found in the log file!")
else:
    # Plot rewards with improved x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, 'b-', linewidth=2)
    
    # Improve title and labels
    plt.title('Mean Episode Reward Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Timesteps (thousands)', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    
    # Format x-axis ticks to show thousands
    def format_ticks(x, pos):
        return f'{int(x/1000)}k'
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    # Set x-axis to start from 0 with equal divisions
    max_timestep = max(timesteps) if timesteps else 1000000
    num_divisions = 10  # Number of divisions on x-axis
    
    # Create evenly spaced ticks from 0 to max_timestep
    tick_positions = np.linspace(0, max_timestep, num_divisions)
    plt.xlim(0, max_timestep)  # Set x-axis limits explicitly
    plt.xticks(tick_positions)
    
    # Add grid for better readability
    plt.grid(True, linestyle='-', alpha=0.7)
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("reward_plot.png", dpi=300)
    print("Done! Plot saved as reward_plot.png with proper x-axis starting from 0")