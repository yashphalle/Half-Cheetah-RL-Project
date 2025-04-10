import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

def plot_ddpg_training(file_path, window_size=10, figsize=(12, 8), save_path=None):
    """
    Plot DDPG training rewards from a log file.
    
    Parameters:
    -----------
    file_path : str
        Path to the training log file (.txt or .csv)
    window_size : int
        Window size for the moving average smoother
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path
    """
    # Read the data
    try:
        data = pd.read_csv(file_path)
    except:
        # Handle potential issues with the file format
        try:
            data = pd.read_csv(file_path, sep=None, engine='python')
        except:
            print(f"Error reading file: {file_path}")
            return
    
    # Make sure we have the expected columns
    if 'episode' not in data.columns or 'reward' not in data.columns:
        print("Required columns 'episode' and 'reward' not found in the data.")
        return
    
    # Sort by episode number if needed
    data = data.sort_values('episode')
    
    # Calculate rolling average for smoothing
    data['smoothed_reward'] = data['reward'].rolling(window=window_size, min_periods=1).mean()
    
    # Setup the figure
    plt.figure(figsize=figsize)
    
    # Plot raw rewards
    plt.plot(data['episode'], data['reward'], color='royalblue', alpha=0.5, 
             label='Raw Reward', marker='.', linestyle='-', markersize=3)
    
    # Plot smoothed rewards
    plt.plot(data['episode'], data['smoothed_reward'], color='crimson', 
             label=f'{window_size}-Episode Moving Average', linewidth=2)
    
    # Add horizontal reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Color zones (training phases)
    quartile_size = len(data) // 4
    
    # Define key episodes and their descriptions
    key_episodes = [
        (10, 'First Consistent Positive Reward'),
        (28, 'Stabilization Point'),
        (58, 'Major Drop'),
        (82, 'Drop'),
        (106, 'High Performance Plateau')
    ]
    
    # Mark quarters
    quartiles = [
        (quartile_size, 'Q1 End'),
        (quartile_size * 2, 'Q2 End'),
        (quartile_size * 3, 'Q3 End')
    ]
    
    # Add vertical lines for key episodes
    for episode, label in key_episodes:
        if episode in data['episode'].values:
            plt.axvline(x=episode, color='purple', linestyle='--', alpha=0.5)
            plt.text(episode, data['reward'].max() * 0.95, label, 
                    rotation=90, verticalalignment='top', fontsize=8, color='purple')
    
    # Add vertical lines for quarters
    for episode, label in quartiles:
        if episode in data['episode'].values:
            plt.axvline(x=episode, color='darkgray', linestyle=':', alpha=0.5)
            plt.text(episode, data['reward'].min() * 0.9, label,
                    rotation=90, verticalalignment='bottom', fontsize=8, color='darkgray')
    
    # Calculate statistics for annotation
    stats = {
        'min_reward': data['reward'].min(),
        'max_reward': data['reward'].max(),
        'final_reward': data['reward'].iloc[-1],
        'final_smoothed': data['smoothed_reward'].iloc[-1],
        'improvement_rate': np.polyfit(data['episode'], data['reward'], 1)[0]
    }
    
    # Add statistical annotation
    stat_text = (
        f"Episodes: {len(data)}\n"
        f"Min Reward: {stats['min_reward']:.2f}\n"
        f"Max Reward: {stats['max_reward']:.2f}\n"
        f"Final Reward: {stats['final_reward']:.2f}\n"
        f"Final Moving Avg: {stats['final_smoothed']:.2f}\n"
        f"Avg Improvement/Episode: {stats['improvement_rate']:.2f}"
    )
    
    # Position the statistics text box
    plt.text(0.02, 0.97, stat_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Analyze training phases based on reward patterns
    # This adds colorized backgrounds to represent different phases
    
    # Define custom training phases and colors
    # (start_episode, label, color, alpha)
    phases = [
        (0, 'Exploration', 'lightcoral', 0.2),
        (10, 'Early Learning', 'khaki', 0.2),
        (30, 'Acceleration', 'lightgreen', 0.2),
        (60, 'Refinement', 'lightblue', 0.2),
        (90, 'Convergence', 'lavender', 0.2)
    ]
    
    # Add colored background for training phases
    handles = []
    for i, (start_ep, label, color, alpha) in enumerate(phases):
        end_ep = phases[i+1][0] if i < len(phases)-1 else data['episode'].iloc[-1]
        
        # Add rectangle patch for the phase
        rect = plt.axvspan(start_ep, end_ep, alpha=alpha, color=color)
        handles.append(mpatches.Patch(color=color, label=f'{label} Phase', alpha=alpha))
    
    # Configure the plot
    plt.title('DDPG Training Performance Analysis', fontsize=16, pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to give some padding
    plt.ylim(data['reward'].min() - 100, data['reward'].max() + 100)
    
    # Force integer tick labels on x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legends
    plt.legend(handles=handles, loc='lower right', fontsize=9)
    plt.legend(loc='upper left', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "training_log.txt"  # Default file name
    
    save_path = file_path.replace('.txt', '.png').replace('.csv', '.png')
    plot_ddpg_training(file_path, save_path=save_path)