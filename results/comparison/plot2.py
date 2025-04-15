import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read data from a text file and plot it
def read_and_plot_rl_data(txt_file_path):
    # Read the text file using pandas (it can read CSV-formatted text files)
    df = pd.read_csv(txt_file_path)
    
    # Print basic statistics
    print(f"Data contains {len(df)} timesteps")
    print(f"Average reward: {df['reward'].mean():.4f}")
    
    # Create a more appealing plot with Seaborn
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot reward vs timestep
    sns.lineplot(x='timestep', y='reward', data=df, linewidth=2, color='#1E88E5')
    
    # Add labels and title
    plt.title('Reward vs Timestep', fontsize=16)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('reward_vs_timestep.png', dpi=300)
    
    # Show the plot
    plt.show()
    
    return df

# Usage example
if __name__ == "__main__":
    # Replace 'your_data.txt' with your actual file path
    df = read_and_plot_rl_data('DDPG.txt')
    
    # Optional: If you want to plot other metrics as well (critic_loss, actor_loss)
    plt.figure(figsize=(15, 10))
    
    # Create 3 subplots
    plt.subplot(3, 1, 1)
    sns.lineplot(x='timestep', y='reward', data=df, color='blue')
    plt.title('Reward over Time')
    
    plt.subplot(3, 1, 2)
    sns.lineplot(x='timestep', y='critic_loss', data=df, color='green')
    plt.title('Critic Loss over Time')
    
    plt.subplot(3, 1, 3)
    sns.lineplot(x='timestep', y='actor_loss', data=df, color='red')
    plt.title('Actor Loss over Time')
    
    plt.tight_layout()
    plt.savefig('all_metrics.png', dpi=300)
    plt.show()