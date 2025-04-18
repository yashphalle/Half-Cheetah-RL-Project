import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import re


methods = {
    'Anti Inversion': ['anti_inversion_ddpg_training.txt', 'anti_inversion_ddpg_training3.txt'],
    'Efficient Cheetah': ['efficient_cheetah_ddpg_training.txt', 'efficient_cheetah_ddpg_training3.txt'],
    'Gait Optimization': ['gait_optimization_ddpg_training.txt', 'gait_optimization_ddpg_training3.txt'],
    'Default': ['default.txt', 'default2.txt']
}

# Define colors for each method
colors = ['#4C72B0', '#55A868', '#C44E52', '#DDAA33']

# Function to calculate moving average
def moving_average(data, window_size=100):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Function to align timesteps for multiple trials
def align_timesteps(dfs, column='timestep', max_timestep=None):
    # Find the common timesteps or create a uniform grid
    if max_timestep is None:
        max_timestep = max([df[column].max() for df in dfs])
    
    # Create a uniform grid of timesteps
    step_size = min([df[column].iloc[1] - df[column].iloc[0] for df in dfs if len(df) > 1])
    aligned_timesteps = np.arange(0, max_timestep + step_size, step_size)
    
    return aligned_timesteps

# Function to interpolate values at aligned timesteps
def interpolate_values(df, aligned_timesteps, value_column, column='timestep'):
    # Sort by timestep
    df = df.sort_values(by=column)
    
    # Interpolate values at the aligned timesteps
    interpolated_values = np.interp(
        aligned_timesteps,
        df[column].values,
        df[value_column].values,
        left=np.nan, right=np.nan
    )
    
    return interpolated_values

# Function to plot with confidence bands
def plot_with_confidence_bands(methods, column_to_plot, title, ylabel, filename, window_size=100):
    plt.figure(figsize=(12, 8), dpi=100)
    
    for i, (method_name, file_paths) in enumerate(methods.items()):
        method_dfs = []
        
        # Load data for each trial
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                if column_to_plot in df.columns and 'timestep' in df.columns:
                    # Sort by timestep
                    df = df.sort_values(by='timestep')
                    method_dfs.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if len(method_dfs) == 0:
            print(f"No valid data found for {method_name}")
            continue
            
        # Align timesteps across trials
        aligned_timesteps = align_timesteps(method_dfs)
        
        # Interpolate values for each trial
        all_values = []
        for df in method_dfs:
            interpolated = interpolate_values(df, aligned_timesteps, column_to_plot)
            
            # Apply moving average
            if len(interpolated) >= window_size:
                smoothed_values = moving_average(interpolated, window_size)
                # Adjust timesteps for the moving average window
                smoothed_timesteps = aligned_timesteps[window_size-1:]
                all_values.append(smoothed_values)
            else:
                print(f"Warning: Not enough data points in a trial of {method_name} for a {window_size}-timestep moving average")
        
        if len(all_values) == 0:
            continue
            
        # Convert to numpy array for calculations
        all_values = np.array(all_values)
        
        # Calculate mean and standard deviation across trials
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)
        
        # Plot mean line
        plt.plot(smoothed_timesteps, mean_values, color=colors[i], linewidth=2, label=method_name)
        
        # Plot confidence band (mean ± std)
        plt.fill_between(
            smoothed_timesteps,
            mean_values - std_values,
            mean_values + std_values,
            color=colors[i],
            alpha=0.2
        )
    
    plt.title(f'{title} (with Confidence Bands)', fontsize=14)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")

# Plot distance traveled
plot_with_confidence_bands(
    methods,
    'distance',
    'Timestep vs Distance Traveled',
    'Distance Traveled',
    'distance_traveled_with_confidence.png'
)

# Plot rewards
plot_with_confidence_bands(
    methods,
    'reward',
    'Timestep vs Reward',
    'Reward',
    'reward_with_confidence.png'
)

# Plot control costs
plot_with_confidence_bands(
    methods,
    'control_cost',
    'Timestep vs Control Cost',
    'Control Cost',
    'control_cost_with_confidence.png'
)

print("All plots created successfully with confidence bands!")