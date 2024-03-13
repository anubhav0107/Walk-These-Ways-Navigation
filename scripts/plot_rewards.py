import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rewards(base_file_name, save_path=None):
    # Create unique file names for each type of reward
    filenames = ['_goal_reward_', '_timeout_penalty_', '_wall_penalty_', '_total_reward_']

     # Read existing data from the CSV files if they exist
    total_reward_data = pd.DataFrame()
    goal_reward_data = pd.DataFrame()
    wall_penalty_data = pd.DataFrame()
    timeout_penalty_data = pd.DataFrame()

    for i in range(3):
        goal_reward_data = pd.concat([goal_reward_data,pd.read_csv(f'{base_file_name}{filenames[0]}{i}.csv')], ignore_index=True)
        wall_penalty_data = pd.concat([wall_penalty_data,pd.read_csv(f'{base_file_name}{filenames[2]}{i}.csv')], ignore_index=True)
        timeout_penalty_data = pd.concat([timeout_penalty_data,pd.read_csv(f'{base_file_name}{filenames[1]}{i}.csv')], ignore_index=True)
        total_reward_data = pd.concat([total_reward_data,pd.read_csv(f'{base_file_name}{filenames[3]}{i}.csv')], ignore_index=True)
    
    # total_reward_file_name = f'{base_file_name}_total_reward.csv'
    # goal_reward_file_name = f'{base_file_name}_goal_reward.csv'
    # wall_penalty_file_name = f'{base_file_name}_wall_penalty.csv'
    # timeout_penalty_file_name = f'{base_file_name}_timeout_penalty.csv'

    # Check if the CSV files exist
    # total_reward_exists = os.path.exists(total_reward_file_name)
    # goal_reward_exists = os.path.exists(goal_reward_file_name)
    # wall_penalty_exists = os.path.exists(wall_penalty_file_name)
    # timeout_penalty_exists = os.path.exists(timeout_penalty_file_name)

   

    # #if total_reward_exists:
    # total_reward_data = pd.read_csv(total_reward_file_name, header=None)

    # #if goal_reward_exists:
    # goal_reward_data = pd.read_csv(goal_reward_file_name, header=None)

    # #if wall_penalty_exists:
    # wall_penalty_data = pd.read_csv(wall_penalty_file_name, header=None)

    # #if timeout_penalty_exists:
    # timeout_penalty_data = pd.read_csv(timeout_penalty_file_name, header=None)

    import numpy as np
    total_reward_data = total_reward_data.sum(axis=1)
    goal_reward_data = goal_reward_data.sum(axis=1)
    wall_penalty_data = wall_penalty_data.sum(axis=1)
    timeout_penalty_data = timeout_penalty_data.sum(axis=1)

    # Apply rolling average for smoothing
    window_size = 150  # Adjust this based on your preference
    total_reward_data = total_reward_data.rolling(window=window_size, center=True).mean()
    goal_reward_data = goal_reward_data.rolling(window=window_size, center=True).mean()
    wall_penalty_data = wall_penalty_data.rolling(window=window_size, center=True).mean()
    timeout_penalty_data = timeout_penalty_data.rolling(window=window_size, center=True).mean()

    # Plot and save each reward type individually
    save_plots = []

    plt.figure(figsize=(10, 6))
    plt.plot(total_reward_data)
    plt.ylim(0, 16)  # Adjust the range based on your data
    plt.yticks(np.arange(0, 16, 5))
    plt.title('Total Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    if save_path:
        save_path_total = os.path.join(save_path, 'total_reward_plot.png')
        plt.savefig(save_path_total)
        save_plots.append(save_path_total)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(goal_reward_data, color='red')
    plt.ylim(0, 16)  # Adjust the range based on your data
    plt.yticks(np.arange(0, 16, 5))
    plt.title('Goal Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    if save_path:
        save_path_goal = os.path.join(save_path, 'goal_reward_plot.png')
        plt.savefig(save_path_goal)
        save_plots.append(save_path_goal)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(wall_penalty_data, color='cyan')
    plt.title('Wall Penalty')
    plt.xlabel('Time Step')
    plt.ylabel('Penalty')
    if save_path:
        save_path_wall = os.path.join(save_path, 'wall_penalty_plot.png')
        plt.savefig(save_path_wall)
        save_plots.append(save_path_wall)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(timeout_penalty_data, color='brown')
    plt.title('Timeout Penalty')
    plt.xlabel('Time Step')
    plt.ylabel('Penalty')
    if save_path:
        save_path_timeout = os.path.join(save_path, 'timeout_penalty_plot.png')
        plt.savefig(save_path_timeout)
        save_plots.append(save_path_timeout)
    plt.show()

    return save_plots

# Example usage with saving to a specific path
save_paths = plot_rewards("/common/home/ag2112/walk-these-ways/walk-these-ways/rewards/storage", save_path="/common/home/ag2112/walk-these-ways/walk-these-ways/imdump/imdump2")

# save_paths will contain the paths to the saved plots if save_path is provided
print("Saved plots:", save_paths)
