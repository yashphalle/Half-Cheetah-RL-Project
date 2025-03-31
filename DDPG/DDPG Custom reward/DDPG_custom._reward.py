import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.file = open("reward_loss_log.txt", "w")
        self.file.write("timestep,reward,loss\n")
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]["r"]
            critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0)
            actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0)
            self.file.write(f"{self.num_timesteps},{reward},{critic_loss},{actor_loss}\n")
        return True
    
    def __del__(self):
        self.file.close()

class CustomHalfCheetahEnv(gym.Wrapper):
    
    def __init__(self):
        env = gym.make("HalfCheetah-v4")
        super().__init__(env)
        
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        reward = self.modified_reward_function(observation, action, original_reward)
        return observation, reward, terminated, truncated, info
    
    def modified_reward_function(self, observation, action, original_reward):
        # Forward velocity reward (using x-coordinate velocity, index 8)
        forward_reward = 1.0 * observation[8]
        
        # Penalize excessive vertical movement (z-coordinate, index 0)
        height_penalty = -0.05 * abs(observation[0] - 0.5)  
        
        # Penalize excessive rotations for stability (angle of second rotor, index 2)
        rotation_penalty = -0.1 * abs(observation[2])  
        
        # Energy efficiency - penalize excessive joint movements
        # Angular velocities from indices 10-16
        energy_penalty = -0.001 * sum(abs(observation[i]) for i in range(10, 17))
        
        # Smooth control - penalize large action changes
        control_penalty = -0.01 * np.sum(np.square(action))
        
        # Balance original reward with custom components
        original_reward_weight = 0.5
        
        # Combined reward
        reward = (
            forward_reward +
            height_penalty +
            rotation_penalty +
            energy_penalty +
            control_penalty +
            original_reward_weight * original_reward
        )
        
        return reward

env = CustomHalfCheetahEnv()


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    

model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    buffer_size=1000000,       # Replay buffer size
    learning_rate=1e-3,        # Learning rate for both actor and critic
    batch_size=100,            # Batch size for training
    tau=0.005,                 # Soft update coefficient
    gamma=0.99,                # Discount factor
    train_freq=(1, "episode"), # Train the policy every episode
    gradient_steps=-1,         # -1 means use buffer_size/batch_size steps
    verbose=1
)
    
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
                                         name_prefix='ddpg_halfcheetah')
logging_callback = LoggingCallback()


logging_callback.file.close()
logging_callback.file = open("reward_loss_log.txt", "w")
logging_callback.file.write("timestep,reward,critic_loss,actor_loss\n")

model.learn(total_timesteps=1000000, callback=[checkpoint_callback, logging_callback])
       
print("Training completed!")