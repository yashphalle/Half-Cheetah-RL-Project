# Half-Cheetah-RL-Project

A reinforcement learning implementation for training a Half-Cheetah agent in the MuJoCo simulation environment.

## Overview

This project implements reinforcement learning algorithms to train a Half-Cheetah agent to run forward as efficiently as possible. The Half-Cheetah environment is part of the MuJoCo environments in Gymnasium (formerly OpenAI Gym) and features a 2D cheetah model with multiple joints that must learn to coordinate movement to maximize forward speed.

## Environment Description

The Half-Cheetah is a 2D robot consisting of:
- 9 links and 8 joints (including two paws)
- 6 degrees of freedom in the continuous action space (values between -1 and +1)
- 17-dimensional observation space (joint positions and velocities)
- Reward based on forward distance traveled (We have experimented with Custom Reward functions- Available in Codes/Reward Comaprison 

## Installation

```bash
# Clone the repository
git clone https://github.com/yashphalle/Half-Cheetah-RL-Project.git
cd Half-Cheetah-RL-Project

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

##Implemented Algorithms
This repository implements and compares Deep Deterministic Policy Gradient (DDPG)
and Proximal Policy Optimization (PPO).

   
