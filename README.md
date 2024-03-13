# Walk-These-Ways-Navigation
# Quadruped Navigation and Obstacle Avoidance

This repository contains the implementation of a quadruped robot navigation and obstacle avoidance project using reinforcement learning in the Isaac Gym simulator. The project was completed as part of the CS562: Advanced Robotics course at Rutgers University.

## Project Overview

The main objective of this project is to enable a robotic dog (Unitree Go1) to navigate from point A to point B within an environment, while avoiding obstacles and collisions with walls. The project is divided into three phases:

1. **Robot Locomotion**: Train a velocity-conditioned neural network policy to control the robot's locomotion, enabling it to walk at commanded velocities.
2. **Robot Navigation within Walls**: Train a policy using reinforcement learning to navigate a walled corridor from a start position to a goal position within a time limit, without colliding with the walls.
3. **Robot Locomotion with Obstacle Avoidance**: Build a new navigation policy with obstacle avoidance capabilities, enabling the robot to avoid obstacles present in the environment.

## Installation

To run this project, you need to have Isaac Gym installed. Follow the installation instructions provided in the Isaac Gym documentation: [Isaac Gym Installation](https://developer.nvidia.com/isaac-gym)

## Usage

1. **Phase 1: Robot Locomotion**
   - Run the script `scripts/play.py` to train the locomotion policy.
   - The trained policy will enable the robot to walk at commanded velocities.

2. **Phase 2: Robot Navigation within Walls with Obstacle Avoidance**
   - Run the script `navigator.py` to train the navigation policy.
   - The trained policy will navigate the robot from the start position to the goal position, avoiding collisions with walls.

## Results

The repository includes recorded trajectories, plots, and presentations demonstrating the performance of the trained policies in each phase of the project.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## Acknowledgments

This project was completed as part of the CS562: Advanced Robotics course at Rutgers University, under the guidance of Prof. Kostas Bekris and TA Dhruv Metha Ramesh.
