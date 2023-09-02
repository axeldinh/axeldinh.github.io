---
title: "Self-Taught Reinforcement Learning"
layout: post
date: 2023-04-01 00:00
tag: Reinforcement Learning
image: /assets/projects/rl_learning/icon.gif
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "Implementation of a small RL library in Python. The goal was to learn the basics of RL by implementing the algorithms from scratch."
category: project
author: axeldinh
externalLink: false 
github: https://github.com/axeldinh/rlib.git
icon: /assets/projects/rl_learning/icon.gif
---


In this project, I developed a small Reinforcement Learning library in Python. The goal was to learn the basics of RL by implementing the algorithms from scratch.

The [documentation](https://axeldinh.github.io/rlib/) of the code has been made using `Sphinx` and the library is available on [GitHub](https://github.com/axeldinh/rlib.git).

## The Library

I wanted the library to be as simple as possible to use. The user only needs to define the environment and the agent, and then run the training loop on a [Gymnasium](https://gymnasium.farama.org/) environment, predefined in the library, or on a custom environment. For example, the following code trains a `PPO` agent on the `BipedalWalker-v3` environment, where a 2D character must learn to walk:

```python
from rlib.learning import PPO

env_kwargs = {'id': 'BipedalWalker-v3'}
agent_kwargs = {'hidden_sizes': [256, 256], 'activation': 'tanh'}

agent = PPO(env_kwargs, agent_kwargs)
agent.train()
```

By doing so, the user can solely focus on the implementation of the environment. The list of different algorithms available in the library is the following:

- `Q-Learning`, where the Q-function is a table of size defined by the user.
- `Deep Q-Learning`, where the Q-function is approximated by a neural network. Does not handle continuous action spaces.
- `Evolution Strategy`, where an estimate of the gradient is computed using randomly sampled perturbations of the parameters. Handles both discrete and continuous action spaces.
- `Deep Deterministic Policy Gradient (DDPG)`, where the policy is approximated by a neural network. Only handles continuous action spaces.
- `Proximal Policy Optimization (PPO)`, where rather than optimizing the Q-function, the policy is optimized directly. Handles both discrete and continuous action spaces.

## Results

We display here some results obtained with the library.

### Q-Learning

We trained a `Q-Learning` agent on the `MountainCar-v0` environment, where a car must learn to climb a hill. The agent was trained on 100,000 episodes and the following plot shows the evolution of the reward during training:

![rewards\label{q_learning_reward}](/assets/projects/rl_learning/qlearning/rewards.png)

The reward here is the opposite of the number of steps before the car reaches the top of the hill, it is hence negative. We can see that the agent learns to climb the hill after about 10,000 episodes, and then improves its performance until the end of the training.

Here is a showcase of the training of the agent:

|The agent moves randomly at the beginning of the training|The agent learned how to swing but struggles to climb the hill| The agent learned how to climb the hill|
|:---:|:---:|:---:|
|![q_learning_ex1](/assets/projects/rl_learning/qlearning/qlearning_iter0.gif)|![q_learning_ex2](/assets/projects/rl_learning/qlearning/qlearning_iter50000.gif)|![q_learning_ex3](/assets/projects/rl_learning/qlearning/qlearning_iter100000.gif)|

Another interesting visualization is to plot the Q-function learned by the agent. This is possible has only two dimensions are needed to represent the state of the environment. By applying a softmax function on the Q-function, we can obtain a sort of probability distribution over the possible actions for each state. I.e. it shows how bigger or lower the impact of one action is compared to the others. The following plots show the Q-function learned by the agent after 100,000 episodes:

|Probability to go left|Probability to do nothing|Probability to go right|
|:---:|:---:|:---:|
|![q_learning_left](/assets/projects/rl_learning/qlearning/q_table_left.png)|![q_learning_nothing](/assets/projects/rl_learning/qlearning/q_table_nothing.png)|![q_learning_right](/assets/projects/rl_learning/qlearning/q_table_right.png)|

We can see a swirling curve on which the agent will move left or right. And that doing nothing is not a good option, as no area of the plot has a high value. Note that the pixelated areas are states where the agent has never been, or not often enough to have a good estimate of the Q-function.
