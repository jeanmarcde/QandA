# In computer science, Q* (or Q-star) is often associated with the concept of Q-learning, a type of reinforcement learning algorithm. Q-learning is an off-policy learner that seeks to find the best action to take given the current state. It's part of the larger family of machine learning algorithms and is primarily used in the field of artificial intelligence.

# The "Q" in Q-learning stands for the quality of a particular action taken in a given state. The algorithm estimates the quality of a state-action combination (state and action pair), which helps to guide the decision-making process in environments where the outcomes of actions are partly random and partly under the control of a decision-maker.

# A simple example of Q-learning could be a robot navigating a maze. The robot learns to reach the goal by trying different paths and receiving feedback (rewards or penalties) based on its actions. Over time, it builds a Q-table that represents its learned knowledge, indicating the expected utility of taking a given action in a given state.

# Here's a rudimentary example:

# Initialize the Q-table: The Q-table is initialized with zeros. The table has rows for each state and columns for each action.

# Explore or Exploit: The algorithm chooses whether to try a new action (explore) or use an action it believes has the best long-term effect (exploit).

# Update the Q-table: After taking an action and observing the outcome and the reward, the Q-value for the state-action pair is updated using the formula:

# Q(state,action)=Q(state,action)+α×(reward+γ×maxQ(nextstate,allactions)−Q(state,action))

# Where:
# α is the learning rate.
# γ is the discount factor.
# Repeat: This process is repeated for each state-action pair until the learning is completed.

# The Python code provided demonstrates a simple Q-learning example. In this example, a Q-table is used to learn the best actions for a hypothetical robot navigating a simple environment. The key aspects of this code are:

# Initialization of the Q-table: The Q-table is initialized with zeros and has dimensions corresponding to the number of states and actions.

# Environment Setup: A reward matrix R is defined to simulate the rewards received for each state-action pair. For simplicity, the rewards are predefined, with 100 representing a goal state.

# Training Loop: The algorithm iterates 10,000 times, simulating the learning process. In each iteration:

# A state is chosen randomly.
# An action is chosen randomly for this example (in a more sophisticated setup, a policy like epsilon-greedy could be used).
# The Q-table is updated using the Q-learning formula:

# Q(state,action)=Q(state,action)+α×(reward+γ×maxQ(nextstate,allactions)−Q(state,action))

# where 
# α is the learning rate and 
# γ is the discount factor.
# Q-Table Update: The Q-values are updated to reflect the learned values, which estimate the utility of taking a certain action in a given state.

# The final output is the Q-table after the learning process, with each entry indicating the learned quality of taking a certain action in a given state. This table guides the decision-making process for the agent.

import numpy as np
import random

# Defining the parameters
gamma = 0.8  # Discount factor
alpha = 0.9  # Learning rate
num_states = 6
num_actions = 6

# Initializing the Q-table
Q = np.zeros((num_states, num_actions))

# Example environment: rewards for each state-action pair
R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# Training phase
for i in range(10000):
    # Randomly choosing the initial state
    current_state = np.random.randint(0, num_states)

    while True:
        # Choosing an action (randomly for this example)
        action = np.random.randint(0, num_actions)

        # Taking the action and moving to the next state
        next_state = action

        # Q-learning formula
        Q[current_state, action] = Q[current_state, action] + alpha * (R[current_state, action] + gamma * np.max(Q[next_state, :]) - Q[current_state, action])

        # Move to the next state
        current_state = next_state

        # Break if the goal state is reached
        if R[current_state, action] == 100:
            break

Q  # Display the final Q-table