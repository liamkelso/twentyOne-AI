# q_learning_agent_21.py
import numpy as np
import random
import time
from counting_game_env import CountingGame21Env

# Initialize environment
env = CountingGame21Env()

# Create a Q-table with dimensions [state, action].
# There are 22 possible states (0 to 21) and 2 actions.
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.2    # Exploration rate
episodes = 1000  # Number of training episodes

for episode in range(episodes):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        # Epsilon-greedy policy: choose a random action or the best known action.
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, _ = env.step(action)
        
        # Q-learning update rule:
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        steps += 1
        
        # Print the current state for real-time feedback.
        print(f"Episode {episode} | Step {steps} | Current Count: {state} | Reward: {reward}")
        time.sleep(0.5)  # Delay to slow down output for observation

    print(f"Episode {episode} completed in {steps} steps\n")

print("Training complete!")
print("Learned Q-table:")
print(Q)
np.save("q_table.npy", Q)


