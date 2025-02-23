# counting_game_env.py
import gym
from gym import spaces
import random

class CountingGame21Env(gym.Env):
    """
    A simple environment for the counting game 21.
    - The state is a single integer representing the current count.
    - Actions: 0 means add 1, 1 means add 2.
    - The agent makes a move, then the opponent (a random agent) makes a move.
    - If the count reaches or exceeds 21, the game ends.
      The agent loses if it is responsible for reaching 21 or more.
      The agent wins if the opponent causes the count to reach 21 or more.
    """
    def __init__(self):
        super(CountingGame21Env, self).__init__()
        self.current_count = 0
        # There are two actions: add 1 or add 2.
        self.action_space = spaces.Discrete(2)
        # The observation is the current count (0 to 21 inclusive).
        self.observation_space = spaces.Discrete(22)
        self.done = False

    def reset(self):
        self.current_count = 0
        self.done = False
        return self.current_count

    def step(self, action):
        # Agent's turn: map action to number to add.
        add_agent = 1 if action == 0 else 2
        self.current_count += add_agent

        # If the agent causes the count to reach/exceed 21, it loses.
        if self.current_count >= 21:
            reward = -1
            self.done = True
            # Clip state to 21 so it’s a valid index.
            return min(self.current_count, 21), reward, self.done, {}

        # Opponent's turn: choose randomly between adding 1 or 2.
        opp_action = random.choice([0, 1])
        add_opp = 1 if opp_action == 0 else 2
        self.current_count += add_opp

        # If the opponent causes the count to reach/exceed 21, the agent wins.
        if self.current_count >= 21:
            reward = 1
            self.done = True
        else:
            reward = 0  # no reward if game continues

        # Clip state to 21 if needed.
        return min(self.current_count, 21), reward, self.done, {}

if __name__ == "__main__":
    env = CountingGame21Env()
    state = env.reset()
    print("Starting count:", state)
    for _ in range(5):
        action = int(input("Enter action (0 for add 1, 1 for add 2): "))
        state, reward, done, _ = env.step(action)
        print("New count:", state, "Reward:", reward, "Done:", done)
        if done:
            break
