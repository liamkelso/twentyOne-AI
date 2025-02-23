# interactive_game_separate.py
import numpy as np
import random

def human_choose_action():
    while True:
        try:
            action = int(input("Choose your action (0 for add 1, 1 for add 2): "))
            if action in [0, 1]:
                return action
        except ValueError:
            pass
        print("Invalid input. Please enter 0 or 1.")

def play_game():
    # Try to load the trained Q-table if it exists.
    try:
        Q = np.load("q_table.npy")
        print("Loaded trained Q-table.")
    except Exception as e:
        print("No trained Q-table found. AI will choose moves randomly.")
        Q = None

    current_count = 0
    print("Game start. Current count:", current_count)

    while True:
        # Human's turn:
        action = human_choose_action()
        add = 1 if action == 0 else 2
        current_count += add
        print(f"You chose: {'add 1' if action == 0 else 'add 2'} | New count: {current_count}")

        if current_count >= 21:
            print("You reached 21 or more on your move. You lose!")
            break

        # AI's turn:
        if Q is not None:
            # Use the Q-table to choose the best action given the current count.
            ai_action = int(np.argmax(Q[current_count]))
        else:
            ai_action = random.randint(0, 1)
        add = 1 if ai_action == 0 else 2
        current_count += add
        print(f"AI chose: {'add 1' if ai_action == 0 else 'add 2'} | New count: {current_count}")

        if current_count >= 21:
            print("AI reached 21 or more on its move. You win!")
            break

if __name__ == "__main__":
    play_game()
