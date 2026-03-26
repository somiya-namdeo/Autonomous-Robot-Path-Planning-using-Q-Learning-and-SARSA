# Importing classes
import numpy as np
import pandas as pd
from env import Environment
from agent_brain import QLearningTable


def update():
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []

    for episode in range(1000):
        # Initial Observation
        observation = env.reset()

        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0

        while True:
            # Refreshing environment
            env.render()

            # RL chooses action based on observation
            action = RL.choose_action(str(observation))

            # RL takes an action and get the next observation and reward
            observation_, reward, done = env.step(action)

            # RL learns from this transition and calculating the cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            # Swapping the observations - current and next
            observation = observation_

            # Calculating number of Steps in the current Episode
            i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                break

    # Showing the final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Save results to CSV
    df = pd.DataFrame({
    "steps": steps,
    "cost": all_costs
    })
    df.to_csv("q_learning.csv", index=False)

    # Performance metrics
    print("\n------ Q-LEARNING PERFORMANCE ------")

    print("Average Steps:", np.mean(steps))
    print("Final 50 Avg Steps:", np.mean(steps[-50:]))

    print("Min Steps:", np.min(steps))
    print("Max Steps:", np.max(steps))

    print("Variance:", np.var(steps))

    print("Average Cost:", np.mean(all_costs))
    print("Max Reward:", np.max(all_costs))

    print("Efficiency (%):",
      ((np.mean(steps[:100]) - np.mean(steps[-100:]))
       / np.mean(steps[:100])) * 100)

    print("Convergence (Last 50 Avg Steps):",
    np.mean(steps[-50:]))
    # Plotting the results
    RL.plot_results(steps, all_costs)


# Commands to be implemented after running this file
if __name__ == "__main__":
    # Calling for the environment
    env = Environment()
    # Calling for the main algorithm
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # Running the main loop with Episodes by calling the function update()
    env.after(100, update)  # Or just update()
    env.mainloop()
     

