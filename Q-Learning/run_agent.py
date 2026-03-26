from env import Environment
from agent_brain import QLearningTable
import pandas as pd
import numpy as np

def run_episodes():
    episode_steps = []  # List to store the number of steps for each episode
    episode_costs = []  # List to store the cost for each episode

    for episode in range(1000):
        # Reset the environment to the initial observation
        observation = env.reset()

        # Initialize episode-specific variables
        num_steps = 0
        episode_cost = 0

        while True:
            # Render the environment
            env.render()

            # Agent chooses an action based on the current observation
            action = agent.choose_action(str(observation))

            # Agent takes an action and receives the next observation and reward
            next_observation, reward, done = env.step(action)

            # Agent learns from this transition and calculates the cost
            episode_cost += agent.learn(str(observation), action, reward, str(next_observation))

            # Update the current observation
            observation = next_observation

            # Increment the step count
            num_steps += 1

            # Break the loop when the episode ends (agent reaches the goal or an obstacle)
            if done:
                episode_steps.append(num_steps)
                episode_costs.append(episode_cost)
                break

    # Show the final route
    env.final()
    

    #Performance metrics
    import numpy as np

    print("\n------ PERFORMANCE METRICS ------")

    print("Average Steps:", np.mean(episode_steps))
    print("Final 50 Episodes Avg Steps:", np.mean(episode_steps[-50:]))

    print("Min Steps (Best Path):", np.min(episode_steps))
    print("Max Steps (Worst Case):", np.max(episode_steps))

    print("Step Variance (Stability):", np.var(episode_steps))

    print("Average Cost:", np.mean(episode_costs))
    print("Max Reward:", np.max(episode_costs))

    print("Efficiency (Steps Reduction %):",
      ((np.mean(episode_steps[:100]) - np.mean(episode_steps[-100:])) 
       / np.mean(episode_steps[:100])) * 100)
    print("Convergence (Last 50 Avg Steps):", np.mean(episode_steps[-50:]))
    
    # Show the Q-table with values for each action
    agent.print_q_table()
    
    df=pd.DataFrame({
        "steps":episode_steps,
        "cost":episode_costs
    })
    df.to_csv("q_learning.csv",index=False)
    # Plot the results
    agent.plot_results(episode_steps, episode_costs)

# Entry point
if __name__ == "__main__":
    # Create the environment
    env = Environment()

    # Create the Q-learning agent
    agent = QLearningTable(actions=list(range(env.n_actions)))

    # Start running episodes
    env.after(100, run_episodes)
    env.mainloop()
