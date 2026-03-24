import pandas as pd
import matplotlib.pyplot as plt

# Load data
q = pd.read_csv("q_learning.csv")
s = pd.read_csv("sarsa.csv")

# ---------------------------
# Steps Comparison
# ---------------------------
plt.figure()
plt.plot(q["steps"], label="Q-Learning", color="blue")
plt.plot(s["steps"], label="SARSA", color="red")
plt.title("Steps per Episode Comparison")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid()
plt.show()

# ---------------------------
# Cost (Reward) Comparison
# ---------------------------
plt.figure()
plt.plot(q["cost"], label="Q-Learning", color="blue")
plt.plot(s["cost"], label="SARSA", color="red")
plt.title("Reward (Cost) Comparison")
plt.xlabel("Episode")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()

# ---------------------------
# Print Summary
# ---------------------------
print("------ COMPARISON ------")
print("Q-Learning Avg Steps:", q["steps"].mean())
print("SARSA Avg Steps:", s["steps"].mean())

print("Q-Learning Avg Cost:", q["cost"].mean())
print("SARSA Avg Cost:", s["cost"].mean())