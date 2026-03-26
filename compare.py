import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all data
q = pd.read_csv("Q-Learning/q_learning.csv")
s = pd.read_csv("Sarsa/sarsa.csv")
eq = pd.read_csv("RL_Q-Learning_E1/q_learning.csv")
es = pd.read_csv("RL_SARSA_E1/exp_sarsa.csv")

# ---------------------------
# Steps Comparison
# ---------------------------
plt.figure()
plt.plot(q["steps"].rolling(20).mean(), label="Q-Learning")
plt.plot(s["steps"].rolling(20).mean(), label="SARSA")
plt.plot(eq["steps"].rolling(20).mean(), label="Exp Q-Learning")
plt.plot(es["steps"].rolling(20).mean(), label="Exp SARSA")

plt.title("Steps Comparison (All Algorithms)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid()
plt.savefig("steps_comparison.png")
plt.show()

# ---------------------------
# Cost Comparison
# ---------------------------
plt.figure()
plt.plot(q["cost"].rolling(20).mean(), label="Q-Learning")
plt.plot(s["cost"].rolling(20).mean(), label="SARSA")
plt.plot(eq["cost"].rolling(20).mean(), label="Exp Q-Learning")
plt.plot(es["cost"].rolling(20).mean(), label="Exp SARSA")

plt.title("Cost Comparison (All Algorithms)")
plt.xlabel("Episode")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.savefig("cost_comparison.png")
plt.show()

# ---------------------------
# Summary Metrics
# ---------------------------
print("\n------ FINAL COMPARISON ------")

def summary(name, data):
    avg_steps = data["steps"].mean()
    final_steps = data["steps"].tail(50).mean()
    variance = data["steps"].var()
    avg_cost = data["cost"].mean()

    # Efficiency (% improvement)
    efficiency = ((data["steps"][:100].mean() - data["steps"][-100:].mean()) 
                  / data["steps"][:100].mean()) * 100

    # Accuracy-like score (lower steps = better)
    score = (1 / final_steps) * 100

    print(f"\n{name}:")
    print("Avg Steps:", round(avg_steps, 2))
    print("Final 50 Avg Steps:", round(final_steps, 2))
    print("Stability (Variance):", round(variance, 2))
    print("Avg Cost:", round(avg_cost, 2))
    print("Efficiency (%):", round(efficiency, 2))
    print("Accuracy-like Score:", round(score, 2))

# Run summary
summary("Q-Learning", q)
summary("SARSA", s)
summary("Exp Q-Learning", eq)
summary("Exp SARSA", es)

# ---------------------------
# Best Model Detection
# ---------------------------
models = {
    "Q-Learning": q,
    "SARSA": s,
    "Exp Q-Learning": eq,
    "Exp SARSA": es
}

best_model = min(models.items(), key=lambda x: x[1]["steps"].tail(50).mean())

print("\n🏆 Best Model Based on Convergence:", best_model[0])