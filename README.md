# Autonomous Robot Path Planning using Reinforcement Learning

This project presents an autonomous robotic navigation system built using reinforcement learning algorithms—Q-Learning and SARSA—to solve path planning problems in a grid-based environment. The system enables an agent to learn optimal navigation strategies through interaction with its environment, balancing efficiency and safety.

---

## Overview

Path planning is a core challenge in robotics and artificial intelligence. This project demonstrates how reinforcement learning can be applied to enable an agent to autonomously discover optimal paths while avoiding obstacles and minimizing cost.

The agent continuously improves its policy by learning from rewards and penalties, leading to increasingly efficient navigation over time.

---

## Methodology

### Environment

* Two-dimensional grid-based simulation
* Configurable grid size and layout
* Static obstacles to simulate real-world constraints
* Defined goal state representing the destination

---

### Algorithms

#### Q-Learning (Off-Policy Learning)

* Learns the optimal policy independent of the agent’s current behavior
* Updates values using the maximum expected future reward
* Typically converges faster
* May produce aggressive or risk-prone paths

---

#### SARSA (On-Policy Learning)

* Learns based on the actual actions taken by the agent
* Updates values using the current policy
* More stable and conservative learning
* Produces safer navigation behavior

---

## Results and Observations

* Progressive reduction in steps required to reach the goal
* Increase in cumulative reward over training episodes
* Q-Learning demonstrates faster convergence
* SARSA results in safer and more stable path selection
* Clear trade-off observed between optimality and safety

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Tkinter
* Pillow

---

## Project Structure

```text
RL_Q-Learning_E1/
RL_Sarsa_E1/
Q-Learning/
Sarsa/
```

Each directory contains an independent implementation of the respective algorithm along with execution scripts.

---

## Usage

### Run Q-Learning (Experimental Setup)

```bash
cd RL_Q-Learning_E1
python run_agent.py
```

### Run SARSA (Experimental Setup)

```bash
cd RL_Sarsa_E1
python run_agent.py
```

### Run Standard Q-Learning

```bash
cd Q-Learning
python run_agent.py
```

### Run Standard SARSA

```bash
cd Sarsa
python run_agent.py
```

---

## Key Highlights

* Implementation of both off-policy and on-policy reinforcement learning methods
* Comparative analysis of learning behavior and performance
* Modular and extensible environment design
* Visualization of agent learning and path optimization

---

## Applications

* Autonomous robotics navigation
* Path optimization in constrained environments
* Game AI and simulation systems
* Smart transportation and logistics

---

## Author

Somiya Namdeo
