# 🚀 Autonomous Robot Path Planning using Reinforcement Learning (Q-Learning & SARSA)

This project implements an intelligent robotic navigation system using **Reinforcement Learning (RL)** algorithms—**Q-Learning** and **SARSA**—to solve path planning problems in a grid-based environment.

The agent learns to navigate efficiently, avoid obstacles, and reach a goal through continuous interaction with the environment.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Environment](#environment)
  - [Algorithms](#algorithms)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Usage](#usage)

---

## 📌 Overview

Path planning is a fundamental problem in robotics and artificial intelligence. This project demonstrates how reinforcement learning can be used to train an autonomous agent to discover optimal paths in a structured environment.

The agent improves its performance over time by learning from rewards and penalties associated with its actions.

---

## 🧠 Methodology

### 🔹 Environment

- 2D grid-based simulation  
- Customizable grid size  
- Static obstacles placed within the grid  
- Goal location defined as the target state  

---

### 🔹 Algorithms

#### 🟢 Q-Learning (Off-Policy)

- Learns optimal policy independent of current actions  
- Uses maximum future reward for updates  
- Faster convergence but may choose risky paths  

---

#### 🔵 SARSA (On-Policy)

- Learns based on the current action policy  
- Updates values using actual actions taken  
- Safer and more stable learning  

---

## 📊 Results

- Decrease in number of steps over episodes  
- Increase in cumulative reward  
- Q-Learning shows faster learning  
- SARSA demonstrates safer navigation behavior  

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Tkinter  
- Pillow  

---

## ▶Usage

### Run Different algorithms (Both basic and advanced)
```bash
cd RL_Q-Learning_E1
python run_agent.py

cd ../RL_Sarsa_E1
python run_agent.py

cd ../Q-Learning
python run_agent.py

cd ../Sarsa
python run_agent.py
