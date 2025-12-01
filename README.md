# Computation Offloading Optimization in MEC based on DQN

This repository is a fork of [WangYichi1/Computation-offloading-based-on-DQN](https://github.com/WangYichi1/Computation-offloading-based-on-DQN). 

It focuses on optimizing **Task Offloading** in Mobile Edge Computing (MEC) environments using **Deep Reinforcement Learning (DRL)**. We conducted three progressive experiments to analyze performance changes based on episode length, neural network capacity, and algorithm architecture (Dueling DQN).

## 📌 Project Overview

In MEC environments, mobile devices must decide whether to process tasks locally or offload them to edge servers to minimize delay and energy consumption. This project implements a **Deep Q-Network (DQN)** agent to learn the optimal offloading policy in a dynamic environment.

* **Goal:** Minimize task processing delay (maximize reward).
* **Environment:** Dynamic wireless channels, varying task sizes, and multiple MEC servers.
* **Agent:** DQN Agent (epsilon-greedy policy).

## 📂 File Structure

* `main.py`: Main execution file for training the agent.
* `dqn.py`: Implementation of the DQN Agent and Neural Network architecture.
* `env.py`: Definition of the MEC environment (State, Action, Reward, State Transition).
* `constants.py`: System parameters (CPU frequency, Bandwidth, etc.).
* `func.py`: Helper functions for action space mapping.

---

## 🧪 Experiments & Results

We performed three distinct experiments to improve the agent's learning efficiency and final performance.

### Base Code
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/b577fa65-6511-4c36-953f-3e7592382cf1" />
<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/7482b679-8ca5-43d7-81a3-90bda02e0130" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/ec8f104d-c22e-441f-a0c5-02421ad64689" />
<img width="640" height="480" alt="Figure_4" src="https://github.com/user-attachments/assets/958891a4-1bf6-45a2-b71b-70a12790ad9a" />

Experimental results confirm that the DQN agent significantly outperforms the traditional Q-Learning baseline. Specifically, DQN achieved shorter task processing latency and maximized cumulative rewards.

### Experiment 1: Baseline & Episode Expansion
**Objective:** Verify the basic learning capability of the base code and observe long-term convergence.

* **Settings:**
    * Used the original network structure (Input → 50 → 30 → Output).
    * Increased training episodes from **400 to over 2000**.
    * `dqn.learn()` is triggered every 1000 steps.
* **Results:**
    * **Initial Phase:** High failure rate (Reward -1.0) due to random exploration.
    * **Learning Phase:** Significant performance jumps observed every 1000 steps.
    * **Conclusion:** The agent successfully learns to avoid timeouts and maximize rewards given sufficient training time.
 
      <img width="1706" height="856" alt="Figure_1" src="https://github.com/user-attachments/assets/dcc1a183-1c1d-45ff-a3f5-6e7b37327cf2" />
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/d2706942-5cf9-4904-a2b1-4472fbfdc204" />
<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/01f1becd-8ec9-4da5-a3d8-8f05c54993e8" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/ad2802dd-98c7-4902-a660-12afcfe1dabb" />
<img width="640" height="480" alt="Figure_4" src="https://github.com/user-attachments/assets/d8df82e3-51b3-4e93-95ad-04f0fd01a324" />


### Experiment 2: Neural Network Capacity Expansion
**Objective:** Enhance the model's ability to represent complex environmental features by deepening and widening the network.

* **Modifications:**
    * **Wider Layers:** Increased nodes from 50/30 to **128/128/64**.
    * **Deeper Network:** Increased hidden layers from 2 to **3**.
* **Results:**
    * **Initial Phase:** Higher failure rate compared to Exp 1. The increased number of parameters required more time to tune from random initialization.
    * **Final Phase:** Achieved **higher success rates and precision** after convergence.
    * **Conclusion:** A larger capacity allows the agent to capture finer details of the environment, leading to better long-term performance despite a slower start.
 
      <img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/3e064fe4-9d8f-4972-a254-7f7a30b3b751" />


### Experiment 3: Dueling DQN Architecture
**Objective:** Accelerate convergence and improve stability by decoupling state value and action advantage.

* **Modifications:**
    * Implemented **Dueling Network Architecture** (Wang et al., 2016).
    * Split the final layer into two streams:
        1.  **Value Stream $V(s)$:** Estimates how good the current state is.
        2.  **Advantage Stream $A(s,a)$:** Estimates the relative importance of each action.
    * **Aggregation:** $Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))$
* **Results:**
    * **Fast Convergence:** The agent quickly identified valuable states without needing to explore all actions.
    * **Robustness:** Maintained high rewards (near 1.0) with minimal fluctuation after convergence.
    * **Conclusion:** Dueling DQN significantly outperformed the standard DQN in learning speed and stability by isolating state values from action choices.

<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/5c4f63fe-ece7-479d-b92e-3492334a9d2e" />


---

## ⚙️ System Model

### MDP Formulation
* **State ($S$):** 6-dimensional vector including Task Size, CPU States of 3 MEC servers, and Channel Gains.
* **Action ($A$):** 66 discrete actions representing a combination of:
    1.  Offloading Ratio (0.0 ~ 1.0)
    2.  Target Server Selection (MEC 1, 2, 3)
    3.  Transmission Power / Channel Selection
* **Reward ($R$):** Based on total delay ($t$).
    * If $t > \text{deadline}$: Reward = -1 (Failure).
    * Else: Reward = $e^{-10t}$ (Higher reward for faster processing).

---

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/DaeTaChulDong/Computation-offloading-based-on-DQN.git](https://github.com/DaeTaChulDong/Computation-offloading-based-on-DQN.git)
    cd Computation-offloading-based-on-DQN
    ```

2.  **Install dependencies**
    ```bash
    pip install torch numpy matplotlib gym
    ```

3.  **Run the simulation**
    ```bash
    python main.py
    ```

## 📚 References

* **Original Repository:** [WangYichi1/Computation-offloading-based-on-DQN](https://github.com/WangYichi1/Computation-offloading-based-on-DQN)
* **Paper:** Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML, 2016.
