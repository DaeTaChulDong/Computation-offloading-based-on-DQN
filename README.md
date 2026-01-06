# Mobility-Aware Computation Offloading in Vehicular Networks using Transformer and DRL

This project implements a **Mobility-Aware Computation Offloading System** for vehicular edge computing (VEC) environments. It proposes a **Transformer-based mobility prediction model** integrated with a Deep Reinforcement Learning (DRL) agent to optimize task offloading decisions. The system is benchmarked against a traditional LSTM-based approach in a complex urban simulation.

## 📌 Overview

Efficient computation offloading in vehicular networks relies heavily on accurate mobility prediction to prevent connection drops during task execution. This project:

1. **Compares** two mobility prediction models: **Baseline (LSTM)** vs. **Proposed (Transformer)**.
2. **Evaluates** offloading performance under three scenarios using a PPO agent and a Rule-Based verifier.
3. **Demonstrates** the superiority of the Transformer approach in complex environments (Manhattan grid) using a **Deep Feature Fusion** mechanism.

---

## 🛠️ Experimental Design

### 1. Mobility Prediction Models

We predict the future trajectory of vehicles based on historical data.

* **Baseline (LSTM):** Recurrent Neural Network architecture standard in existing surveys.
* **Proposed (Transformer):** Self-Attention based architecture optimized for long-term time-series dependencies.
* **Input/Output:** Past 30s trajectory  Future 15s prediction.

### 2. Offloading Scenarios

We evaluate the offloading decision-making process under three conditions:

* **Scenario A (No Prediction):** Decisions are based solely on the current state (Distance, Speed).
* **Scenario B (with LSTM):** Decisions utilize future information predicted by the LSTM model.
* **Scenario C (with Transformer):** Decisions utilize future information predicted by the Transformer model.

---

## ⚙️ Environment Setup

### Prerequisites

* **SUMO (Simulation of Urban MObility):** Version 1.20.0 or higher.
* **Python 3.8+:** Required libraries include `torch`, `pandas`, `numpy`, `traci`.

### Map & Traffic Generation

* **Tool:** OSM Web Wizard (`tools/osmWebWizard.py`).
* **Location:** **Manhattan, NY** (Changed from Sinchon for higher complexity).
* **Traffic:** Dense traffic flow with frequent intersections to simulate realistic urban mobility.
<img width="639" height="378" alt="02 Manhattan" src="https://github.com/user-attachments/assets/40cffc53-2fe2-45f6-991b-93e3b1002d8c" />

---

## 🚀 Usage Guide

### 1. Data Collection

Runs the SUMO simulation to generate the trajectory dataset.

```bash
python data_collector.py

```

* **Operation:** Simulates 3600 steps (1 hour). Records vehicle `(x, y)` coordinates and speed every second.
* **Output:** `mobility_dataset.csv`

### 2. Model Training

Trains both the LSTM and Transformer models using the collected dataset.

```bash
python train_fusion_models.py

```

* **Feature:** Implements **Deep Feature Fusion**, where models output both coordinate predictions and a 64-dim hidden state vector.

### 3. Performance Verification

Evaluates the offloading performance using a precision-based reward system.

```bash
python model_verify_performance.py

```

---

## 🔄 Methodology Evolution (History of Experiments)

To achieve robust results, the experimental methodology was iteratively refined through the following stages:

1. **Data Normalization:** Addressed the issue of large coordinate values (0~1000) preventing model convergence by applying normalization scaling (`1/1000`).
2. **Reward Shaping:** Solved the **Local Optima** problem where the agent refused to offload due to high failure penalties. Adjusted rewards (Success +20, Local -1) to encourage active offloading.
3. **Long-term Prediction (Hard Mode):** Extended the prediction window from 5s to **15s** to clearly differentiate the long-term dependency learning capabilities of LSTM and Transformer.
4. **MGCO Benchmarking (Feature Fusion):** Adopted the "Generative Offloading" concept. The Transformer now passes a **Context Vector (Hidden State)** to the agent, sharing uncertainty and environmental context.
5. **High Complexity & Penalty:**
* **Map:** Switched to **Manhattan** to introduce complex turns.
* **Penalty:** Increased failure penalty to **-100**, creating a "high-risk" environment where inaccurate predictions lead to severe score drops.


6. **Precision Bonus Verification:** Instead of waiting for full PPO convergence, we implemented a **Precision Bonus** metric (up to +50 points) to directly quantify how prediction accuracy translates to Quality of Service (QoS).

---

## 📈 Experimental Results

### 1. Mobility Prediction Accuracy

The Transformer model significantly outperformed the LSTM baseline in minimizing prediction error (MSE).

| Model | Final Loss (Epoch 30) | Improvement |
| --- | --- | --- |
| **Baseline (LSTM)** | 17,362.52 | - |
| **Proposed (Transformer)** | **1,432.27** | **~12x Lower Error** |
<img width="400" height="128" alt="Terminal_LSTM_predictor" src="https://github.com/user-attachments/assets/99efa9c0-2ed7-43e7-bc4d-ef046899f61a" />
<img width="435" height="127" alt="Terminal_Transformer_predictor" src="https://github.com/user-attachments/assets/5a19d8f7-d5ad-4fe7-8242-98dea0ee33ea" />

**Analysis:** The Self-Attention mechanism of the Transformer successfully captured the complex maneuvering patterns of vehicles in the Manhattan grid, whereas LSTM struggled with the high variance.

### 2. Offloading Performance (Cumulative Reward)

Using the Precision Bonus metric, we observed a clear divergence in performance:

* **Scenario A (Gray):** **Lowest Performance.** Without prediction, the agent failed to anticipate connection drops at the edge of coverage, leading to a downward spiral of penalties.
* **Scenario B (Blue):** **Moderate Performance.** LSTM provided decent stability but lacked the precision to maximize the bonus rewards.
* **Scenario C (Red):** **Highest Performance.** The Transformer's high-precision predictions allowed the agent to maximize rewards by maintaining connections only when reliability was guaranteed. The final cumulative score was approximately **15% higher** than Scenario B.
<img width="1024" height="532" alt="good" src="https://github.com/user-attachments/assets/20cbaa6b-8273-4f10-b391-1abd824b3684" />

---

## 📚 References

This project is inspired by and benchmarks the following key studies:

1. **Baseline System Design:**
* **Paper:** *Mobility-Aware Edge Caching and Computing in Vehicle Networks: A Deep Reinforcement Learning Approach*.
* **Usage:** Reference for the initial DRL environment setup and mobility-aware reward estimation logic.


2. **Proposed Methodology (Benchmarked):**
* **Paper:** *MGCO: Mobility-Aware Generative Computation Offloading in Edge-Cloud Systems*.
* **Usage:** Adopted the **Seq2Seq** and **Feature Fusion** concepts. We integrated the Transformer's latent features into the decision-making process to enhance the agent's context awareness, as suggested by the generative offloading framework.
