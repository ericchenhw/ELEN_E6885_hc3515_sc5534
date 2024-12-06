# RTB Optimization using Reinforcement Learning

This project implements a reinforcement learning framework for Real-Time Bidding (RTB) optimization using Proximal Policy Optimization (PPO).

## Code Structure

- **`bidding.py`**: Main implementation of the RTB optimization framework.
  - Defines the auction environment (`BidGym`) to simulate bidding and rewards.
  - Implements the PPO algorithm for training a continuous-action RL agent.
  - Contains utilities for data normalization, reward scaling, and actor-critic neural networks.

## Code Flow

1. **Data Loading and Preprocessing**:
   - `get_data()` function loads data from `imp.csv`, splits it into training and testing datasets, and prepares auction histories.
2. **Environment Initialization**:
   - `BidGym` simulates the auction process using bidding history and market dynamics.
3. **Agent Training**:
   - The PPO agent interacts with the environment, updates the policy using collected experiences, and learns optimal bidding strategies.
4. **Evaluation and Testing**:
   - `evaluate_policy()` monitors training performance.
   - `test_policy()` evaluates the trained agent on the test dataset.

## How to Run

## Requirements

To run this project, install the following Python dependencies:

```bash
pip install -r requirements.txt
```

## Steps

1. **Prepare the Dataset**:
   - Download the impression data from [iPinYou RTB Dataset](https://contest.ipinyou.com/).
   - Place the `imp.csv` file in the project directory.

2. **Run the Code**:
   ```bash
   python bidding.py \
    --state_dim 4 \
    --action_dim 12 \
    --batch_size 30 \
    --mini_batch_size 30 \
    --max_train_steps 3000000 \
    --evaluate_freq 100 \
    --policy_dist Beta \
    --max_action 2.0 \
    --hidden_width 1024 \
    --lr_a 0.0003 \
    --lr_c 0.0003 \
    --lamda 1.0 \
    --gamma 1.0 \
    --epsilon 0.2 \
    --K_epochs 10 \
    --entropy_coef 0.001 \
    --set_adam_eps True \
    --budget 5000000 \
    --seed 42
