# Reinforcement Learning Games – Assignment 1

This repository contains the code and report for **Assignment 1 – Reinforcement Learning**, implementing and analyzing RL agents for two classic environments:

- Snake
- Tic-Tac-Toe

The work is accompanied by a detailed report:
- `Extras/Assignment 1 Reinforcement Learning Report.pdf`

---

## Repository Structure
```text
.
├─ .gitignore
├─ Extras
│  ├─ Assignment 1 Reinforcement Learning Report.docx
│  ├─ Assignment 1 Reinforcement Learning Report.pdf
│  └─ Assignment 1 Reinforcement Learning.pdf
├─ readme.md
├─ snake
│  ├─ configs.py
│  ├─ main.py
│  ├─ snake_play.py
│  └─ snake_train.py
└─ tictactoe
   ├─ configs.py
   ├─ main.py
   ├─ tictactoe_play.py
   └─ tictactoe_train.py

- `Extras/` – Assignment report and related documents.
- `snake/` – RL agent and scripts for the Snake environment.
- `tictactoe/` – RL agent and scripts for the Tic-Tac-Toe environment.

For details on **state representation, feature engineering, and how states are displayed**, see page 5 of the report:
- `Extras/Assignment 1 Reinforcement Learning Report.pdf` (p. 5).

---

## Environments and Agents

### Snake

The Snake implementation uses a reinforcement learning agent (e.g., value-based method with function approximation or Q-table, as described in the report) with a carefully engineered state space.  

Key components (see code for exact interfaces):

- `snake/configs.py`  
  Configuration parameters for training and evaluation such as:
  - Learning rate, discount factor, exploration schedule
  - Network or table sizes (if applicable)
  - Training episodes and logging options

- `snake/snake_train.py`  
  Training script for the Snake agent. Handles:
  - Environment interaction loop
  - Action selection (exploration vs exploitation)
  - Learning updates and saving of model or weights

- `snake/snake_play.py`  
  Script for **playing and visualizing** a trained Snake agent. Uses the trained model and state representation described in the report (see p. 5 for state display and feature engineering details).

- `snake/main.py`  
  Entrypoint / helper to run training or playing, or to integrate configs and modules.

### Tic-Tac-Toe

The Tic-Tac-Toe implementation uses an RL agent that learns an optimal (or near-optimal) policy through self-play or play against a fixed opponent, as discussed in the report.

Key components:

- `tictactoe/configs.py`  
  Configuration of learning parameters and game setup:
  - Learning rate, discount factor
  - Exploration strategy
  - Number of training episodes

- `tictactoe/tictactoe_train.py`  
  Training loop for the Tic-Tac-Toe agent:
  - Game episodes and reward assignment
  - Policy/value updates
  - Model persistence (if applicable)

- `tictactoe/tictactoe_play.py`  
  Script for **playing against the trained agent** or for having the agent play test games. Uses the same state display and feature engineering principles outlined in the report.

- `tictactoe/main.py`  
  Entrypoint / dispatcher for running training, evaluation, or game sessions.

---

## Getting Started

### Prerequisites

- Python 3.x
- Recommended packages (depending on implementation; adjust per your `requirements.txt` if you have one):
  - `numpy`
  - `matplotlib`
  - Any RL or plotting libraries used in the codebase

Install dependencies (example):

bash
pip install -r requirements.txt

(if a `requirements.txt` file exists; otherwise install packages used in the code manually).

---

## Usage

### Snake

#### Train the agent

From the project root:

bash
cd snake
python snake_train.py

This will:
- Initialize the Snake environment
- Train an RL agent for the configured number of episodes
- Save the learned policy/model to disk (see code for exact path/format)

You can adjust training parameters in:

python
# snake/configs.py

#### Play with the trained agent

bash
cd snake
python snake_play.py

This:
- Loads the trained model
- Runs the Snake game with the agent’s actions
- Displays the game and state representation (see report, p. 5, for details on how states/features are visualized)

Alternatively, `snake/main.py` may provide a command-line interface to choose between training and playing:

bash
cd snake
python main.py

(see the file for details).

---

### Tic-Tac-Toe

#### Train the agent

bash
cd tictactoe
python tictactoe_train.py

This will:
- Run multiple episodes of Tic-Tac-Toe
- Update the agent’s value function / policy
- Save the final model or value table

Modify learning settings in:

python
# tictactoe/configs.py

#### Play against the trained agent

bash
cd tictactoe
python tictactoe_play.py

You can:
- Play as a human against the trained agent, or
- Let the agent play test games (depending on implementation in the script)

`main.py` may also serve as a unified entrypoint:

bash
cd tictactoe
python main.py

---

## Report and Documentation

The main write‑up for this assignment is:

- `Extras/Assignment 1 Reinforcement Learning Report.pdf`

It includes:

- Problem formulation and RL setup for Snake and Tic-Tac-Toe
- State representation and **feature engineering** (see especially p. 5, which describes how the game state is displayed and encoded)
- Reward structure and learning algorithm details
- Experimental results and analysis

For further implementation details, consult:
- Inline comments in `configs.py`, `*_train.py`, and `*_play.py`
- The full report for theoretical and design decisions.

---

## Reproducibility

To reproduce results similar to those reported:

1. Ensure configurations in:
   - `snake/configs.py`
   - `tictactoe/configs.py`
   match the hyperparameters used in the report.
2. Run training scripts for the same number of episodes.
3. Use the play scripts to visualize policies and verify qualitative behavior.
