# Franka Kitchen GCRL-HER

This repository contains a PyTorch implementation of Goal-Conditioned Reinforcement Learning (GCRL) combined with Hindsight Experience Replay (HER) for the Franka Kitchen environment. The project aims to solve sparse-reward manipulation tasks using a robotic arm.

## Project Architecture

The implementation uses an off-policy Actor-Critic approach with the following components:

* **Algorithm:** DDPG (Deep Deterministic Policy Gradient) with parameter noise for exploration.
* **Hindsight Experience Replay (HER):** Relabeling failed trajectories as successful ones by treating the achieved state as the goal, improving sample efficiency.
* **Networks:**
    * **Actor:** Maps state, goal, and task info to actions.
    * **Critic:** Estimates Q-values for state-action-goal tuples.
* **Environment:** `FrankaKitchen-v1` via Gymnasium Robotics.

## Environment Management (Pixi)

This project strictly uses **Pixi** for package management. This ensures that all dependencies, including system-level libraries like MuJoCo and specific PyTorch versions (v2.9.0), are reproducible across different machines.

### Prerequisites

Install Pixi globally:

  [Pixi Installation Guide](https://pixi.prefix.dev/latest/installation/)
  
You can also check the dependencies on `pixi.toml` file and install manually without Pixi.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kutay-dogan/franka-kitchen-gcrl-her.git
   ```

2. Initialize the environment and install dependencies:
   ```bash
   pixi install
   ```
   This reads `pixi.toml` and sets up the isolated Python environment.

## Data Preparation

To accelerate training, you can download human demonstration datasets. This script fetches the data from HuggingFace and places it in `./my_kitchen_data`.

```bash
python download-human-data.py
```


## Usage
To ensure all dependencies are correctly loaded, execute commands within the Pixi environment by prefixing them with `pixi run` (e.g., `pixi run python train_agent.py`) 
or by entering the shell via `pixi shell`. (e.g.,`pixi shell`, `python train_agent.py`) 

### 1. Training
Run the main training loop. This script handles environment interaction, HER relabeling, and checkpointing.

```bash
python train_agent.py
```
* **Determinism:** Seeds are set (default: 42) for reproducibility.
* **Logging:** Live plots of loss and success rates are updated during training.
* **Checkpoints:** Models are saved every 500 episodes to the `checkpoints/` folder.

### 2. Evaluation
Evaluate a trained policy deterministically on all tasks.

```bash
python evaluate-agent.py
```
This runs 100 episodes per task and saves success rate plots to `plots/`.

### 3. Visualizing Results
Generate comprehensive plots for training metrics (Actor/Critic loss, Q-values, and efficiency/steps to solve).

```bash
python create-results.py
```


### 4. Storage Management
You can remove the replay buffer from a checkpoint file (irreversible).

```bash
python delete-replay-buffer.py
```

### 5. Rendering
Visualize the agent's behavior in real-time. This script runs the evaluation loop with `render_mode="human"`, allowing you to watch the robot attempt specific tasks sequentially.

```bash
pixi run python render-robot.py
```

## Code Structure

* `pixi.toml`: Dependency file for Pixi.
* `train_agent.py`: Main training script.
* `agent.py`: DDPG Agent class.
* `actor_critic.py`: Neural network definitions.
* `replay_buffer.py`: Replay buffer class.
* `functions.py`: Utilities for One-Hot Encoding and data loading.
* `checkpoint.py`: Save/Load logic.

## License

* **Dependencies:** This project relies on `gymnasium-robotics` and the MuJoCo physics engine.
* **Data:** Human demonstration datasets are sourced from the Hugging Face repository `robertcowher/farama-kitchen-sac-hrl-youtube`.
