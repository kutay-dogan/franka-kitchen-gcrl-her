import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gymnasium_robotics
import gymnasium as gym
import torch
import os
from tqdm import tqdm
from utils.checkpoint import load_checkpoint
from utils.agent import DDPGAgent
from utils.functions import flatten_info, ohe_goals

ENV_NAME = "FrankaKitchen-v1"
TASKS = [
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "top burner",
    "bottom burner",
]
OHE_LOOKUP = ohe_goals(TASKS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists("plots/"):
    os.makedirs("plots/")

agent = DDPGAgent(59, 9, 7, 17, DEVICE)

# Load weights (replay buffer not needed for eval)

load_checkpoint(
    resume_path="checkpoints/checkpoint_3500.pth",
    agent=agent,
    load_replay_buffer=False,
    verbose=False,
)
STEPS = 100
agent.actor.eval()
eval_env = gym.make(
    ENV_NAME,
    max_episode_steps=STEPS,
    tasks_to_complete=TASKS,
)

# --- EVALUATION LOOP ---
print("Starting Evaluation (Noise=0, Steps=3000, Episodes=100 per task)...")

# Store completion times: { "microwave": [45, 32, 105, ...], ... }
# If a task is never solved in an episode, we store None
completion_times = {task: [] for task in TASKS}

for task_name in TASKS:
    print(f"Evaluating Task: {task_name}")
    target_ohe = OHE_LOOKUP[task_name]

    for ep in range(100):
        raw_state, _ = eval_env.reset()
        state, _, desired_goal = flatten_info(raw_state, TASKS)

        solved_at_step = None

        for step in tqdm(range(STEPS), desc=f"Ep {ep + 1}/100", leave=False):
            # DETERMINISTIC ACTION: noise=0
            action = agent.select_action(
                state, target_ohe, desired_goal, noise=0.0, use_parameter_noise=False
            )

            next_raw_state, reward, terminated, truncated, info = eval_env.step(action)

            if task_name in info["step_task_completions"]:
                solved_at_step = step
                break

            state = flatten_info(next_raw_state, TASKS)[0]

            if terminated or truncated:
                break

        completion_times[task_name].append(solved_at_step)


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
colors = sns.color_palette("husl", 7)

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

x_steps = np.arange(0, 101)

print("\n--- FINAL RESULTS ---")
print(f"{'Task':<15} | {'Final Success %':<15} | {'Avg Steps'}")

for i, task in enumerate(TASKS):
    ax = axes[i]
    times = completion_times[task]

    # Create Binary Matrix: 100 episodes x 100 steps
    # Row j, Col t is 1 if task solved by time t in episode j
    success_over_time = np.zeros((100, 101))

    solved_count = 0
    solved_steps_sum = 0

    for ep_idx, t_solve in enumerate(times):
        if t_solve is not None:
            success_over_time[ep_idx, t_solve:] = 1.0  # Stays solved
            solved_count += 1
            solved_steps_sum += t_solve

    # Calculate Mean Success Rate at each step t (0 to 100)
    mean_success_curve = np.mean(success_over_time, axis=0) * 100  # Convert to %

    # Plot
    ax.plot(x_steps, mean_success_curve, color=colors[i], linewidth=2.5)
    ax.fill_between(x_steps, mean_success_curve, color=colors[i], alpha=0.2)

    # Metrics for Title/Table
    final_rate = mean_success_curve[-1]
    avg_steps = (solved_steps_sum / solved_count) if solved_count > 0 else 0

    # Styling
    ax.set_title(f"{task.title()}\nSuccess: {final_rate:.1f}%", fontweight="bold")
    ax.set_xlabel("Episode Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # Print stats
    print(f"{task:<15} | {final_rate:<15.1f} | {avg_steps:.1f}")

# Hide 8th subplot
axes[7].axis("off")

plt.suptitle(
    "Evaluation: Success Rate Evolution over 100 Steps\n(Deterministic Policy, 100 Runs/Task)",
    fontsize=16,
    y=0.98,
)
plt.tight_layout()
plt.savefig("plots/evaluation_success_time_cdf.png", dpi=300)
plt.show()
