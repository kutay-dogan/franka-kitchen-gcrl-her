import numpy as np
import os
import gymnasium_robotics
import matplotlib.pyplot as plt
import seaborn as sns
from utils.checkpoint import load_checkpoint
from utils.agent import DDPGAgent
from utils.functions import ohe_goals

device = "cuda"

agent = DDPGAgent(59, 9, 7, 17, device)

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

if not os.path.exists("plots/"):
    os.makedirs("plots/")

(
    start_episode,
    task_history,
    cumulative_victories,
    loss_history,
    q_value_history,
    efficiency_history,
    total_steps,
    _,  # no replay buffer
) = load_checkpoint(
    resume_path="checkpoints/checkpoint_3500.pth", agent=agent, load_replay_buffer=False, verbose=False
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
colors = sns.color_palette("husl", 7)

tasks = [
    "microwave", "kettle", "light switch", "slide cabinet", 
    "hinge cabinet", "top burner", "bottom burner"
]

# --- Figure 1: Cumulative Success (7 Subplots) ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, task in enumerate(tasks):
    ax = axes[i]
    if task in task_history:
        # Data
        steps = range(len(task_history[task]))
        data = task_history[task]
        
        # Plot
        ax.plot(steps, data, color=colors[i], linewidth=2)
        ax.fill_between(steps, data, color=colors[i], alpha=0.3)
        
        # Styling
        ax.set_title(task.replace(" ", " ").title(), fontweight='bold')
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Total Solved")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')

# Turn off the 8th empty subplot
axes[7].axis('off')

plt.suptitle("Cumulative Task Success Over 3,500 Episodes", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("plots/cumulative_success_subplots.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Figure 2: Efficiency (7 Subplots) ---
def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, task in enumerate(tasks):
    ax = axes[i]
    if task in efficiency_history and len(efficiency_history[task]) > 10:
        # Data
        raw_data = efficiency_history[task]
        smoothed = moving_average(raw_data)
        
        # Plot
        ax.plot(smoothed, color=colors[i], linewidth=2)
        
        # Styling
        ax.set_title(task.title(), fontweight='bold')
        ax.set_xlabel("Successful Trials")
        ax.set_ylabel("Steps to Solve")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')

# Turn off the 8th empty subplot
axes[7].axis('off')

plt.suptitle("Efficiency Improvement: Reduction in Steps to Solve", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("plots/efficiency_subplots.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Figure 3: Training Stability ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(loss_history["critic_loss"], color="tab:red", alpha=0.7, linewidth=1)
axes[0].set_title("Critic Loss (MSE)")
axes[0].set_xlabel("Updates")
axes[0].set_yscale("log")

axes[1].plot(loss_history["actor_loss"], color="tab:green", alpha=0.7, linewidth=1)
axes[1].set_title("Actor Loss")
axes[1].set_xlabel("Updates")

axes[2].plot(q_value_history, color="tab:purple", alpha=0.7, linewidth=1)
axes[2].set_title("Average Q-Value Estimate")
axes[2].set_xlabel("Updates")

plt.tight_layout()
plt.savefig("plots/training_stability.png", dpi=300)
plt.show()

