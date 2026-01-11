import torch
import numpy as np
import gymnasium_robotics
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
import random
import os
from utils.functions import ohe_goals, flatten_info, add_human_demonstrations
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.agent import DDPGAgent
from utils.replay_buffer import ReplayBuffer

# DETERMINISM SETUP
SEED = 42
task_rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)


# ENV AND TRAINING PARAMETERS
ENV_NAME = "FrankaKitchen-v1"
MAX_EPISODES = 20000
MAX_STEPS = 1000
BATCH_SIZE = 512
START_TRAINING_AFTER = 3000
NOISE = 0.05
K = 1
HUMAN_DATA_FREQ = 500
CHECKPOINT_FREQ = 500
PLOT_FREQ = 1
RESUME_PATH = "checkpoints/checkpoint_1500.pth"
CHECKPOINT_DIR = "checkpoints"
PARAMETER_NOISE = False
TASKS = [
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "top burner",
    "bottom burner",
]

# creating ohe lookup table, which is crucial for the agent!
# agent should understand what is the aimed task of the (s,a,r) tuple
# therefore, we give current task info with OHE vector as a feature
# so that the agent can distinguish which actions should be taken for a specific task.
OHE_LOOKUP = ohe_goals(TASKS)
# what about relabled "fake" tasks in HER mechanism?
# we simply use a zero vector.

# CREATING ENVIRONMENT AND INITIALIZING AGENT
env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, tasks_to_complete=TASKS)
env.action_space.seed(SEED)
obs_dict = env.reset()[0]
state_dim = obs_dict["observation"].shape[0]
action_dim = env.action_space.shape[0]
sample_goal = np.concatenate([obs_dict["desired_goal"][k] for k in TASKS])
goal_dim = sample_goal.shape[0]
ohe_dim = len(TASKS)

device = "cuda"

agent = DDPGAgent(state_dim, action_dim, ohe_dim, goal_dim, device)

task_history = {task: [0] for task in TASKS}
cumulative_victories = {task: 0 for task in TASKS}
loss_history = {"critic_loss": [], "actor_loss": []}
q_value_history = []
efficiency_history = {task: [] for task in TASKS}

start_episode = 1
total_steps = 0

plot_display = display.display(plt.figure(figsize=(16, 12)), display_id=True)

# If RESUME_PATH exist, checkpoint is loaded 
(
    start_episode,
    task_history,
    cumulative_victories,
    loss_history,
    q_value_history,
    efficiency_history,
    total_steps,
    replay_buffer,
) = load_checkpoint(CHECKPOINT_DIR, RESUME_PATH, agent, load_replay_buffer=False)

# IMPORTANT
# comment the replay buffer line below if you want to use saved replay buffer!
replay_buffer = ReplayBuffer(
    state_dim, action_dim, ohe_dim, goal_dim, device, max_size=2_000_000, seed=SEED
)

# for HER, we will use this zero vector instead of OHE
# because HER is not task specific.
generic_task_ohe = np.zeros(ohe_dim)

print(f"State Dim: {state_dim}, Goal Dim: {goal_dim}, OHE Dim: {ohe_dim}")
print(f"Running On Device: {device}")

add_human_demonstrations(replay_buffer, OHE_LOOKUP, TASKS)

for episode in range(start_episode, MAX_EPISODES):
    raw_state, _ = env.reset(seed=SEED)
    state, achieved_goal, desired_goal = flatten_info(raw_state, TASKS)

    episode_cache = []
    victory_map = {}
    current_start_index = 0
    undone_tasks = list(TASKS)

    ep_critic_losses = []
    ep_actor_losses = []
    ep_q_values = []

    current_task_name = task_rng.choice(undone_tasks)
    current_ohe = OHE_LOOKUP[current_task_name]
    
    for step in range(MAX_STEPS):
        action = agent.select_action(
            state,
            current_ohe,
            desired_goal,
            noise=NOISE,
            use_parameter_noise=PARAMETER_NOISE,
        )

        next_raw_state, reward, terminated, truncated, info = env.step(action)
        next_state, next_achieved, next_desired = flatten_info(next_raw_state, TASKS)

        # we add all steps into episode cache
        # episode cache is also used for HER later.
        episode_cache.append(
            {
                "state": state,
                "action": action,
                "next_state": next_state,
                "achieved_goal": next_achieved,
                "desired_goal": next_desired,
            }
        )

        # checking if "ANY" task is completed at current step.
        # because agent can complete other tasks too.
        if info["step_task_completions"]:
            for task_name in info["step_task_completions"]:
                # it's unlikely, impossible, to complete more than 1 task in just 1 step
                # but we are using for loop just in case.
                if task_name not in victory_map:
                    victory_map[task_name] = step

                    duration = step - current_start_index
                    efficiency_history[task_name].append(duration)

                    # agent can solve a different task!
                    # hence, we will pretend the agent was aimed for that goal
                    # starting from step = current start index
                    # we obtain the solved task's OHE vector for relabeling
                    solved_ohe = OHE_LOOKUP[task_name]

                    for i in range(current_start_index, step + 1):
                        data = episode_cache[i]
                        
                        # reward is given at the end.
                        # other states' reward are zero 
                        is_finish = i == step
                        
                        replay_buffer.add(
                            data["state"],
                            data["action"],
                            solved_ohe,
                            data["desired_goal"],
                            data["next_state"],
                            reward=1 if is_finish else 0,
                            done=1 if is_finish else 0,
                        )
                        # we keep the done info for the q_value update!
                        # please check utils/agent.py

                    current_start_index = step + 1
                    # if agent will complete another task after this step
                    # we should treat current step as the beggining for that task for adding into buffer
                    # hence, we update the current start index variable with step+1.
                    if task_name in undone_tasks:
                        undone_tasks.remove(task_name)

                    if undone_tasks:
                        current_task_name = task_rng.choice(undone_tasks)
                        current_ohe = OHE_LOOKUP[current_task_name]

            if len(info["episode_task_completions"]) == len(TASKS):
                # if all tasks are completed...
                break

        state = next_state
        total_steps += 1
        
        # training starts after observing "START_TRAINING_AFTER" steps.
        if total_steps > START_TRAINING_AFTER:
            for _ in range(K):
                c_loss, a_loss, q_val = agent.train(replay_buffer, BATCH_SIZE)
                ep_critic_losses.append(c_loss)
                ep_actor_losses.append(a_loss)
                ep_q_values.append(q_val)

        if terminated or truncated:
            break

    if current_start_index < len(episode_cache):
        final_achieved_goal = episode_cache[-1]["achieved_goal"]

        for i in range(current_start_index, len(episode_cache)):
            data = episode_cache[i]
            is_last = i == len(episode_cache) - 1

            # HER mechanism.
            # we treat the last step as the desired goal
            # adding "fake" labels into buffer
            # hoping that agent will learn environment physics.
            replay_buffer.add(
                data["state"],
                data["action"],
                generic_task_ohe,
                final_achieved_goal,
                data["next_state"],
                reward=1 if is_last else 0,
                done=1 if is_last else 0,
            )

            # adding "real" data, no relabeling here.
            replay_buffer.add(
                data["state"],
                data["action"],
                current_ohe,
                data["desired_goal"],
                data["next_state"],
                reward=0,
                done=0,
            )

    for task in TASKS:
        if task in victory_map:
            cumulative_victories[task] += 1
        task_history[task].append(cumulative_victories[task])

    # PLOT
    if ep_critic_losses:
        loss_history["critic_loss"].append(np.mean(ep_critic_losses))
        loss_history["actor_loss"].append(np.mean(ep_actor_losses))
        q_value_history.append(np.mean(ep_q_values))
    else:
        loss_history["critic_loss"].append(0)
        loss_history["actor_loss"].append(0)
        q_value_history.append(0)

    if episode % HUMAN_DATA_FREQ == 0:
        add_human_demonstrations(replay_buffer, OHE_LOOKUP, TASKS)

    if episode % PLOT_FREQ == 0:
        plt.clf()

        for i, task in enumerate(TASKS):
            plt.subplot(3, 4, i + 1)
            plt.plot(task_history[task], label="cumulative reward", color="blue")
            plt.title(f"{task} cumulative reward")
            plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 8)
        plt.plot(q_value_history, color="purple")
        plt.title("Avg Q-Value")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 9)
        plt.plot(loss_history["critic_loss"][int(episode / 2) :], color="red")
        plt.title("Critic Loss (last 50%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 10)
        plt.plot(loss_history["critic_loss"], color="red")
        plt.title("Critic Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 11)
        plt.plot(loss_history["actor_loss"][int(episode / 2) :], color="green")
        plt.title("Actor Loss (last 50%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 12)
        plt.plot(loss_history["actor_loss"], color="green")
        plt.title("Actor Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_display.update(plt.gcf())
    
    save_checkpoint(
        CHECKPOINT_FREQ,
        episode,
        agent,
        task_history,
        cumulative_victories,
        loss_history,
        efficiency_history,
        total_steps,
        replay_buffer,
        q_value_history,
        CHECKPOINT_DIR,
    )

    c_loss_print = (
        round(loss_history["critic_loss"][-1], 5) if loss_history["critic_loss"] else 0
    )
    print(
        f"Ep {episode}: Solved {list(victory_map.keys())} | Loss: {c_loss_print} | Buffer: {replay_buffer.size} | Unsolved: {current_task_name}"
    )
