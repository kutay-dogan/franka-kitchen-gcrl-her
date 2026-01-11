import numpy as np
import gymnasium_robotics
import gymnasium as gym
import torch
import os
import time
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
TARGET_FPS = 15
FRAME_DELAY = 1.0 / TARGET_FPS

if not os.path.exists("plots/"):
    os.makedirs("plots/")

agent = DDPGAgent(59, 9, 7, 17, DEVICE)

load_checkpoint(
    resume_path="checkpoints/checkpoint_3500.pth",
    agent=agent,
    load_replay_buffer=False,
    verbose=False,
)

STEPS = 100
agent.actor.eval()

env = gym.make(
    ENV_NAME,
    render_mode="human",
    max_episode_steps=STEPS,
    tasks_to_complete=TASKS,
    robot_noise_ratio=0,
    object_noise_ratio=0,
)

for task_name in TASKS:
    print(f"\n>>> DEMONSTRATING: {task_name.upper()} <<<")
    
    target_ohe = OHE_LOOKUP[task_name]
    
    raw_state, _ = env.reset()
    state, _, desired_goal = flatten_info(raw_state, TASKS)
    
    for step in range(STEPS):
        start_time = time.time()

        action = agent.select_action(
            state, 
            target_ohe, 
            desired_goal, 
            noise=0.0, 
            use_parameter_noise=False
        )
        
        next_raw_state, reward, terminated, truncated, info = env.step(action)
        state = flatten_info(next_raw_state, TASKS)[0]
        
        if "step_task_completions" in info:
            if task_name in info["step_task_completions"]:
                print(f"Success: {task_name} completed at step {step}!")
                time.sleep(2.0)
                break
        
        if terminated or truncated:
            print("Failed or Time Limit Reached")
            break

        elapsed = time.time() - start_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

print("\nDemonstration Finished.")
env.close()