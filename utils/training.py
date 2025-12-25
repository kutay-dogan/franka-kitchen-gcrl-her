import numpy as np
from utils.functions import encode_all_goals, compute_reward
from IPython import display
import matplotlib.pyplot as plt
import os
import torch


def train_agent(
    env,
    agent,
    replay_buffer,
    tasks: list[str],
    task_lengths,
    max_episodes: int = 5000,
    batch_size: int = 256,
    start_steps: int = 2000,
    her_ratio: int = 4,
    reward_threshold: float = 0.001,
    resume_path: str = None,
):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    task_history = {task: [0] for task in tasks}
    cumulative_reward = {task: 0 for task in tasks}

    plot_display = display.display(plt.figure(figsize=(16, 8)), display_id=True)

    start_episode = 0
    total_steps = 0
    num_tasks = len(tasks)

    if resume_path is not None:
        print(f"--> Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)

        # A. Restore Models
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])

        # B. Restore Optimizers (CRITICAL for smooth resuming)
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )

        # C. Restore Metrics
        task_history = checkpoint["task_history"]
        cumulative_reward = checkpoint["cumulative_reward"]
        total_steps = checkpoint["total_steps"]
        start_episode = checkpoint["episode"] + 1

        print(f"--> Resumed successfully from Episode {start_episode}")
    else:
        print("--> Starting fresh training.")

    for episode in range(start_episode, max_episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict["observation"]

        # Inside your episode or HER loop:
        

        all_goals = encode_all_goals(obs_dict["desired_goal"], tasks, num_tasks)

        active_task_idx = np.random.randint(0, num_tasks)
        desired_goal = all_goals[active_task_idx]

        episode_cache = []
        episode_reward = 0
        done = False

        while not done:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, desired_goal, noise=0.05)

            # 4. Step (Unpack dictionary return)
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_obs = next_obs_dict["observation"]

            # Pre-calculate achieved vectors for this step
            current_step_achieved_vecs = encode_all_goals(
                next_obs_dict["achieved_goal"], tasks, num_tasks
            )

            achieved_vec_at_t = current_step_achieved_vecs[active_task_idx]

            task_name = tasks[active_task_idx]
            actual_len = task_lengths[task_name]


            step_reward = compute_reward(
                achieved_vec_at_t, desired_goal, reward_threshold, actual_length=actual_len)
            
            episode_reward += step_reward

            total_steps += 1

            # Store in Cache
            episode_cache.append(
                {
                    "state": obs,
                    "action": action,
                    "next_state": next_obs,
                    "done": done,
                    # Store the raw dictionaries for HER encoding later
                    "achieved_goal_dict": next_obs_dict["achieved_goal"],
                    "desired_goal_dict": next_obs_dict["desired_goal"],
                    "achieved_vecs": current_step_achieved_vecs,
                }
            )

            obs = next_obs

            # Train
            if total_steps > start_steps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)
        cumulative_reward[tasks[active_task_idx]] += episode_reward
        task_history[tasks[active_task_idx]].append(
            cumulative_reward[tasks[active_task_idx]]
        )

        if (episode + 1) % 500 == 0:
            checkpoint_data = {
                "episode": episode,
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
                "task_history": task_history,
                "cumulative_reward": cumulative_reward,
                "total_steps": total_steps,
            }
            save_filename = f"checkpoints/checkpoint_{episode + 1}.pth"
            torch.save(checkpoint_data, save_filename)
            print(f"--> Checkpoint saved: {save_filename}")

        if episode % 5 == 0:
            plt.clf()
            for i, task in enumerate(task_history):
                plt.subplot(2, 4, i + 1)
                plt.plot(task_history[task], label="cumulative_reward", color="red")
                plt.xlabel("Episode")
                plt.ylabel("Cumulative Reward")
                plt.title(f"Cumulative Rewards - {task}")
                plt.legend()
                plt.grid(True)
            plot_display.update(plt.gcf())

        for t in range(len(episode_cache)):
            state = episode_cache[t]["state"]
            action = episode_cache[t]["action"]
            next_state = episode_cache[t]["next_state"]
            done = episode_cache[t]["done"]

            # --- A. Real Experience ---
            # Get what we actually achieved for the active task
            achieved_vec_at_t = episode_cache[t]["achieved_vecs"][active_task_idx]

            # Calculate Real Reward (1.0 or 0.0)
            r_real = compute_reward(achieved_vec_at_t, desired_goal, reward_threshold, actual_length=actual_len)

            replay_buffer.add(state, action, desired_goal, next_state, r_real, done)

            # --- B. HER (Future Strategy) ---
            for _ in range(her_ratio):
                future_idx = np.random.randint(t, len(episode_cache))

                # Get what we actually achieved in the future for this task
                future_achieved_all = encode_all_goals(
                    episode_cache[future_idx]["achieved_goal_dict"], tasks, num_tasks
                )
                her_goal = future_achieved_all[active_task_idx]

                # Get what we had at time t
                current_achieved_all = encode_all_goals(
                    episode_cache[t]["achieved_goal_dict"], tasks, num_tasks
                )
                achieved_vec_her = current_achieved_all[active_task_idx]

                # Compute replayed reward
                r_her = compute_reward(achieved_vec_her, her_goal, reward_threshold)

                replay_buffer.add(state, action, her_goal, next_state, r_her, done)

        if episode % 10 == 0:
            print(
                f"Episode {episode} | Steps {total_steps} | Buffer {replay_buffer.size}"
            )
