import torch
import os
# FUNCTIONS FOR THE CHECKPOINT MECHANISM


def save_checkpoint(
    checkpoint_frequency,
    episode,
    agent,
    task_history,
    cumulative_victories,
    loss_history,
    efficiency_history,
    total_steps,
    replay_buffer,
    q_value_history,
    checkpoint_dir,
    enabled: bool = True,
    verbose: bool = True,
):
    if enabled and episode % checkpoint_frequency == 0:

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_data = {
            "episode": episode,
            "actor_state_dict": agent.actor.state_dict(),
            "actor_target_state_dict": agent.actor_target.state_dict(),
            "actor_pertrubed_state_dict": agent.perturbed_actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "critic_target_state_dict": agent.critic_target.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            "task_history": task_history,
            "cumulative_reward": cumulative_victories,
            "loss_history": loss_history,
            "efficiency_history": efficiency_history,
            "total_steps": total_steps,
            "replay_buffer": replay_buffer,
            "q_value_history": q_value_history,
        }

        save_filename = f"{checkpoint_dir}/checkpoint_{episode}.pth"
        torch.save(checkpoint_data, save_filename)
        if verbose:
            print(f"--> Checkpoint saved: {save_filename}")


def load_checkpoint(resume_path, agent, verbose: bool = True):
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, weights_only=False)

        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        agent.perturbed_actor.load_state_dict(checkpoint["actor_pertrubed_state_dict"])

        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )

        task_history = checkpoint["task_history"]
        cumulative_victories = checkpoint["cumulative_reward"]

        loss_history = checkpoint["loss_history"]
        q_value_history = checkpoint["q_value_history"]
        efficiency_history = checkpoint["efficiency_history"]

        total_steps = checkpoint["total_steps"]
        start_episode = checkpoint["episode"] + 1
        replay_buffer = checkpoint["replay_buffer"]

        if verbose:
            print(f"Resumed successfully from Episode {start_episode}")

        return (
            start_episode,
            task_history,
            cumulative_victories,
            loss_history,
            q_value_history,
            efficiency_history,
            total_steps,
            replay_buffer,
        )
