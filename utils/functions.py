import numpy as np
import glob
# HELPER FUNCTIONS


def ohe_goals(tasks):
    # function for creating ohe lookup table

    dim = len(tasks)
    lookup = {}
    for i, task in enumerate(tasks):
        vec = np.zeros(dim, dtype=np.float32)
        vec[i] = 1.0
        lookup[task] = vec
    return lookup


def flatten_info(state_dict, tasks):
    # function for concatenating achieved and desired goal vectors into a single vector

    obs = state_dict["observation"]
    achieved = np.concatenate([state_dict["achieved_goal"][k] for k in tasks])
    desired = np.concatenate([state_dict["desired_goal"][k] for k in tasks])
    return obs, achieved, desired


def add_human_demonstrations(
    replay_buffer, ohe_lookup, tasks, desired_goal:dict=None, dir: str = "./my_kitchen_data"
):
    # function for adding human demonstration data into the replay buffer (batched)
    # accepts only .npz format, task name should be included in the file name starting with "human_memory"

    if not desired_goal:
        desired_goal = {
            "kettle": [-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06],
            "microwave": [-0.75],
            "hinge cabinet": [0.0, 1.45],
            "bottom burner": [-0.88, -0.01],
            "top burner": [-0.92, -0.01],
            "light switch": [-0.69, -0.05],
            "slide cabinet": [0.37],
        }

    desired_goal = np.concatenate([desired_goal[k] for k in tasks])

    paths = [_ for _ in glob.glob(f"{dir}/human_memory*.npz")]

    for path in paths:
        data = np.load(path)

        for task in ohe_lookup:
            if task in path.replace("_", " "):
                break

        batch_size = len(data["state"])
        ohe = ohe_lookup[task]

        # duplicating ohe and goal vector wrt batch size
        ohe_batch = np.tile(ohe, (batch_size, 1))
        goal_batch = np.tile(desired_goal, (batch_size, 1))

        print(f"adding human data for task: {task}, batch_size: {batch_size}")
        replay_buffer.add_batch(
            batch_size,
            data["state"],
            data["action"],
            ohe_batch,
            goal_batch,
            data["next_state"],
            data["reward"].reshape(batch_size, 1),
            data["reward"].reshape(batch_size, 1),
        )
