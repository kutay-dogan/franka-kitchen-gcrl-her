import numpy as np
# HELPER FUNCTIONS

def ohe_goals(tasks):
    dim = len(tasks)
    lookup = {}
    for i, task in enumerate(tasks):
        vec = np.zeros(dim, dtype=np.float32)
        vec[i] = 1.0
        lookup[task] = vec
    return lookup

def flatten_info(state_dict, tasks):
    obs = state_dict['observation']
    achieved = np.concatenate([state_dict["achieved_goal"][k] for k in tasks])
    desired = np.concatenate([state_dict["desired_goal"][k] for k in tasks])
    return obs, achieved, desired

def compute_reward(achieved_vec, desired_vec, threshold=0.01, actual_length=7):
    dist = np.linalg.norm(achieved_vec[:actual_length] - desired_vec[:actual_length])
    return 1.0 if dist <= threshold else 0.0