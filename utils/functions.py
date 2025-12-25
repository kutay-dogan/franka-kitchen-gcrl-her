import numpy as np
# HELPER FUNCTIONS

def encode_all_goals(goal_dict, tasks, max_goal_length):

    encoded_goals = []
    num_tasks = len(tasks)

    for i, task_name in enumerate(tasks):
        achieved_goal = goal_dict[task_name]
        goal_vec = np.zeros(max_goal_length)
        goal_vec[:len(achieved_goal)] = achieved_goal

        id_vec = np.zeros(num_tasks)
        id_vec[i] = 1.0

        vector = np.concatenate([goal_vec, id_vec])
        encoded_goals.append(vector)

    return encoded_goals

def get_max_goal_length(achieved_goals):
    max_goal_length = 0

    for i in achieved_goals.values():
        if max_goal_length < len(i):
            max_goal_length = len(i)
    return max_goal_length

def compute_reward(achieved_vec, desired_vec, threshold=0.01, actual_length=7):
    dist = np.linalg.norm(achieved_vec[:actual_length] - desired_vec[:actual_length])
    return 1.0 if dist <= threshold else 0.0