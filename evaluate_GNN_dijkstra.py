from eval_gnn import explore_v2
from str2env import str2env
from config import set_random_seed
from str2name import str2name
import torch
from tqdm import tqdm as tqdm
import numpy as np

def path_cost(path, env):
    path = np.array(path)
    cost = 0
    for i in range(0, len(path) - 1):
        # cost += np.linalg.norm(path[i + 1] - path[i])
        cost += env.distance(path[i], path[i+1])
    return cost


def eval_gnn(str, seed, env, indexes, model=None, model_s=None, use_tqdm=False, smooth=True, batch=500, t_max=1000, k=30,
             **kwargs):
    set_random_seed(seed)
    if model is None:
        _, model, model_path, _, _ = str2name(str)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if model_s is None:
        _, _, _, model_s, model_s_path = str2name(str)
        model_s.load_state_dict(torch.load(model_s_path, map_location=torch.device("cpu")))

    solutions = []
    paths = []
    smooth_paths = []
    model.eval()
    model_s.eval()

    pbar = tqdm(indexes) if use_tqdm else indexes
    for index in pbar:

        env.init_new_problem(index)
        result = explore_v2(env, model, model_s, smooth, batch=batch, t_max=t_max, k=k, smoother='oracle', **kwargs)

        paths.append(result['path'])
        smooth_paths.append(result['smooth_path'])
        solutions.append(
            (result['success'], path_cost(result['path'], env), path_cost(result['smooth_path'], env),
             result['c_explore'], result['c_smooth'], result['total'], result['total_explore'], path_cost(result['optim_path'], env)))
        # if use_tqdm:
        #     pbar.set_description("gnn %.2fs, search %.2fs, explored %d" %
        #                          (result['forward'], result['total'] - result['forward'], len(result['explored'])))

    n_success = sum([s[0] for s in solutions])
    collision_explore = np.mean([s[3] for s in solutions])
    collision = np.mean([(s[3] + s[4]) for s in solutions])
    running_time = float(sum([s[5] for s in solutions if s[0]])) / n_success
    gnnonly__cost = float(sum([(s[1]) for s in solutions if s[0]])) / n_success
    smooth__cost = float(sum([(s[2]) for s in solutions if s[0]])) / n_success
    optimal_cost = float(sum([(s[7]) for s in solutions if s[0]])) / n_success
    # total_time = sum([s[5] for s in solutions])
    # total_time_explore = sum([s[6] for s in solutions])

    print('success rate:', n_success)
    print('collision check: %.2f' % collision)
    print('collision check explore: %.2f' % collision_explore)
    print('running time: %.2f' % running_time)
    print('gnn_only path cost: %.2f' % gnnonly__cost)
    print('smooth path cost: %.2f' % smooth__cost)
    print('optimal path cost: %.2f' % optimal_cost)
    # print('total time: %.2f' % total_time)
    # print('total time explore: %.2f' % total_time_explore)
    print('')



# evaluation on the test cases
env, indexes = str2env('snake7')  # choose env among ('maze2easy', 'maze2hard', 'kuka7', 'ur5', 'snake7', 'kuka13', 'kuka14')
# evaluation with GNN
_ = eval_gnn(str(env), 1234, env=env, indexes=indexes[:200], smooth=True, use_tqdm=True)