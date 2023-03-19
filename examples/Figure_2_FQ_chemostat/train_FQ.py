
import sys
import os
IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)



import math
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import time

from RED.agents.fitted_Q_agents import KerasFittedQAgent
from RED.environments.OED_env import OED_env
from RED.environments.chemostat.xdot_chemostat import xdot
import json

import multiprocessing


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_2_FQ_chemostat")
def train_FQ(cfg : DictConfig):
    #run setup
    cfg = cfg.example
    
    n_cores = multiprocessing.cpu_count()
    
    actual_params = DM(cfg.environment.actual_params)
    if cfg.use_old_state:
        normaliser = np.array(cfg.old_state_normaliser)
    else:
        normaliser = np.array(cfg.environment.normaliser)
    n_params = actual_params.size()[0]
    n_system_variables = len(cfg.environment.y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = cfg.environment.n_observed_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = actual_params
    
    agent = instantiate(
        cfg.model,
        layer_sizes=[
            cfg.environment.n_observed_variables + n_params + n_FIM_elements + 1,
            *cfg.hidden_layer_sizes,
            cfg.environment.num_inputs ** cfg.environment.n_controlled_inputs
        ]
    )
    
    args = cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, \
        cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, \
        cfg.environment.dt, cfg.environment.control_interval_time, normaliser
    env = OED_env(*args)
    env.param_guesses = DM(actual_params)
    actual_params = np.random.uniform(low=[1, 0.00048776, 0.00006845928], high=[1, 0.00048776, 0.00006845928],
                                      size=(cfg.environment.skip, 3))
    env.mapped_trajectory_solver = env.CI_solver.map(cfg.environment.skip, "thread", n_cores)
    explore_rate = cfg.init_explore_rate
    alpha = cfg.init_alpha
    t = time.time()
    all_returns = []


    for episode in range(int(cfg.environment.n_episodes//cfg.environment.skip)): # training loop

        states  = [env.get_initial_RL_state_parallel(use_old_state = cfg.use_old_state, i=i) for i in range(cfg.environment.skip)]

        e_returns = [0 for _ in range(cfg.environment.skip)]
        e_actions = []
        e_rewards = [[] for _ in range(cfg.environment.skip)]
        trajectories = [[] for _ in range(cfg.environment.skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(cfg.environment.skip)]
        env.detFIMs = [[] for _ in range(cfg.environment.skip)]

        for e in range(0, cfg.environment.N_control_intervals): # run an episode

            actions = agent.get_actions(states, explore_rate)
            e_actions.append(actions)
            outputs = env.map_parallel_step(np.array(actions).T, actual_params, use_old_state = cfg.use_old_state)
            next_states = []

            for i,o in enumerate(outputs): # extract outputs from episodes that have been run in parallel
                next_state, reward, done, _, u = o
                next_states.append(next_state)
                state = states[i]
                action = actions[i]

                if e == cfg.environment.N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    next_state = [None]*agent.layer_sizes[0]
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

                state = next_state

            states = next_states

        for j, trajectory in enumerate(trajectories): # add trajectory to memory
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability

                agent.memory.append(trajectory)
                all_returns.append(e_returns[j])


        if episode != 0: # train agent
            explore_rate = agent.get_rate(episode, 0, 1, cfg.environment.n_episodes / (11*cfg.environment.skip))
            alpha = agent.get_rate(episode, 0, 1, cfg.environment.n_episodes / (10*cfg.environment.skip))

            if explore_rate == 1:
                n_iters = 0
            else:
                n_iters = 1


            for iter in range(n_iters):
                history = agent.fitted_Q_update(alpha = alpha)

        print()
        print('EPISODE: ', episode * cfg.environment.skip)
        print('explore rate: ', explore_rate)

        print('av return: ', np.mean(all_returns[-cfg.environment.skip:]))

    #save results and plot
    agent.save_network(cfg.save_path)
    np.save(os.path.join(cfg.save_path, 'all_returns.npy'), np.array(all_returns))

    np.save(os.path.join(cfg.save_path, 'actions.npy'), np.array(agent.actions))
    np.save(os.path.join(cfg.save_path, 'values.npy'), np.array(agent.values))

    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)



    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(cfg.save_path,'return.pdf'))



    plt.show()


if __name__ == '__main__':
    train_FQ()
