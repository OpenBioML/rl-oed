
import sys
import os
IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)



import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt


import time

from RED.agents.fitted_Q_agents import KerasFittedQAgent
from RED.environments.OED_env import OED_env
from RED.environments.chemostat.xdot_chemostat import xdot
import json

import multiprocessing




use_old_state = True
if __name__ == '__main__':

    #run setup
    n_cores = multiprocessing.cpu_count()
    param_dir = os.path.join(os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'RED',
                     'environments'), 'chemostat'))
    params = json.load(open(os.path.join(param_dir, 'params_chemostat.json')))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]
    actual_params = DM(actual_params)
    if use_old_state:
        normaliser = np.array([1e3, 1e1, 1e-3, 1e-4, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e2])
    normaliser = np.array(normaliser)
    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_observed_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = actual_params
    save_path = './results/'
    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 1, 150, 150, 150, num_inputs ** n_controlled_inputs])
    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)
    actual_params = np.random.uniform(low=[1, 0.00048776, 0.00006845928], high=[1, 0.00048776, 0.00006845928],
                                      size=(skip, 3))
    env.param_guesses = DM(actual_params)
    explore_rate = 1
    alpha = 1
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    t = time.time()
    all_returns = []


    for episode in range(int(n_episodes//skip)): # training loop

        states  = [env.get_initial_RL_state_parallel(use_old_state = use_old_state, i=i) for i in range(skip)]

        e_returns = [0 for _ in range(skip)]
        e_actions = []
        e_rewards = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]

        for e in range(0, N_control_intervals): # run an episode

            actions = agent.get_actions(states, explore_rate)
            e_actions.append(actions)
            outputs = env.map_parallel_step(np.array(actions).T, actual_params, use_old_state = use_old_state)
            next_states = []

            for i,o in enumerate(outputs): # extract outputs from episodes that have been run in parallel
                next_state, reward, done, _, u = o
                next_states.append(next_state)
                state = states[i]
                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
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
            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11*skip))
            alpha = agent.get_rate(episode, 0, 1, n_episodes / (10*skip))

            if explore_rate == 1:
                n_iters = 0
            else:
                n_iters = 1


            for iter in range(n_iters):
                history = agent.fitted_Q_update(alpha = alpha)

        print()
        print('EPISODE: ', episode * skip)
        print('explore rate: ', explore_rate)

        print('av return: ', np.mean(all_returns[-skip:]))

    #save results and plot
    agent.save_network(save_path)
    np.save(save_path + 'all_returns.npy', np.array(all_returns))

    np.save(save_path + 'actions.npy', np.array(agent.actions))
    np.save(save_path + 'values.npy', np.array(agent.values))

    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel('bacteria')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'bacteria_trajectories.pdf')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel( 'C')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c_trajectories.pdf')

    plt.figure()
    plt.plot(env.true_trajectory[2, :].elements(), label='true')
    plt.legend()
    plt.ylabel('C0')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c0_trajectories.pdf')


    plt.figure()
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')

    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(save_path + 'return.pdf')



    plt.show()
