
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

from RED.agents.continuous_agents import RT3D_agent
from RED.environments.OED_env import OED_env
from RED.environments.chemostat.xdot_chemostat import xdot

import time

import tensorflow as tf

import multiprocessing
import json


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_5_RT3D_chemostat_prior")
def train_RT3D_prior(cfg : DictConfig):
    #setup
    cfg = cfg.example
    
    actual_params = DM(cfg.environment.actual_params)
    normaliser = np.array(cfg.environment.normaliser)
    n_params = actual_params.size()[0]
    n_system_variables = len(cfg.environment.y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = actual_params
    physical_devices = tf.config.list_physical_devices('GPU')
    
    n_cores = multiprocessing.cpu_count()
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    #agent setup
    pol_layer_sizes = [cfg.environment.n_observed_variables + 1, cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, cfg.hidden_layer_size[0], cfg.hidden_layer_size[1], cfg.environment.n_controlled_inputs]
    val_layer_sizes = [cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, cfg.hidden_layer_size[0], cfg.hidden_layer_size[1], 1]

    agent = instantiate(
        cfg.model,
        val_layer_sizes=val_layer_sizes,
        pol_layer_sizes=pol_layer_sizes,
        batch_size=int(cfg.environment.N_control_intervals * cfg.environment.skip),
        max_length=11,
    )

    explore_rate = cfg.explore_rate
    update_count = 0
    alpha = 1
    all_returns = []
    all_test_returns = []

    #env setup
    args = cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, \
        cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, \
        cfg.environment.dt, cfg.environment.control_interval_time, normaliser
    env = OED_env(*args)
    env.mapped_trajectory_solver = env.CI_solver.map(cfg.environment.skip, "thread", n_cores)

    for episode in range(int(cfg.environment.n_episodes//cfg.environment.skip)): #training loop

        actual_params = np.random.uniform(low=cfg.environment.lb, high=cfg.environment.ub,  size = (cfg.environment.skip, 3)) # sample from uniform distribution
        env.param_guesses = DM(actual_params)
        states = [env.get_initial_RL_state_parallel() for i in range(cfg.environment.skip)]
        e_returns = [0 for _ in range(cfg.environment.skip)]
        e_actions = []
        e_exploit_flags =[]
        e_rewards = [[] for _ in range(cfg.environment.skip)]
        e_us = [[] for _ in range(cfg.environment.skip)]
        trajectories = [[] for _ in range(cfg.environment.skip)]
        sequences = [[[0]*pol_layer_sizes[1]] for _ in range(cfg.environment.skip)]
        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(cfg.environment.skip)]
        env.detFIMs = [[] for _ in range(cfg.environment.skip)]

        for e in range(0, cfg.environment.N_control_intervals): # run an episode
            inputs = [states, sequences]
            if episode < 1000 // cfg.environment.skip:
                actions = agent.get_actions(inputs, explore_rate = 1, test_episode=cfg.test_episode, recurrent=True)
            else:
                actions = agent.get_actions(inputs, explore_rate=explore_rate, test_episode=cfg.test_episode, recurrent=True)

            e_actions.append(actions)
            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous = True)
            next_states = []

            for i,o in enumerate(outputs): #extract outputs from parallel experiments
                next_state, reward, done, _, u  = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]
                action = actions[i]

                if e == cfg.environment.N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward
            states = next_states

        if cfg.test_episode:
            trajectories = trajectories[:-1]
        for trajectory in trajectories:
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                agent.memory.append(trajectory)


        if episode > 1000 // cfg.environment.skip: #train agent
            print('training', update_count)
            t = time.time()
            for _ in range(cfg.environment.skip):

                update_count += 1
                policy = update_count % cfg.policy_delay == 0

                agent.Q_update(policy=policy, fitted=False, recurrent=True, low_mem=False)
            print('fitting time', time.time() - t)

        explore_rate = agent.get_rate(episode, 0, 1, cfg.environment.n_episodes / (11 * cfg.environment.skip)) * cfg.max_std

        if cfg.test_episode:
            all_returns.extend(e_returns[:-1])
            all_test_returns.append(np.sum(np.array(e_rewards)[-1, :]))
        else:
            all_returns.extend(e_returns)
        all_returns.extend(e_returns)
        print()
        print('EPISODE: ', episode, episode*cfg.environment.skip)


        print('av return: ', np.mean(all_returns[-cfg.environment.skip:]))
        print()

        if cfg.test_episode:
            print('test actions:', np.array(e_actions)[:, -1])
            print('test rewards:', np.array(e_rewards)[-1, :])
            print('test return:', np.sum(np.array(e_rewards)[-1, :]))
            print()

    #plot and save results
    agent.save_network(cfg.save_path)
    np.save(os.path.join(cfg.save_path, 'all_returns.npy'), np.array(all_returns))
    np.save(os.path.join(cfg.save_path, 'actions.npy'), np.array(agent.actions))

    if cfg.test_episode:
        np.save(cfg.save_path + '/all_test_returns.npy', np.array(all_test_returns))


    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)

    print(all_returns)
    print(agent.actions)


    #plt.plot(all_returns)
    #plt.show()


if __name__ == '__main__':
    train_RT3D_prior()
