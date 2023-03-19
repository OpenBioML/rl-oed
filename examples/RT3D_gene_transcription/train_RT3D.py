import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)
print(IMPORT_PATH)
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import time
import tensorflow as tf
from RED.environments.OED_env import OED_env
from RED.environments.gene_transcription.xdot_gene_transcription import xdot
from RED.agents.continuous_agents import RT3D_agent
import multiprocessing
import json
import math


def action_scaling(u):
    return 10**u

@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/RT3D_gene_transcription")
def train_RT3D(cfg : DictConfig):
    cfg = cfg.example
    
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)

    actual_params = DM(cfg.environment.actual_params)
    n_params = actual_params.size()[0]
    n_system_variables = len(cfg.environment.y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = actual_params

    normaliser = np.array(cfg.environment.normaliser)

    os.makedirs(cfg.save_path, exist_ok=True)

    if cfg.recurrent:
        # pol_layer_sizes = [cfg.environment.n_observed_variables + 1, cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, [32, 32], [64,64,64], cfg.environment.n_controlled_inputs]
        pol_layer_sizes = [cfg.environment.n_observed_variables + 1, cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs,
                           cfg.hidden_layer_size[0], cfg.hidden_layer_size[1], cfg.environment.n_controlled_inputs]
        val_layer_sizes = [cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs,
                           cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, cfg.hidden_layer_size[0], cfg.hidden_layer_size[1],
                           1]
        # agent = DQN_agent(layer_sizes=[cfg.environment.n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, cfg.environment.num_inputs ** cfg.environment.n_controlled_inputs])
    else:
        pol_layer_sizes = [cfg.environment.n_observed_variables + 1, 0, [], [128, 128], cfg.environment.n_controlled_inputs]
        val_layer_sizes = [cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs, 0, [], [128, 128], 1]

    # agent = DRPG_agent(layer_sizes=layer_sizes, learning_rate = 0.0004, critic = True)
    agent = instantiate(
        cfg.model,
        pol_layer_sizes=pol_layer_sizes,
        val_layer_sizes=val_layer_sizes,
        batch_size=int(cfg.environment.N_control_intervals * cfg.environment.skip),
        max_length=cfg.environment.N_control_intervals + 1,
    )
    
    args = cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, cfg.environment.dt, cfg.environment.control_interval_time, normaliser
    env = OED_env(*args)

    explore_rate = cfg.explore_rate
    alpha = 1
    env.mapped_trajectory_solver = env.CI_solver.map(cfg.environment.skip, "thread", n_cores)
    total_t = time.time()

    unstable = 0
    n_unstables = []
    all_returns = []
    all_test_returns = []
    update_count = 0
    fitted = False

    if cfg.load_agent_network:
        agent.load_network(cfg.agent_network_path)
    print('time:', cfg.environment.control_interval_time)
    
    for episode in range(int(cfg.environment.n_episodes // cfg.environment.skip)):
        print(episode)

        if cfg.environment.prior:
            actual_params = np.random.uniform(low=cfg.environment.lb, high=cfg.environment.ub, size=(cfg.environment.skip, 5))
        else:
            actual_params = np.random.uniform(low=[20, 500000, 1.09e+09, 0.000257, 4], high=[20, 500000, 1.09e+09, 0.000257, 4], size=(cfg.environment.skip, 5))
        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel() for i in range(cfg.environment.skip)]

        e_returns = [0 for _ in range(cfg.environment.skip)]

        e_actions = []

        e_exploit_flags = []
        e_rewards = [[] for _ in range(cfg.environment.skip)]
        e_us = [[] for _ in range(cfg.environment.skip)]
        trajectories = [[] for _ in range(cfg.environment.skip)]

        sequences = [[[0] * pol_layer_sizes[1]] for _ in range(cfg.environment.skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(cfg.environment.skip)]
        env.detFIMs = [[] for _ in range(cfg.environment.skip)]

        for e in range(0, cfg.environment.N_control_intervals):

            if cfg.recurrent:
                inputs = [states, sequences]
            else:
                inputs = [states]

            if episode < 1000 // cfg.environment.skip:
                actions = agent.get_actions(inputs, explore_rate=0, test_episode=cfg.test_episode)
            else:
                actions = agent.get_actions(inputs, explore_rate=cfg.explore_rate, test_episode=cfg.test_episode)

            e_actions.append(actions)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous=True, scaling=action_scaling)
            next_states = []
            print(actions)

            for i, o in enumerate(outputs):
                next_state, reward, done, _, u = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]

                action = actions[i]

                if e == cfg.environment.N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    # next_state = [0]*pol_layer_sizes[0] # maybe dont need this
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1:  # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward


            # print('sequences', np.array(sequences).shape)
            # print('sequences', sequences[0])
            states = next_states

        if cfg.test_episode:
            trajectories = trajectories[:-1]

        for trajectory in trajectories:
            if np.all([np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]) and not math.isnan(
                    np.sum(trajectory[-1][0])):  # check for instability
                agent.memory.append(trajectory)  # monte carlo, fitted

            else:
                unstable += 1
                print('UNSTABLE!!!')
                print((trajectory[-1][0]))


        if episode > 1000 // cfg.environment.skip:
            print('training', update_count)
            t = time.time()
            for hello in range(cfg.environment.skip):
                # print(e, episode, hello, update_count)
                update_count += 1
                policy = update_count % cfg.policy_delay == 0

                agent.Q_update(policy=policy, fitted=fitted, recurrent=cfg.recurrent, low_mem = True)
            print('fitting time', time.time() - t)

        explore_rate = agent.get_rate(episode, 0, 1, cfg.environment.n_episodes / (11 * cfg.environment.skip)) * cfg.max_std
        '''
        if episode > 1000//cfg.environment.skip:
            update_count += 1
            agent.Q_update( policy=update_count%cfg.policy_delay == 0, fitted=True)
        '''

        print('n unstable ', unstable)
        n_unstables.append(unstable)

        if cfg.test_episode:
            all_returns.extend(e_returns[:-1])
            all_test_returns.append(np.sum(np.array(e_rewards)[-1, :]))
        else:
            all_returns.extend(e_returns)

        print()
        print('EPISODE: ', episode, episode * cfg.environment.skip)

        print('moving av return:', np.mean(all_returns[-10 * cfg.environment.skip:]))
        print('explore rate: ', explore_rate)
        print('alpha:', alpha)
        print('av return: ', np.mean(all_returns[-cfg.environment.skip:]))
        print()

        # print('us:', np.array(e_us)[0, :])



        if cfg.test_episode:
            print('test actions:', np.array(e_actions)[:, -1])
            print('test rewards:', np.array(e_rewards)[-1, :])
            print('test return:', np.sum(np.array(e_rewards)[-1, :]))
            print()

    print('time:', time.time() - total_t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])
    np.save(cfg.save_path + '/all_returns.npy', np.array(all_returns))
    if cfg.test_episode:
        np.save(cfg.save_path + '/all_test_returns.npy', np.array(all_test_returns))

    np.save(cfg.save_path + '/n_unstables.npy', np.array(n_unstables))
    np.save(cfg.save_path + '/actions.npy', np.array(agent.actions))
    agent.save_network(cfg.save_path)

    # np.save(cfg.save_path + 'values.npy', np.array(agent.values))
    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)

    plt.plot(all_test_returns)
    plt.figure()
    plt.plot(all_returns)
    plt.show()


if __name__ == '__main__':
    train_RT3D()
