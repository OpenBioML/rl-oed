import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

import time
import tensorflow as tf
from RED.agents.fitted_Q_agents import KerasFittedQAgent
from RED.environments.OED_env import OED_env
from RED.environments.gene_transcription.xdot_gene_transcription import xdot

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':

    n_episodes = 20000

    save_path = './results'
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    #setup
    agent = KerasFittedQAgent(layer_sizes = [23, 150, 150, 150, 12])
    all_returns = []
    actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])
    input_bounds = [-3, 3] # on the log scale
    n_params = actual_params.size()[0]
    n_system_variables = 2
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    num_inputs = 12  # number of discrete inputs available to RL
    dt = 1 / 100
    param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5])
    param_guesses = actual_params
    y0 = [0.000001, 0.000001]
    normaliser = np.array([1e3, 1e4, 1e2, 1e6, 1e10, 1e-3, 1e1, 1e9, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1, 1e9, 1, 1e7,10])
    N_control_intervals = 6
    control_interval_time = 100
    n_observed_variables = 2
    n_controlled_inputs = 1
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
    explore_rate = 1

    for episode in range(n_episodes): #training loop

        env.reset()
        state = env.get_initial_RL_state(use_old_state=True)
        e_return = 0
        e_actions =[]
        e_rewards = []
        trajectory = []

        for e in range(0, N_control_intervals): # run episode
            t = time.time()
            action = agent.get_action(state.reshape(-1, 23), explore_rate)
            next_state, reward, done, _ = env.step(action, use_old_state = True)
            if e == N_control_intervals - 1:
                next_state = [None]*23
                done = True
            transition = (state, action, reward, next_state, done)
            trajectory.append(transition)
            e_actions.append(action)
            e_rewards.append(reward)
            state = next_state
            e_return += reward

        agent.memory.append(trajectory)
        #train the agent
        skip = 200
        if episode % skip == 0 or episode == n_episodes - 2: #train agent
            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / 10)
            if explore_rate == 1:
                n_iters = 0
            elif len(agent.memory[0]) * len(agent.memory) < 40000:
                n_iters = 1
            else:
                n_iters = 2

            for iter in range(n_iters):
                agent.fitted_Q_update()

        all_returns.append(e_return)

        if episode %skip == 0 or episode == n_episodes -1:
            print()
            print('EPISODE: ', episode)
            print('explore rate: ', explore_rate)
            print('return: ', e_return)
            print('av return: ', np.mean(all_returns[-skip:]))

    # save and plot
    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))
    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    np.save(save_path + 'values.npy', np.array(agent.values))
    t = np.arange(N_control_intervals) * int(control_interval_time)
    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'rna_trajectories.pdf')
    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel( 'protein')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'prot_trajectories.pdf')
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
