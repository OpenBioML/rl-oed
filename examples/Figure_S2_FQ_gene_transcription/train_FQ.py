import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)

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
from RED.agents.fitted_Q_agents import KerasFittedQAgent
from RED.environments.OED_env import OED_env
from RED.environments.gene_transcription.xdot_gene_transcription import xdot

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def action_scaling(u):
    return 10**u

@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_S2_FQ_gene_transcription")
def train_FQ(cfg : DictConfig):
    cfg = cfg.example

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    #setup
    agent = instantiate(cfg.model)
    actual_params = DM(cfg.environment.actual_params)
    n_params = actual_params.size()[0]
    n_FIM_elements = sum(range(n_params + 1))
    # ??? vvv
    param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5])
    param_guesses = actual_params
    # ??? ^^^
    normaliser = np.array(cfg.environment.normaliser)
    env = OED_env(cfg.environment.y0, xdot, param_guesses, actual_params, \
        cfg.environment.n_observed_variables, cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, \
        cfg.environment.input_bounds, cfg.environment.dt, cfg.environment.control_interval_time, normaliser)
    
    explore_rate = cfg.init_explore_rate
    all_returns = []

    for episode in range(cfg.environment.n_episodes): #training loop

        env.reset()
        state = env.get_initial_RL_state(use_old_state=True)
        e_return = 0
        e_actions =[]
        e_rewards = []
        trajectory = []

        for e in range(0, cfg.environment.N_control_intervals): # run episode
            t = time.time()
            action = agent.get_action(state.reshape(-1, 23), explore_rate)
            next_state, reward, done, _ = env.step(action, use_old_state = True, scaling = action_scaling)
            if e == cfg.environment.N_control_intervals - 1:
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
        if episode % skip == 0 or episode == cfg.environment.n_episodes - 2: #train agent
            explore_rate = agent.get_rate(episode, 0, 1, cfg.environment.n_episodes / 10)
            if explore_rate == 1:
                n_iters = 0
            elif len(agent.memory[0]) * len(agent.memory) < 40000:
                n_iters = 1
            else:
                n_iters = 2

            for iter in range(n_iters):
                agent.fitted_Q_update()

        all_returns.append(e_return)

        if episode %skip == 0 or episode == cfg.environment.n_episodes -1:
            print()
            print('EPISODE: ', episode)
            print('explore rate: ', explore_rate)
            print('return: ', e_return)
            print('av return: ', np.mean(all_returns[-skip:]))

    # save and plot
    agent.save_network(cfg.save_path)
    np.save(os.path.join(cfg.save_path, 'trajectories.npy'), np.array(env.true_trajectory))
    np.save(os.path.join(cfg.save_path, 'true_trajectory.npy'), env.true_trajectory)
    np.save(os.path.join(cfg.save_path, 'us.npy'), np.array(env.us))
    np.save(os.path.join(cfg.save_path, 'all_returns.npy'), np.array(all_returns))
    np.save(os.path.join(cfg.save_path,'actions.npy'), np.array(agent.actions))
    np.save(os.path.join(cfg.save_path,'values.npy'), np.array(agent.values))
    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)
    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.savefig(os.path.join(cfg.save_path,'rna_trajectories.pdf'))
    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    plt.legend()
    plt.ylabel( 'protein')
    plt.xlabel('time (mins)')
    plt.savefig(os.path.join(cfg.save_path, 'prot_trajectories.pdf'))
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(os.path.join(cfg.save_path, 'log_us.pdf'))
    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(cfg.save_path, 'return.pdf'))
    plt.show()


if __name__ == '__main__':
    train_FQ()
