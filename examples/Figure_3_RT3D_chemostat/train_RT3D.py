
import math
import os
import sys
import time

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)
print(IMPORT_PATH)

import multiprocessing

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from RED.agents.continuous_agents.rt3d import RT3D_agent
from RED.environments.chemostat.xdot_chemostat import xdot
from RED.environments.OED_env import OED_env

OmegaConf.register_new_resolver("eval", eval) # https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#how-to-perform-arithmetic-using-eval-as-a-resolver


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_3_RT3D_chemostat")
def train_RT3D(cfg : DictConfig):
    ### config setup
    cfg = cfg.example
    print(
        "--- Configuration ---",
        OmegaConf.to_yaml(cfg, resolve=True),
        "--- End of configuration ---",
        sep="\n\n"
    )

    ### prepare save path
    save_path = os.path.join(cfg.save_path, time.strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_path, exist_ok=True)
    print("Results will be saved in: ", save_path)

    ### agent setup
    agent = instantiate(cfg.model)
    explore_rate = cfg.initial_explore_rate
    seq_dim = cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs

    ### env setup
    env, n_params = setup_env(cfg)
    
    total_episodes = cfg.environment.n_episodes // cfg.environment.skip
    history = {k: [] for k in ["returns", "actions", "rewards", "us", "explore_rate"]}
    update_count = 0

    ### training loop
    for episode in range(total_episodes):
        actual_params = np.random.uniform(
            low=cfg.environment.actual_params,
            high=cfg.environment.actual_params,
            size=(cfg.environment.skip, n_params)
        )
        env.param_guesses = DM(actual_params)
        
        ### episode buffers for agent
        states = [env.get_initial_RL_state_parallel() for i in range(cfg.environment.skip)]
        trajectories = [[] for _ in range(cfg.environment.skip)]
        sequences = [[[0] * seq_dim] for _ in range(cfg.environment.skip)]

        ### episode logging buffers
        e_returns = [0 for _ in range(cfg.environment.skip)]
        e_actions = []
        e_rewards = [[] for _ in range(cfg.environment.skip)]
        e_us = [[] for _ in range(cfg.environment.skip)]

        ### reset env between episodes
        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(cfg.environment.skip)]
        env.detFIMs = [[] for _ in range(cfg.environment.skip)]

        ### run an episode
        for control_interval in range(0, cfg.environment.N_control_intervals):
            inputs = [states, sequences]

            ### get agent's actions
            if episode < 1000 // cfg.environment.skip:
                actions = agent.get_actions(inputs, explore_rate=1, test_episode=True, recurrent=True)
            else:
                actions = agent.get_actions(inputs, explore_rate=explore_rate, test_episode=True, recurrent=True)

            e_actions.append(actions)
            outputs = env.map_parallel_step(actions.T, actual_params, continuous = True)
            next_states = []

            for i, obs in enumerate(outputs): #extract outputs from experiments run in parallel
                state = states[i]
                action = actions[i]
                
                next_state, reward, done, _, u  = obs
                e_us[i].append(u)
                next_states.append(next_state)

                ### set done flag
                if control_interval == cfg.environment.N_control_intervals - 1 \
                    or np.all(np.abs(next_state) >= 1) \
                    or math.isnan(np.sum(next_state)):
                    done = True

                ### memorize transition
                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward
            states = next_states

        ### append trajectories to memory
        for trajectory in trajectories:
            # check for instability
            if np.all([np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]) \
                and not math.isnan(np.sum(trajectory[-1][0])):
                agent.memory.append(trajectory)

        ### train agent
        if episode > 1000 // cfg.environment.skip:
            for _ in range(cfg.environment.skip):
                update_count += 1
                policy = update_count % cfg.policy_delay == 0
                agent.Q_update(policy=policy, recurrent=True)

        ### update explore rate
        explore_rate = cfg.max_std * agent.get_rate(
            episode=episode,
            min_rate=0,
            max_rate=1,
            denominator=cfg.environment.n_episodes / (11 * cfg.environment.skip)
        )

        ### log results
        history["returns"].extend(e_returns)
        history["actions"].extend(e_actions)
        history["rewards"].extend(e_rewards)
        history["us"].extend(e_us)
        history["explore_rate"].append(explore_rate)

        print()
        print(f"EPISODE: [{episode}/{total_episodes}] ({episode * cfg.environment.skip})")
        print(f"explore rate: {explore_rate}")
        print(f"average return: {np.mean(history['returns'][-cfg.environment.skip:])}")

    ### save results and plot
    agent.save_network(save_path)
    for k in history.keys():
        np.save(os.path.join(save_path, f'{k}.npy'), np.array(history[k]))

    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)
    plt.plot(history['returns'])
    plt.show()


def setup_env(cfg):
    n_cores = multiprocessing.cpu_count()
    actual_params = DM(cfg.environment.actual_params)
    normaliser = np.array(cfg.environment.normaliser)
    n_params = actual_params.size()[0]
    param_guesses = actual_params
    args = cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, \
        cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, \
        cfg.environment.dt, cfg.environment.control_interval_time, normaliser
    env = OED_env(*args)
    env.mapped_trajectory_solver = env.CI_solver.map(cfg.environment.skip, "thread", n_cores)
    return env, n_params


if __name__ == '__main__':
    train_RT3D()
