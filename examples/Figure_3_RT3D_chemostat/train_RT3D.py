import math
import os
import sys

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)

import multiprocessing

import hydra
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from RED.agents.continuous_agents.rt3d import RT3D_agent
from RED.environments.chemostat.xdot_chemostat import xdot
from RED.environments.OED_env import OED_env
from RED.utils.visualization import plot_returns

# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#how-to-perform-arithmetic-using-eval-as-a-resolver
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_3_RT3D_chemostat")
def train_RT3D(cfg: DictConfig):
    ### config setup
    cfg = cfg.example
    print(
        "--- Configuration ---",
        OmegaConf.to_yaml(cfg, resolve=True),
        "--- End of configuration ---",
        sep="\n\n"
    )

    # start a new wandb run to track this script
    wandb.init(project=cfg.project_name, config=dict(cfg))

    ### prepare save path
    os.makedirs(cfg.save_path, exist_ok=True)
    print("Results will be saved in: ", cfg.save_path)

    ### agent setup
    agent = instantiate(cfg.model)
    explore_rate = cfg.initial_explore_rate
    seq_dim = cfg.environment.n_observed_variables + 1 + cfg.environment.n_controlled_inputs

    ### env setup
    env, n_params = setup_env(cfg)
    total_episodes = cfg.environment.n_episodes // cfg.environment.n_parallel_experiments
    skip_first_n_episodes = cfg.environment.skip_first_n_experiments // cfg.environment.n_parallel_experiments

    history = {k: [] for k in ["returns", "actions", "rewards", "us", "explore_rate"]}
    update_count = 0

    ### training loop
    for episode in range(total_episodes):
        actual_params = np.random.uniform(
            low=cfg.environment.actual_params,
            high=cfg.environment.actual_params,
            size=(cfg.environment.n_parallel_experiments, n_params)
        )
        env.param_guesses = DM(actual_params)

        ### episode buffers for agent
        states = [env.get_initial_RL_state_parallel() for i in range(cfg.environment.n_parallel_experiments)]
        trajectories = [[] for _ in range(cfg.environment.n_parallel_experiments)]
        sequences = [[[0] * seq_dim] for _ in range(cfg.environment.n_parallel_experiments)]

        ### episode logging buffers
        e_returns = [0 for _ in range(cfg.environment.n_parallel_experiments)]
        e_actions = []
        e_rewards = [[] for _ in range(cfg.environment.n_parallel_experiments)]
        e_us = [[] for _ in range(cfg.environment.n_parallel_experiments)]

        ### reset env between episodes
        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(cfg.environment.n_parallel_experiments)]
        env.detFIMs = [[] for _ in range(cfg.environment.n_parallel_experiments)]

        ### run an episode
        for control_interval in range(0, cfg.environment.N_control_intervals):
            inputs = [states, sequences]

            ### get agent's actions
            if episode < skip_first_n_episodes:
                actions = agent.get_actions(inputs, explore_rate=1, test_episode=cfg.test_episode, recurrent=True)
            else:
                actions = agent.get_actions(inputs, explore_rate=explore_rate, test_episode=cfg.test_episode,
                                            recurrent=True)
            e_actions.append(actions)

            ### step env
            outputs = env.map_parallel_step(actions.T, actual_params, continuous=True)
            next_states = []
            for i, obs in enumerate(outputs):
                state, action = states[i], actions[i]
                next_state, reward, done, _, u = obs

                ### set done flag
                if control_interval == cfg.environment.N_control_intervals - 1 \
                        or np.all(np.abs(next_state) >= 1) \
                        or math.isnan(np.sum(next_state)):
                    done = True

                ### memorize transition
                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))

                ### log episode data
                e_us[i].append(u)
                next_states.append(next_state)
                if reward != -1:  # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

            states = next_states

        ### do not memorize the test trajectory (the last one)
        if cfg.test_episode:
            trajectories = trajectories[:-1]

        ### append trajectories to memory
        for trajectory in trajectories:
            # check for instability
            if np.all([np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]) \
                    and not math.isnan(np.sum(trajectory[-1][0])):
                agent.memory.append(trajectory)

        ### train agent
        if episode > skip_first_n_episodes:
            for _ in range(cfg.environment.n_parallel_experiments):
                update_count += 1
                update_policy = update_count % cfg.policy_delay == 0
                agent.Q_update(policy=update_policy, recurrent=True)

        ### update explore rate
        explore_rate = cfg.explore_rate_mul * agent.get_rate(
            episode=episode,
            min_rate=0,
            max_rate=1,
            denominator=cfg.environment.n_episodes / (11 * cfg.environment.n_parallel_experiments)
        )

        ### log results
        history["returns"].extend(e_returns)
        history["actions"].extend(np.array(e_actions).transpose(1, 0, 2))
        history["rewards"].extend(e_rewards)
        history["us"].extend(e_us)
        history["explore_rate"].append(explore_rate)

        ### log results to w and b
        for i in range(len(e_returns)):
            wandb.log({"returns": e_returns[i], "actions": np.array(e_actions).transpose(1, 0, 2)[i],
                       "us": e_us[i], "explore_rate": explore_rate})

        print(
            f"\nEPISODE: [{episode}/{total_episodes}] ({episode * cfg.environment.n_parallel_experiments} experiments)",
            f"explore rate:\t{explore_rate:.2f}",
            f"average return:\t{np.mean(e_returns):.5f}",
            sep="\n",
        )

        if cfg.test_episode:
            print(
                f"test actions:\n{np.array(e_actions)[:, -1]}",
                f"test rewards:\n{np.array(e_rewards)[-1, :]}",
                f"test return:\n{np.sum(np.array(e_rewards)[-1, :])}",
                sep="\n",
            )

        ### checkpoint
        if cfg.ckpt_freq is not None and episode % cfg.ckpt_freq == 0:
            ckpt_dir = os.path.join(cfg.save_path, f"ckpt_{episode}")
            os.makedirs(ckpt_dir, exist_ok=True)
            agent.save_network(ckpt_dir)
            for k in history.keys():
                np.save(os.path.join(ckpt_dir, f"{k}.npy"), np.array(history[k]))

    ### save results and plot
    agent.save_network(cfg.save_path)
    for k in history.keys():
        np.save(os.path.join(cfg.save_path, f"{k}.npy"), np.array(history[k]))
    plot_returns(
        returns=history["returns"],
        explore_rates=history["explore_rate"],
        show=False,
        save_to_dir=cfg.save_path,
        conv_window=25,
    )

    wandb.finish()


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
    env.mapped_trajectory_solver = env.CI_solver.map(cfg.environment.n_parallel_experiments, "thread", n_cores)
    return env, n_params


if __name__ == '__main__':
    train_RT3D()
