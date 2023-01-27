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
from omegaconf import DictConfig

from RED.environments.OED_env import OED_env
from RED.environments.chemostat.xdot_chemostat import xdot
import json

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_2_FQ_chemostat")
def OSAO_param_inf(cfg : DictConfig):
    cfg = cfg.example
    os.makedirs(cfg.save_path, exist_ok=True)

    #setup
    actual_params = DM(cfg.environment.actual_params)
    normaliser = np.array(cfg.environment.normaliser)
    n_params = actual_params.size()[0]
    n_system_variables = len(cfg.environment.y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = DM((np.array(cfg.environment.ub) + np.array(cfg.environment.lb))/2)
    env = OED_env(cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, \
        cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, \
        cfg.environment.dt, cfg.environment.control_interval_time, normaliser)
    input_bounds = np.array(cfg.environment.input_bounds)
    u0 = (10.0**input_bounds[:,1] + 10.0**input_bounds[:,0])/2
    env.u0 = DM(u0)
    e_rewards = []


    #run optimisation
    for e in range(0, cfg.environment.N_control_intervals):
        next_state, reward, done, _ = env.step()


        if e == cfg.environment.N_control_intervals - 1:
            next_state = [None]*24
            done = True

        e_rewards.append(reward)
        state = next_state

    # save results and plot
    np.save(os.path.join(cfg.save_path, 'trajectories.npy'), np.array(env.true_trajectory))

    np.save(os.path.join(cfg.save_path, 'true_trajectory.npy'), env.true_trajectory)
    np.save(os.path.join(cfg.save_path, 'us.npy'), np.array(env.us))

    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label='true')
    plt.legend()
    plt.ylabel('bacteria')
    plt.xlabel('time (mins)')
    plt.savefig(os.path.join(cfg.save_path, 'bacteria_trajectories.pdf'))

    plt.figure()
    plt.plot(env.true_trajectory[1, :].elements(), label='true')
    plt.legend()
    plt.ylabel('C')
    plt.xlabel('time (mins)')
    plt.savefig(os.path.join(cfg.save_path, 'c_trajectories.pdf'))

    plt.figure()
    plt.plot(env.true_trajectory[2, :].elements(), label='true')
    plt.legend()
    plt.ylabel('C0')
    plt.xlabel('time (mins)')
    plt.savefig(os.path.join(cfg.save_path, 'c0_trajectories.pdf'))

    plt.figure()
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(os.path.join(cfg.save_path, 'log_us.pdf'))
    plt.show()


if __name__ == '__main__':
    OSAO_param_inf()
