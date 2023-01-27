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

SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/Figure_2_FQ_chemostat")
def MPC_param_inf(cfg : DictConfig):
    # setup
    cfg = cfg.example

    actual_params = DM(cfg.environment.actual_params)
    normaliser = np.array(cfg.environment.normaliser)
    os.makedirs(cfg.save_path, exist_ok=True)

    param_guesses = DM((np.array(cfg.environment.ub) + np.array(cfg.environment.lb))/2)
    args = cfg.environment.y0, xdot, param_guesses, actual_params, cfg.environment.n_observed_variables, \
        cfg.environment.n_controlled_inputs, cfg.environment.num_inputs, cfg.environment.input_bounds, \
        cfg.environment.dt, cfg.environment.control_interval_time, normaliser

    env = OED_env(*args)
    input_bounds = np.array(cfg.environment.input_bounds)
    u0 = (input_bounds[:,1] + input_bounds[:,0])/2
    env.u0 = DM(u0)


    def get_full_u_solver():
        '''
        creates and return the solver which will optimise a full exepiments inputs wrt the FI
        :return: solver
        '''
        us = SX.sym('us', cfg.environment.N_control_intervals * cfg.environment.n_controlled_inputs)
        trajectory_solver = env.get_sampled_trajectory_solver(cfg.environment.N_control_intervals, cfg.environment.control_interval_time, cfg.environment.dt)
        est_trajectory = trajectory_solver(env.initial_Y, param_guesses, reshape(us , (cfg.environment.n_controlled_inputs, cfg.environment.N_control_intervals)))

        FIM = env.get_FIM(est_trajectory)

        q, r = qr(FIM)

        obj = -trace(log(r))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = True) # for some reason limited mem works better for the MPC
        return solver


    # run optimisation
    u0 = (input_bounds[:,1] + input_bounds[:,0])/2
    u_solver = get_full_u_solver()
    sol = u_solver(
        x0=u0,
        lbx = [input_bounds[0][0]] * cfg.environment.n_controlled_inputs * cfg.environment.N_control_intervals,
        ubx = [input_bounds[0][1]] * cfg.environment.n_controlled_inputs * cfg.environment.N_control_intervals
    )
    us = sol['x']

    # save results and plot
    np.save(os.path.join(cfg.save_path, 'us.npy'), np.array(env.us))


    t = np.arange(cfg.environment.N_control_intervals) * int(cfg.environment.control_interval_time)


if __name__ == '__main__':
    MPC_param_inf()
