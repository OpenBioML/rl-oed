import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)


from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
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



if __name__ == '__main__':


    # setup
    param_dir =  os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'RED', 'environments'), 'chemostat'))
    params = json.load(open(os.path.join(param_dir, 'params_chemostat.json')))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]
    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)
    save_path = os.path.join('.', 'results')
    os.makedirs(save_path, exist_ok=True)

    param_guesses = DM((np.array(ub) + np.array(lb))/2)
    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser

    env = OED_env(*args)
    input_bounds = np.array(input_bounds)
    u0 = (input_bounds[:,1] + input_bounds[:,0])/2
    env.u0 = DM(u0)


    def get_full_u_solver():
        '''
        creates and return the solver which will optimise a full exepiments inputs wrt the FI
        :return: solver
        '''
        us = SX.sym('us', N_control_intervals * n_controlled_inputs)
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, param_guesses, reshape(us , (n_controlled_inputs, N_control_intervals)))

        FIM = env.get_FIM(est_trajectory)

        q, r = qr(FIM)

        obj = -trace(log(r))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = True) # for some reason limited mem works better for the MPC
        return solver


    # run optimisation
    u0 = (input_bounds[:,1] + input_bounds[:,0])/2
    u_solver = get_full_u_solver()
    sol = u_solver(x0=u0, lbx = [input_bounds[0][0]]*n_controlled_inputs*N_control_intervals, ubx = [input_bounds[0][1]]*n_controlled_inputs*N_control_intervals)
    us = sol['x']

    # save results and plot
    np.save(os.path.join(save_path, 'us.npy'), np.array(env.us))


    t = np.arange(N_control_intervals) * int(control_interval_time)


