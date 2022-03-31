import sys
import os

IMPORT_PATH = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'RED', 'environments'))
sys.path.append(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'RED', 'environments'), 'chemostat'))
sys.path.append(IMPORT_PATH)




from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
import time


from xdot_chemostat import xdot
import json

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':

    #setup
    params = json.load(open(os.path.join(IMPORT_PATH, 'params_chemostat.json')))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]
    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)
    save_path = './results/'
    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = DM((np.array(ub) + np.array(lb))/2)
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
    explore_rate = 1
    input_bounds = np.array(input_bounds)
    u0 = (input_bounds[:,1] + input_bounds[:,0])/2
    env.u0 = DM(u0)
    e_rewards = []


    #run optimisation
    for e in range(0, N_control_intervals):
        next_state, reward, done, _ = env.step()


        if e == N_control_intervals - 1:
            next_state = [None]*24
            done = True

        e_rewards.append(reward)
        state = next_state

    # save results and plot
    print('return: ', np.sum(e_rewards))

    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))

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


    plt.show()
