from casadi import *
import numpy as np

import math
import time


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class OED_env():
    '''
       Class for OED for time course experiments on systems governed by differential equations
    '''
    def __init__(self, x0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser):
        '''
        initialises the environment
        :param x0: initial system state
        :param xdot: the governig differential equations
        :param param_guesses: inital parameter guesses
        :param actual_params: the actual system params
        :param n_observed_variables: number of variables that can be measured
        :param n_controlled_inputs: number of inputs that can be controlled
        :param num_inputs: number of discrete inputs for FQ
        :param input_bounds: the minimum and maximmum inputs
        :param dt: timestep for RK4 simulation
        :param control_interval_time: time between each control input
        :param normaliser: the normaliser for the RL observation
        '''

        # build the reinforcement learning state

        self.n_system_variables = len(x0)
        self.FIMs = []
        self.detFIMs = []
        self.logdetFIMs = [] # so we dont have to multiply large eignvalues
        self.n_sensitivities = []

        self.dt = dt
        self.control_interval_time = control_interval_time
        self.n_observed_variables = n_observed_variables
        self.initial_params = param_guesses
        self.param_guesses = param_guesses
        self.n_params = len(self.param_guesses.elements())
        self.n_sensitivities = self.n_observed_variables * self.n_params
        self.n_FIM_elements = sum(range(self.n_params+1))
        self.n_tot = self.n_system_variables + self.n_sensitivities + self.n_FIM_elements
        print(self.n_params, self.n_sensitivities, self.n_FIM_elements)
        print('n fim: ', self.n_FIM_elements)
        print('n_tot: ', self.n_tot)
        print('n_sense: ', self.n_sensitivities)
        self.x0 = x0
        self.n_controlled_inputs = n_controlled_inputs
        self.normaliser = normaliser
        self.initial_Y = DM([0] * (self.n_tot))
        self.initial_Y[0:len(x0)] = x0
        self.Y = self.initial_Y

        #TODO: remove t his as too much memory
        self.Ys = [self.initial_Y.elements()]
        self.xdot = xdot # f(x, u, params)
        self.all_param_guesses = []
        self.all_RL_states = []
        self.us = []
        self.actual_params = actual_params
        self.num_inputs = num_inputs
        self.input_bounds = np.array(input_bounds)
        self.current_tstep = 0 # to keep track of time in parallel
        #self.CI_solver = self.get_control_interval_solver(control_interval_time, dt)

    def reset(self, partial = False):

        '''
        resets the environment between episodes
        :param partial: only reset the FIM elements
        :return:
        '''
        self.param_guesses = self.initial_params
        if partial:
            for i in range(self.Y[self.n_system_variables:, :].size()[1]):
                self.Y[self.n_system_variables:, i] = self.initial_Y[self.n_system_variables:]
        else:
            self.Y = self.initial_Y
        self.FIMs = []
        self.detFIMs = []
        self.logdetFIMs =[]
        self.us = []
        self.true_trajectory = []
        self.est_trajectory = []
        self.current_tstep = 0

    def G(self, Y, theta, u):
        '''
        Uses the system equations to setup the full derivatives of system variables plus the FIM
        :param Y: system state
        :param theta: parameters
        :param u: inputs
        :return RHS: the full system of derivatives
        '''


        RHS = SX.sym('RHS', len(Y.elements()))

        # xdot = (sym_theta[0] * sym_u/(sym_theta[1] + sym_u))*sym_Y[0]

        dx = self.xdot(Y, theta,u)


        sensitivities_dot = jacobian(dx[0:self.n_observed_variables], theta) + mtimes(jacobian(dx[0:self.n_observed_variables], Y[0:self.n_observed_variables]), jacobian(Y[0:self.n_observed_variables], theta))

        #TODO: dont need this as parameters not dimensioned and helps FIM not become nan

        for i in range(sensitivities_dot.size()[0]):  # logarithmic sensitivities
            sensitivities_dot[i, :] *= (fabs(theta.T)+1e-5) # absolute value becuase we have negative params



        std = 0.05 * Y[0:self.n_observed_variables]  # to stop divde by zero when conc = 0


        inv_sigma = SX.sym('sig', self.n_observed_variables, self.n_observed_variables)  # sigma matrix in Nates paper

        for i in range(self.n_observed_variables):
            for j in range(self.n_observed_variables):

                if i == j:
                    inv_sigma[i, j] = 1/(std[i] * Y[i])
                else:
                    inv_sigma[i, j] = 0

        sensitivities = reshape(Y[self.n_system_variables:self.n_system_variables + self.n_params *self.n_observed_variables],
                                (self.n_observed_variables, self.n_params))
        FIM_dot = mtimes(transpose(sensitivities), mtimes(inv_sigma, sensitivities))
        FIM_dot = self.get_unique_elements(FIM_dot)

        RHS[0:self.n_system_variables] = dx
        sensitivities_dot = reshape(sensitivities_dot, (sensitivities_dot.size(1) * sensitivities_dot.size(2), 1))
        RHS[self.n_system_variables:self.n_system_variables + self.n_sensitivities] = sensitivities_dot

        RHS[self.n_system_variables + self.n_sensitivities:] = FIM_dot

        return RHS

    def get_one_step_RK(self, theta, u, dt, mode = 'OED'):
        '''
        create the function that performs one step of RK4
        :param theta: parameters
        :param u: inputs
        :param dt: timestep
        :param mode: switch between OED and just simulating the system with no FIM
        :return G_1: the casadi function that performs one step of RK4
        '''

        if mode == 'OED':
            Y = SX.sym('Y', self.n_tot)
            RHS = self.G(Y, theta, u)
        else:
            Y = SX.sym('Y', self.n_system_variables)
            RHS = self.xdot(Y, theta,u)


        g = Function('g', [Y, theta, u], [RHS])

        Y_input = SX.sym('Y_input', RHS.shape[0])

        k1 = g(Y_input, theta, u)


        k2 = g(Y_input + dt / 2.0 * k1, theta, u)
        k3 = g(Y_input + dt / 2.0 * k2, theta, u)
        k4 = g(Y_input + dt * k3, theta, u)

        Y_output = Y_input + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        G_1 = Function('G_1', [Y_input, theta, u], [Y_output])
        return G_1

    def get_control_interval_solver(self, control_interval_time, dt, mode = 'OED'):
        '''
        creates the function that performs simulation of one control interval
        :param control_interval_time:
        :param dt: finite different timestep
        :param mode: switch between OED and just simulation without the FIM
        :return G: function that simulates a single control interval
        '''

        # TODO: try mapaccum in here to reduce memory usage

        #theta = SX.sym('theta', len(self.actual_params.elements())) used for the chemostat OED

        theta = SX.sym('theta', self.actual_params.size())

        u = SX.sym('u', self.n_controlled_inputs)

        G_1 = self.get_one_step_RK(theta, u, dt, mode = mode)  # pass theta and u in just in case#


        if mode == 'OED':
            Y_0 = SX.sym('Y_0', self.n_tot)
        else:
            Y_0 = SX.sym('Y_0', self.n_system_variables)
        Y_iter = Y_0


        for i in range(int(control_interval_time / dt)):
            Y_iter = G_1(Y_iter, theta, u)

        G = Function('G', [Y_0, theta, u], [Y_iter])

        #G = G_1.mapaccum('control_interval', int(control_interval_time / dt)) # should use less memory than the for loop.. This messes up the shap of action inputs
        return G

    def get_sampled_trajectory_solver(self, N_control_intervals, control_interval_time, dt, mode = 'OED'):
        '''
        simulates a whole experiment returns the observed measurements at the control intervals
        :param N_control_intervals: number of control inputs in hte experiment
        :param control_interval_time: time between control inputs
        :param dt: finite difference timestep
        :param mode: switch between OED and just simulation without the FIM
        :return trajectory_solver: the casadi function that performs the simulation
        '''
        #CI_solver = self.get_control_interval_solver(control_interval_time, dt, mode = mode)

        #opt = {'base':1}
        trajectory_solver = self.CI_solver.mapaccum('trajectory', N_control_intervals)

        return trajectory_solver

    def get_full_trajectory_solver(self,  N_control_intervals, control_interval_time, dt):
        '''
        simulates a whole experiment returns the full trajectory
        :param N_control_intervals: number of control inputs in hte experiment
        :param control_interval_time: time between control inputs
        :param dt: finite difference timestep
        :param mode: switch between OED and just simulation without the FIM
        :return trajectory_solver: the casadi function that performs the simulation
        '''
        # need to expand the us before putting into this solver
        theta = SX.sym('theta', len(self.actual_params.elements()))
        u = SX.sym('u', self.n_controlled_inputs)
        G = self.get_one_step_RK(theta, u, dt)

        trajectory_solver = G.mapaccum('trajectory', int(N_control_intervals * control_interval_time / dt))
        return trajectory_solver

    def gauss_newton(self, e,nlp,V, max_iter = 3000, limited_mem = False):
        '''
        creates a gauss newton solver
        :param e: objective to minimise
        :param nlp: the non-linear program
        :param V: the inputs to optimise wrt
        :param max_iter: maximmum bumber of iterations
        :param limited_mem: an approximation to reduce memory usage
        :return: solver
        '''
        J = jacobian(e,V)
        print('jacobian init')
        H = triu(mtimes(J.T, J))
        print('hessian init')
        sigma = SX.sym("sigma")
        hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                       ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                       dict(jit=False, compiler='clang', verbose = False))
        print('hesslag init')

        #IPOPT options https://coin-or.github.io/Ipopt/OPTIONS.html
        #return nlpsol("solver","ipopt", nlp, dict(ipopt={'max_iter':20}, hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

        # using the limited memory hessian approximation for ipopt seems to make it unstable
        ipopt_opt = {'max_iter': max_iter}
        if limited_mem:
            ipopt_opt['hessian_approximation'] = 'limited-memory'
        return nlpsol("solver","ipopt", nlp, dict(ipopt = ipopt_opt, hess_lag=hessLag, jit=False, compiler='clang', verbose_init=False, verbose=False))
        #'acceptable_tol':10, 'acceptable_iter':30,'s_max':1e10,  'obj_scaling_factor': 1e5
        #return nlpsol("solver","ipopt", nlp, dict(ipopt={'hessian_approximation':'limited_memory'}))



    def get_u_solver(self):
        '''
        optimises the next input to maximise the det(FIM)
        :return: solver
        '''



        u = SX.sym('u', self.n_controlled_inputs)
        trajectory_solver = self.get_sampled_trajectory_solver(len(self.us) + 1, self.control_interval_time, self.dt)
        # self.past_trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us))

        all_us = SX.sym('all_us', (len(self.us) + 1, self.n_controlled_inputs))

        print('all', all_us.shape)
        print('us',self.us)
        all_us[0: len(self.us), :] = np.array(self.us).reshape(-1, self.n_controlled_inputs)
        all_us[-1, :] = u

        est_trajectory = trajectory_solver(self.initial_Y, self.param_guesses, transpose(all_us))

        FIM = self.get_FIM(est_trajectory)

        # past_trajectory = self.past_trajectory_solver(self.initial_Y, self.us, self.param_guesses)
        # current_FIM = self.get_FIM(past_trajectory)

        q,r = qr(FIM)

        obj = -trace(log(r))
        #obj = -log(det(FIM))
        nlp = {'x': u, 'f': obj}
        solver = self.gauss_newton(obj, nlp, u)
        #solver.print_options()
        #sys.exit()

        return solver  # , current_FIM

    def get_param_solver(self, trajectory_solver, test_trajectory=None, initial_Y = None):
        '''
        creates the solver to fit the params
        :param trajectory_solver: the solver for the trajectory given the params
        :param test_trajectory: the true trajectory
        :return: parameter solver
        '''
        sym_theta = SX.sym('theta', len(self.param_guesses.elements()))

        if initial_Y is None:
            initial_Y = self.initial_Y

        if test_trajectory is None:
            trajectory = trajectory_solver(DM(initial_Y), self.actual_params, np.array(self.us).T, mode = 'param')

            print('p did:', trajectory.shape)
        else:
            trajectory = test_trajectory
            print('p did:', trajectory.shape)

        est_trajectory_sym = trajectory_solver(DM(initial_Y), sym_theta, np.array(self.us).T)
        print('sym trajectory initialised')
        print('sym traj:', est_trajectory_sym.shape)
        print('traj:', trajectory.shape)

        e = trajectory[0:self.n_observed_variables, :].T - est_trajectory_sym[0:self.n_observed_variables, :].T
        print('e shape:', e.shape)
        print(dot(e, e).shape)

        nlp = {'x': sym_theta, 'f': 0.5 * dot(e / (0.05 * trajectory[0:self.n_observed_variables, :].T + 0.00000001),
                                              e)}  # weighted least squares
        print('nlp initialised')
        #solver = self.gauss_newton(e, nlp, sym_theta, max_iter = 100000)
        solver = self.gauss_newton(e, nlp, sym_theta)
        print('solver initialised')


        return solver

    def step(self, action = None, continuous = True, use_old_state = False):
        '''
        performs one RL ste
        :param action:
        :param continuous:
        :param use_old_state:
        :return: state, action, reward, done
        '''

        self.current_tstep += 1
        if action is None: # Traditional OED step
            u_solver = self.get_u_solver()
            #u = u_solver(x0=self.u0, lbx = 10**self.input_bounds[0], ubx = 10**self.input_bounds[1])['x']
            u = u_solver(x0=self.u0, lbx = self.input_bounds[:,0], ubx = self.input_bounds[:,1])['x']
            self.us.append(u.elements())
        else: #RL step
            if continuous:
                u = action
            else:
                u = self.action_to_input(action).T
            #self.us.append(10**u)
            self.us.append(u)
        N_control_intervals = len(self.us)
        #N_control_intervals = 12
        sampled_trajectory_solver = self.get_sampled_trajectory_solver(N_control_intervals, self.control_interval_time, self.dt) # the sampled trajectory seen by the agent


        #trajectory_solver = self.get_full_trajectory_solver(N_control_intervals, control_interval_time, self.dt) # the true trajectory of the system
        #trajectory_solver = trajectory_solver(N_control_intervals, control_interval_time, dt ) #this si the symbolic trajectory
        t = time.time()
        self.true_trajectory = sampled_trajectory_solver(self.initial_Y,  self.actual_params, np.array(self.us).T)
        #self.est_trajectory = sampled_trajectory_solver(self.initial_Y, self.param_guesses, self.us )

        #param_solver = self.get_param_solver(sampled_trajectory_solver)
        # estimate params based on whole trajectory so far
        #disablePrint()
        #self.param_guesses = param_solver(x0=self.param_guesses, lbx = 0)['x']
        #enablePrint()
        #self.all_param_guesses.append(self.param_guesses.elements())

        #reward = self.get_reward(self.est_trajectory)

        reward = self.get_reward(self.true_trajectory)

        done = False

        #state = self.get_RL_state(self.true_trajectory, self.est_trajectory)

        state = self.get_RL_state(self.true_trajectory, self.true_trajectory, use_old_state = use_old_state)



        self.all_RL_states.append(state)
        return state, reward, done, None


    def get_reward(self, est_trajectory):
        '''
        calculates the reward for an RL agent
        :param est_trajectory:
        :return: reward
        '''
        FIM = self.get_FIM(est_trajectory)

        #use this method to remove the small negatvie eigenvalues

        # casadi QR seems better,gives same results as np but some -ves in different places and never gives -ve determinant
        q, r = qr(FIM)

        det_FIM = np.prod(diag(r).elements())

        logdet_FIM = trace(log(r)).elements()[0] # do it like this to protect from numerical errors from multiplying large EVs

        if det_FIM <= 0:
            print('----------------------------------------smaller than 0')
            eigs = np.real(np.linalg.eig(FIM)[0])
            eigs[eigs<0] = 0.00000000000000000000000001
            det_FIM = np.prod(eigs)
            logdet_FIM = np.log(det_FIM)

        self.FIMs.append(FIM)
        self.detFIMs.append(det_FIM)
        self.logdetFIMs.append(logdet_FIM)

        try:
            #reward = np.log(det_FIM-self.detFIMs[-2])
            reward = logdet_FIM - self.logdetFIMs[-2]
            #print('det adfa: ', det_FIM)
            #print(det_FIM - self.detFIMs[-2])
        except:

            reward = logdet_FIM

        if math.isnan(reward):
            pass
            print()
            print('nan reward, FIM might have negative determinant !!!!')

            reward = -100
        return reward/100


    def action_to_input(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        :param action: the descrete action
        :returns:action
        '''

        # calculate which bucket each eaction belongs in

        buckets = np.unravel_index(action, [self.num_inputs] *self.n_controlled_inputs)

        # convert each bucket to a continuous state variable
        Cin = []
        for r in buckets:
            Cin.append(self.input_bounds[0] + r*(self.input_bounds[1]-self.input_bounds[0])/(self.num_inputs-1))

        Cin = np.array(Cin).reshape(-1,1)

        return np.clip(Cin, self.input_bounds[0], self.input_bounds[1])


    def get_FIM(self, trajectory):
        '''
        assembles the FIM from an experimental trajectory
        :param trajectory:
        :return: FIM
        '''

        # Tested on 2x2 and 5x5 matrices
        FIM_start = self.n_system_variables + self.n_sensitivities

        FIM_end = FIM_start + self.n_FIM_elements

        FIM_elements = trajectory[FIM_start:FIM_end, -1]

        start = 0
        end = self.n_params
        # FIM_elements = np.array([11,12,13,14,15,22,23,24,25,33,34,35,44,45,55]) for testing
        FIM = reshape(FIM_elements[start:end], (1, self.n_params))  # the first row

        for i in range(1, self.n_params):  # for each row
            start = end
            end = start + self.n_params - i

            # get the first n_params - i elements
            row = FIM_elements[start:end]

            # get the other i elements

            for j in range(i - 1, -1, -1):
                row = horzcat(FIM[j, i], reshape(row, (1, -1)))

            reshape(row, (1, self.n_params))  # turn to row ector

            FIM = vertcat(FIM, row)

        #sys.exit()
        return FIM

    def get_unique_elements(self, FIM):
        '''
        gets the unique elements of the FIM
        :param FIM:
        :return: unique elements
        '''

        n_unique_els = sum(range(self.n_params + 1))

        UE = SX.sym('UE', n_unique_els)
        start = 0
        end = self.n_params
        for i in range(self.n_params):
            UE[start:end] = transpose(FIM[i, i:])
            start = end
            end += self.n_params - i - 1

        return UE

    def normalise_RL_state(self, state):
        #print(state)

        return state / self.normaliser

    def get_RL_state(self, true_trajectory, est_trajectory, use_old_state = False, use_time = True):
        '''
        from a trajectory assemble the RL state
        :param true_trajectory: experimetnal trajectory
        :param est_trajectory: estimated trajectory, if not doing iterative inference this should be true_trajectory
        :param use_old_state:
        :param use_time:
        :return: RL state
        '''

        # get the current measured system state
        sys_state = true_trajectory[:self.n_observed_variables, -1]  # TODO: measurement noise

        if use_old_state:
            state = sys_state
        else:
            state = np.sqrt(sys_state)
        # get current fim elements
        FIM_start = self.n_system_variables + self.n_sensitivities

        FIM_end = FIM_start + self.n_FIM_elements

        # FIM_elements = true_trajectory[FIM_start:FIM_end]
        FIM_elements = est_trajectory[FIM_start:FIM_end, -1]

        FIM_signs = np.sign(FIM_elements)
        FIM_elements = FIM_signs * sqrt(fabs(FIM_elements))

        if use_old_state:
            state = np.append(sys_state, np.append(self.param_guesses, FIM_elements))

        if use_time:
            state = np.append(state, self.current_tstep)
        else:
            state = np.append(state, 0)

            #state = np.append(state, self.logdetFIMs[-1])

        return self.normalise_RL_state(state)

    def get_initial_RL_state(self, use_old_state = False):
        '''
        create the initial RL state for the beginning of an episode
        :param use_old_state:
        :return: initial state
        '''
        if use_old_state:
            state = np.array(list(np.sqrt(self.x0[0:self.n_observed_variables])) + self.param_guesses.elements() + [0] * self.n_FIM_elements)
        else:
            state = np.array(list(np.sqrt(self.x0[0:self.n_observed_variables])))
        state = np.append(state, 0) #time
        #state = np.append(state, 0) #logdetFIM

        return self.normalise_RL_state(state)

    def get_initial_RL_state_parallel(self, o0 = None, use_old_state = False, i = 0):
        '''
        create the initial RL state for the beginning of an episode
        :param o0: inital state of the system observables
        :param use_old_state:
        :param i: the index of the parallel experiment
        :return: initial RL state
        '''

        #state = np.array(list(np.sqrt(self.x0[0:self.n_observed_variables])) + self.param_guesses[i,:].elements() + [0] * self.n_FIM_elements)
        if o0 is None:
            o0 = self.x0


        if use_old_state:
            state = np.array(list(np.sqrt(o0[0:self.n_observed_variables])) + self.param_guesses[i,:].elements() + [
                0] * self.n_FIM_elements)
        else:
            state = np.array(list(np.sqrt(o0[0:self.n_observed_variables])))


        state = np.append(state, 0) #time
        #state = np.append(state, 0) #logdetFIM

        return self.normalise_RL_state(state)

    def map_parallel_step(self, actions, actual_params, continuous = False, Ds = False, use_old_state = False, use_time = True):
        '''
        runs step in parrallel using casadi map function
        :param actions: actions for all the parallel experiments
        :param actual_params: the parameters for each expreiment
        :param continuous:
        :param Ds: use Ds design
        :param use_old_state:
        :param use_time: add time to the state
        :return: the transitions of all parallel experiments
        '''
        self.current_tstep += 1
        # actions, actual_params = args

        # all_us = []
        # for As in actions:
        # us = [self.action_to_input(action) for action in actions]
        if not continuous:
            us = self.actions_to_inputs(actions)
        else:
            us = self.input_bounds[:, 0].reshape(-1, 1) + (self.input_bounds[:,1] - self.input_bounds[:, 0]).reshape(-1, 1)*actions


        actual_params = DM(actual_params)

        N_control_intervals = len(us)

        # set sampled trajectory solver in script to ensure thread safety
        true_trajectories = self.mapped_trajectory_solver(self.Y, actual_params.T, np.array(us))
        transitions = []
        t = time.time()

        for i in range(true_trajectories.shape[1]):
            true_trajectory = true_trajectories[:, i]
            reward = self.get_reward_parallel(true_trajectory, i, Ds = Ds)

            done = False

            # state = self.get_RL_state(self.true_trajectory, self.est_trajectory)
            state = self.get_RL_state_parallel(true_trajectory, true_trajectory,i, use_old_state = use_old_state, use_time=use_time)


            transitions.append((state, reward, done, None, us[:,i]))

        self.Y = true_trajectories
        self.Ys.append(self.Y.elements())
        return transitions

    def actions_to_inputs(self, actions):
        '''
        PARALLEL action to input

        Takes a discrete action index and returns the corresponding continuous state
        vector

        :param action: the descrete action
        :returns:action
        '''

        # calculate which bucket each eaction belongs in

        buckets = np.unravel_index(actions, [self.num_inputs] * self.n_controlled_inputs)
        buckets = np.array(buckets)
        # convert each bucket to a continuous state variable
        # TODO: make this work with multiple different input doounds
        Cin = self.input_bounds[0][0] + buckets * (self.input_bounds[0][1] - self.input_bounds[0][0]) / (self.num_inputs - 1)

        return np.clip(Cin, self.input_bounds[0][0], self.input_bounds[0][1])

    def get_reward_parallel(self, est_trajectory, i, Ds = False):
        '''
        parrallel get reward
        :param est_trajectory:
        :param i: parallel index
        :param Ds: use Ds optimality
        :return: reward
        '''
        FIM = self.get_FIM(est_trajectory)
        if Ds: # partition FIM and get determinant of the params we are interested in (the elements of LV matrix)
            M11 = FIM[0:-4, 0:-4]
            q11, r11 = qr(M11)

            logdet_M11 = trace(log(r11)).elements()[0]  # do it like this to protect from numerical errors from multiplying large EVs
            q, r = qr(FIM)
            det_FIM = np.prod(diag(r).elements())
            logdet_FIM = trace(log(r)).elements()[0]  # do it like this to protect from numerical errors from multiplying large EVs
            logdet_FIM -= logdet_M11
        else:
            # use this method to remove the small negatvie eigenvalues

            # casadi QR seems better,gives same results as np but some -ves in different places and never gives -ve determinant
            q, r = qr(FIM)
            det_FIM = np.prod(diag(r).elements())
            logdet_FIM = trace(log(r)).elements()[0]  # do it like this to protect from numerical errors from multiplying large EVs
            if det_FIM <= 0:
                print('----------------------------------------smaller than 0')
                eigs = np.real(np.linalg.eig(FIM)[0])
                eigs[eigs < 0] = 0.00000000000000000000000001
                det_FIM = np.prod(eigs)
                logdet_FIM = np.log(det_FIM)

                print(det_FIM)

        self.detFIMs[i].append(det_FIM)
        self.logdetFIMs[i].append(logdet_FIM)

        try:
            reward = logdet_FIM - self.logdetFIMs[i][-2]

        except:

            reward = logdet_FIM

        if math.isnan(reward):
            pass
            print()
            print('nan reward, FIM might have negative determinant !!!!')

            reward = -100

        return reward/100

    def get_RL_state_parallel(self, true_trajectory, est_trajectory,i, use_old_state = False, use_time = True):
        '''

        parallel get Rl state
        :param true_trajectory:
        :param est_trajectory:
        :param i: parallel index
        :param use_old_state:
        :param use_time:
        :return: state
        '''

        # get the current measured system state


        sys_state = true_trajectory[:self.n_observed_variables, -1]  # TODO: measurement noise


        if use_old_state:
            state = np.sqrt(sys_state)
        else:
            state = np.sqrt(sys_state)


        # get current fim elements
        FIM_start = self.n_system_variables + self.n_sensitivities

        FIM_end = FIM_start + self.n_FIM_elements

        # FIM_elements = true_trajectory[FIM_start:FIM_end]
        FIM_elements = est_trajectory[FIM_start:FIM_end, -1]

        FIM_signs = np.sign(FIM_elements)
        FIM_elements = FIM_signs * sqrt(fabs(FIM_elements))



        if use_old_state:
            state = np.append(state, np.append(self.param_guesses[i,:], FIM_elements))


        if use_time:
            state = np.append(state, self.current_tstep)
        else:
            state = np.append(state, 0)


        #state = np.append(state, self.logdetFIMs[i][-1])


        return self.normalise_RL_state(state)

