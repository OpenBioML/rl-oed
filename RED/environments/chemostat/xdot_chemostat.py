import numpy as np
from casadi import *


def monod(C, C0, umax, Km, Km0):
    '''
    Calculates the growth rate based on the monod equation

    Parameters:
        C: the concetrations of the auxotrophic nutrients for each bacterial
            population
        C0: concentration of the common carbon source
        Rmax: array of the maximum growth rates for each bacteria
        Km: array of the saturation constants for each auxotrophic nutrient
        Km0: array of the saturation constant for the common carbon source for
            each bacterial species
    '''

    # convert to numpy

    growth_rate = ((umax * C) / (Km + C)) * (C0 / (Km0 + C0))

    return growth_rate


def xdot(sym_y, sym_theta, sym_u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        dsol: array of the derivatives for all state variables
    '''

    print(sym_y)

    #q = sym_u[0]
    Cin = sym_u[0]
    C0in = sym_u[1]

    q = 0.5

    #y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]
    #y, y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]
    #y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    umax, Km, Km0 = [sym_theta[i] for i in range(3)]


    y = np.array([480000.])
    y0 = np.array([520000.])


    num_species = Km.size()[0]

    # extract variables
    N = sym_y[0]
    C = sym_y[1]
    C0 = sym_y[2]

    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives

    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / y) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - sum(1 / y0[i] * R[i] * N[i] for i in range(num_species))

    # consstruct derivative vector for odeint

    xdot = SX.sym('xdot', 2*num_species + 1)


    xdot[0] = dN
    xdot[1] = dC
    xdot[2] = dC0


    return xdot

def xdot_scaled(sym_y, sym_theta, sym_u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        dsol: array of the derivatives for all state variables
    '''

    #q = sym_u[0]
    Cin = sym_u[0]
    C0in = sym_u[1]

    q = 0.5

    #y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]
    #y, y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]
    #y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    umax, Km, Km0 = [sym_theta[i] for i in range(3)]


    y = np.array([4.8])
    y0 = np.array([5.2])

    print('params:', y, y0, umax, Km, Km0 )
    num_species = Km.size()[0]
    print('num species:', num_species)

    # extract variables
    N = sym_y[0]
    C = sym_y[1]
    C0 = sym_y[2]
    print(N.shape, C.shape, C0.shape)
    R = monod(C, C0, umax, Km, Km0)
    print(R.shape)
    # calculate derivatives

    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / y) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - sum(1 / y0[i] * R[i] * N[i] for i in range(num_species))

    print(dN.shape, dC.shape, dC0.shape)
    if dC.shape == (2, 2):
        print(q, Cin.shape, C0, C, y, R, N)  # C0in

    # consstruct derivative vector for odeint

    xdot = SX.sym('xdot', 2*num_species + 1)


    xdot[0] = dN
    xdot[1] = dC
    xdot[2] = dC0


    return xdot

