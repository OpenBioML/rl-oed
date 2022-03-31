import numpy as np
from casadi import *

def xdot(sym_y, sym_theta, sym_u):
    a, Kt, Krt, d, b = [sym_theta[i] for i in range(sym_theta.size()[0])] #intrinsic parameters
    #a = 20min^-1
    Kr = 40 # practically unidentifiable
    Km = 750
    #Kt = 5e5
    #Krt = 1.09e9
    #d = 2.57e-4 um^-3min^-1
    #b = 4 min-1
    #Km = 750 um^-3

    u = sym_u[0] # for now just choose u
    lam = 0.03465735902799726 #min^-1 GROWTH RATE
    #lam = 0.006931471805599453
    #lam = sym_u[1]

    C = 40
    D = 20
    V0 = 0.28
    V = V0*np.exp((C+D)*lam) #eq 2
    G =1/(lam*C) *(np.exp((C+D)*lam) - np.exp(D*lam)) #eq 3

    l_ori = 0.26 # chose this so that g matched values for table 2 for both growth rates as couldnt find it defined in paper

    g = np.exp( (C+D-l_ori*C)*lam)#eq 4

    rho = 0.55
    k_pr = -6.47
    TH_pr0 = 0.65

    k_p = 0.3
    TH_p0 = 0.0074
    m_rnap = 6.3e-7

    k_a = -9.3
    TH_a0 = 0.59

    Pa = rho*V0/m_rnap *(k_a*lam + TH_a0) * (k_p*lam + TH_p0) * (k_pr*lam + TH_pr0) *np.exp((C+D)*lam) #eq 10

    k_r = 5.48
    TH_r0 = 0.03
    m_rib = 1.57e-6
    Rtot = (k_r*lam + TH_r0) * (k_pr*lam + TH_pr0)*(rho*V0*np.exp((C+D)*lam))/m_rib

    TH_f = 0.1
    Rf = TH_f*Rtot #eq 17
    n = 5e6
    eta = 900  # um^-3min^-1


    rna, prot = sym_y[0], sym_y[1]

    rna_dot = a*(g/V)*(  (Pa/(n*G)*Kr + (Pa*Krt*u)/(n*G)**2)  /  (1 + (Pa/n*G)*Kr + (Kt/(n*G) + Pa*Krt/(n*G)**2) *u )) - d*eta*rna/V

    prot_dot = ((b*Rf/V) / (Km + Rf/V))  * rna/V - lam*prot/V

    xdot = SX.sym('xdot', 2)

    xdot[0] = rna_dot
    xdot[1] = prot_dot

    return xdot