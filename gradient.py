import numpy as np
from parameters import *


def f(u, J, C_r, R, k, V, C_e, lamb):
    C_v = k*np.sign(lamb*V - R*u)*(lamb*V - R*u)**2
    return (C_v - C_e - C_r)/J

def grad(CeIni, omIni, relaxTime, step, eps,
         fun, dt,
         V, J=_J, C_r=_C_r, R=_R, k=_k, lamb=_lamb):
    
    Ce = CeIni
    omega = omIni

    # Computing the relaxation of omega towards its equilibrium
    def relax(omega, Ce, relaxTime):
        for t in range(int(relaxTime / dt)):
            omega = omega + dt * fun(omega, J, C_r, R, k, V, Ce, lamb)
        if (omega < 0):
            omega = 0
        return omega

    # Waiting for omega to reach its equilibrium
    omega = relax(omega, Ce, relaxTime)
    
    # Computing a step to the right (we're gonna check if this is the right direction)
    P =  Ce * omega
    Ce = Ce + step
    
    # Looking for the gradient's direction
    # Since P is concave, we only need to do this once and then keep the same direction
    if relax(omega, Ce, relaxTime) * Ce > P:
        omega = relax(omega, Ce, relaxTime)

        while abs(Ce * omega - (Ce + step) * relax(omega, Ce + step, relaxTime)) > eps:
            Ce = Ce + step
            omega = relax(omega, Ce, relaxTime)

    else:
        Ce = Ce - step

        while abs(Ce * omega - (Ce + step) * relax(omega, Ce - step, relaxTime)) > eps:
            Ce = Ce - step
            omega = relax(omega, Ce, relaxTime)
    
    return Ce, omega
    