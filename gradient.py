import numpy as np
from parameters import *


def f(u, J, C_r, R, k, V, C_e, lamb):
    C_v = k*np.sign(lamb*V - R*u)*(lamb*V - R*u)**2
    return (C_v - C_e - C_r)/J

# Gradient descent algorithm, it also simulates omega between each iteration since the system is dynamic
def grad(CeIni, omIni, relaxTime, step, eps,
         fun, dt,
         V, J=_J, C_r=_C_r, R=_R, k=_k, lamb=_lamb):

    # Initializing C_e and omega
    Ce = CeIni
    omega = omIni

    # Computing the relaxation of omega towards its equilibrium
    def relax(omega, Ce, relaxTime):
        for t in range(int(relaxTime / dt)):
            omega = omega + dt * fun(omega, J, C_r, R, k, V, Ce, lamb)
        if (omega < 0):
            omega = 0
        return omega

    # Computing a new step of C_e knowing which direction should be taken
    def nextStep(Ce, positiveStep):
        if positiveStep:
            return Ce + step
        else:
            return Ce - step

    # Waiting for omega to reach its equilibrium
    omega = relax(omega, Ce, relaxTime)

    # Computing a step to the right (we're gonna check if this is the right direction)
    P = Ce * omega
    Ce = Ce + step

    # Looking for the gradient's direction
    # Since P is concave, we only need to do this once and then keep the same direction
    if relax(omega, Ce, relaxTime) * Ce > P:
        # Updating omega
        omega = relax(omega, Ce, relaxTime)
        # Steps should be done in the positive direction
        positiveStep = True

    else:
        # Going back to the intial value of C_e
        Ce = Ce - step
        # Steps should be done in the negative direction
        positiveStep = False

    # Gradient iterations
    while abs(Ce * omega - nextStep(Ce, positiveStep) * relax(omega, nextStep(Ce, positiveStep), relaxTime)) > eps:
        Ce = nextStep(Ce, positiveStep)
        omega = relax(omega, Ce, relaxTime)

    return Ce, omega
