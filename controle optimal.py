from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


#parametres:
_J = 10
_C_r = 1.5
_R = 0.53
_k = 0.9
_V = 10
_lamb = 1

_T = 20
t_init = 0


def omega(t, C_e, omega0, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    sa = R/J*np.sqrt(k*(C_e+C_r))  # = sqrt(alpha)
    saz = np.sqrt(k / (C_e + C_r)) * (lamb * V - R * omega0)  # = sqrt(alpha)*z0
    tan1 = np.tan(sa * (t - t_init))
    tanh1 = np.tanh(sa * (t - t_init))
    omega1 = lamb * V / R - np.sqrt((C_r + C_e) / k) / R * (saz + tan1) / (1 - saz * tan1)
    omega2 = lamb * V / R - np.sqrt((C_r + C_e) / k) / R * (saz + tanh1) / (1 + saz * tanh1)
    return max(omega1, omega2)
# probleme calcul de omega: on a considéré que C_e est constant alors qu'ici on le fait varier!


def p(t, C_e, omega0, T = _T, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    sa = R/J*np.sqrt(k*(C_e+C_r))  # = sqrt(alpha)
    saz = np.sqrt(k/(C_e+C_r))*(lamb*V-R*omega0)  # = sqrt(alpha)*z0
    az = R/J*k*(lamb*V-R*omega0)  # = alpha*z0
    tan1 = np.tan(sa*(t-t_init))
    tan2 = np.tan(sa*(T-t_init))
    tanh1 = np.tanh(sa*(t-t_init))
    tanh2 = np.tanh(sa*(T-t_init))
    omega1 = lamb*V/R - np.sqrt((C_r+C_e)/k)/R * (saz+tan1)/(1-saz*tan1)
    omega2 = lamb*V/R - np.sqrt((C_r+C_e)/k)/R * (saz+tanh1)/(1+saz*tanh1)
    omega = max(omega1, omega2)

    if omega <= lamb*V/R:
        q1 = ((1+saz*tanh1)/(1+saz*tanh2))**2
        q2 = (1-tanh2**2)/(1-tanh1**2)
        q3 = -C_e/az * (1+saz*tanh2)/(1-tanh2**2)
        q4 = -(1+saz*tanh2)/(1+saz*tanh1) + 1
        return q1 * q2 * q3 * q4

    else:
        q1 = ((1-saz*tan1)/(1-saz*tan2))**2
        q2 = (1+tan2**2)/(1+tan1**2)
        q3 = -C_e/az * (1-saz*tan2)/(1+tan2**2)
        q4 = (1-saz*tan2)/(1-saz*tan1) - 1
        return q1 * q2 * q3 * q4


def f(omega, C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    C_v = k*np.sign(lamb*V - R*omega)*(lamb*V - R*omega)**2
    return (C_v - C_e - C_r)/J


def g(omega, C_e):
    return omega*C_e


def H(omega, p, C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    return p*f(omega, C_e, J, C_r, R, k, V, lamb) + g(omega, C_e)


def argmin_H(C_e, t, omega0, T = _T, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    return H(omega(t, C_e, omega0, J, C_r, R, k, V, lamb),
             p(t, C_e, omega0, T, J, C_r, R, k, V, lamb),
             C_e,
             J, C_r, R, k, V, lamb)


# --- TEST ---

# on se place dans un cas où au départ C_e(0) = 0

# d'où l'expression de omega0:
omega0 = _lamb*_V/_R - np.sqrt(_C_r/_k)
print("omega0 = ", omega0)

# à l'équilibre (t=T doit être suffisamment long) on doit trouver C_e_opt:
C_e_opt = 2/9 * (3*_C_r + (_lamb*_V)**2 * _k + _lamb*_V*np.sqrt(_k*((_lamb*_V)**2 * _k -3*_C_r)))
print("C_e_opt = ", C_e_opt)
omega_opt = _lamb*_V/_R - np.sqrt((_C_r + C_e_opt)/_k)
print("omega_opt = ", omega_opt)

delta_t = 0.2
time_n = 100
time_tab = np.array([i*delta_t for i in range(1, time_n+1)])
C_e_tab = np.zeros(time_n)

for i, t in enumerate(time_tab):
    res = minimize(argmin_H, [10], args=(t, omega0), constraints=({'type': 'ineq', 'fun': lambda x:  x}))
    C_e_tab[i] = res.x
    if i%10 == 0:
        print("C_e = ", C_e_tab[i])
        print("omega = ", omega(t, C_e_tab[i], omega0))

plt.plot(time_tab, C_e_tab)
plt.show()

# normalement on devrait avoir d'abord C_e très grand puis C_e = C_e_opt
# ici C_e reste à zero (pourquoi?)
# probleme: si on met C_e trop grand l'éolienne tourne à l'envers et le système s'emballe
# on peut borner C_e dans minimize?
# autre probleme: si T (temps final atteinte optimum) est trop grand, tanh = 1 et on divise par zero
# comment choisir T pour qu'il soit assez grand pour qu'on atteigne l'optimum, mais pas trop quand même?
