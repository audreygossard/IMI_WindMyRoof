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

_T = 5
t_init = 0

_N_samples_t = 1000

"""
def compute_variation_omega_theo(t_ini, t, om_origin, C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    omega_eq = lamb * V/R - 1/R*np.sqrt((C_r + C_e)/k)
    beta = (C_r + C_e)*R/J
    alpha = R*k*beta/J
    z_0 = (lamb*V - R*om_origin)/beta
    coef_cons = lamb*V/R
    coef_c_r = np.sqrt((C_r + C_e)/k)/R
    coef_ponde = np.sqrt(alpha)*z_0
    coef_tanh = np.tanh(np.sqrt(alpha)*(t - t_ini))
    om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)
    coef_tan = np.tan(np.sqrt(alpha)*(t - t_ini))
    om_neg = coef_cons - coef_c_r * (coef_ponde + coef_tan)/(1 - coef_ponde * coef_tan)
    # cas om_pos <= 0 pour accepter les cas où la tangente hyperbolique explose
    if om_neg >= lamb*V/R and (om_pos >= lamb*V/R or om_pos <= 0):
        omega = om_neg
    else:
        if om_origin >= lamb*V/R:
            om_ini = om_origin
            z_0 = (lamb*V - R*om_ini)/beta
            coef_ponde = np.sqrt(alpha)*z_0
            coef_tanh = np.tanh(np.sqrt(alpha)*(t - t_ini))
            om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)
        omega = om_pos
    return omega
"""


# on calcule tjrs omega à partir de la valeur précédente omega0 = omega[i-1] et donc on remet t_init = 0 et t = dt
def omega(t_ini, t_fin, N_samples_t, om_origin, C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    time = np.linspace(t_ini, t_fin, N_samples_t)
    omega = np.zeros(N_samples_t)
    omega[0] = om_origin

    omega_eq = lamb * V/R - 1/R*np.sqrt((C_r + C_e)/k)
    beta = (C_r + C_e)*R/J
    alpha = R*k*beta/J
    z_0 = (lamb*V - R*om_origin)/beta
    om_ori = om_origin

    for n in range(1, N_samples_t):
        coef_cons = lamb*V/R
        coef_c_r = np.sqrt((C_r + C_e)/k)/R
        coef_ponde = np.sqrt(alpha)*z_0

        coef_tanh = np.tanh(np.sqrt(alpha)*(time[n] - t_ini))
        om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)

        coef_tan = np.tan(np.sqrt(alpha)*(time[n] - t_ini))
        om_neg = coef_cons - coef_c_r * (coef_ponde + coef_tan)/(1 - coef_ponde * coef_tan)

        if om_neg >= lamb*V/R and (om_pos >= lamb*V/R or om_pos <= 0):
            omega[n] = om_neg
            continue
        else:
            if (omega[n-1] >= lamb*V/R):
                t_ini = t[n-1]
                om_ini = omega[n-1]
                z_0 = (lamb*V - R*om_ini)/beta
                coef_ponde = np.sqrt(alpha)*z_0
                coef_tanh = np.tanh(np.sqrt(alpha)*(time[n] - t_ini))

                om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)
            omega[n] = om_pos
    return omega[-1]


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
    omega_c = omega(0, t, _N_samples_t, omega0, C_e, J, C_r, R, k, V, lamb)

    if omega_c <= lamb*V/R:
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
    return -omega*C_e


def H(omega, p, C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    return p*f(omega, C_e, J, C_r, R, k, V, lamb) + g(omega, C_e)


def argmin_H(C_e, t, delta_t, omega0, omega_prec, T = _T, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    return H(omega(0, t, _N_samples_t, omega0, C_e, J, C_r, R, k, V, lamb),
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
print("\n")

delta_t = 1e-2
time_n = 500
time_tab = np.array([i*delta_t for i in range(time_n)])
C_e_tab = np.zeros(time_n)
omega_tab = np.zeros(time_n)
omega_tab[0] = omega0

for i in range(1, time_n):
    #print(i)
    t = time_tab[i]
    res = minimize(argmin_H, [50], args=(t, delta_t, omega0, omega_tab[i-1]), constraints=({'type': 'ineq', 'fun': lambda x:  x}, {'type': 'ineq', 'fun': lambda x:  100-x}))
    C_e_tab[i] = res.x
    omega_tab[i] = omega(0, t, _N_samples_t, omega0, C_e_tab[i])
    if i%10 == 0:
        print("C_e = ", C_e_tab[i])
        print("omega = ", omega_tab[i])

plt.plot(time_tab, C_e_tab)
plt.show()
