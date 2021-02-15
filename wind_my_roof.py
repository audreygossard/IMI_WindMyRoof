import numpy as np
from matplotlib import pyplot as plt

_J = 10
_C_r = 1.5
_R = 0.53
_k = 0.9
_V = 10
_C_e = 20
_lamb = 1

om_theo = (_lamb*_V)/_R - 1/_R * np.sqrt((_C_r + _C_e)/_k)
# on écrit l'équation sous la forme u'(t) = f(t, u(t))
# dans notre cas, u = omega

def f(t, u):
    C_v = _k*np.sign(_lamb*_V - _R*u)*(_lamb*_V - _R*u)**2
    C_e = _C_e
    C_r = _C_r
    return (C_v - C_e - C_r)/_J

def euler(fun, t_ini, t_fin, N_samples, u_origin):

    dt = (t_fin - t_ini)/N_samples
    t_array = np.linspace(t_ini, t_fin, N_samples)
    om_array = np.zeros(N_samples)
    om_array[0] = u_origin
    for n in range(0, N_samples-1):
        cur_time = t_array[n]
        om_array[n+1] = om_array[n] + dt*f(cur_time, om_array[n])


    V_array = np.array([_V]*N_samples)
    V_array[0] = 0
    C_e_array = np.array([_C_e]*N_samples)
    C_e_array[0] = 0
    P_array = om_array*_C_e

    return t_array, V_array, om_array, C_e_array, P_array


def display(t_array, V_array, om_array, C_e_array, P_array):
    fig, axs = plt.subplots(4)
    # ===========| Plot Vitesse V |================
    axs[0].plot(t_array, V_array)
    axs[0].set_xlabel("temps (s)")
    axs[0].set_ylabel("Vitesse (m/s)")
    # ===========| Plot omega |================
    axs[1].plot(t_array, om_array)
    axs[1].set_xlabel("temps (s)")
    axs[1].set_ylabel("omega")
    # ===========| Plot C_e |================
    axs[2].plot(t_array, C_e_array)
    axs[2].set_xlabel("temps (s)")
    axs[2].set_ylabel("C_e (N.m)")
    # ===========| Plot Puissance |================
    axs[3].plot(t_array, om_array*_C_e)
    axs[3].set_xlabel("temps (s)")
    axs[3].set_ylabel("Puissance (W)")

    # pour enlever l'abcisse sur les graphes supérieurs
    # for ax in axs.flat:
    #     ax.label_outer()

    fig.tight_layout()
    plt.show()


def compute_C_e_opt_theo(C_r, lamb, V, k):
    first_comp = 3*C_r + (lamb*V)**2 * k
    second_comp = lamb*V*np.sqrt(k*(k*(lamb*V)**2 - 3*C_r))
    return 2/9*(first_comp + second_comp)



# ============| Affichage des données |=================
t, v, o, c, p = euler(f, 0, 20, 1000, 0)
print(f"La valeur optimale expérimentale de omega est : {o[-1]:.3f}\nCelle de P est : {p[-1]:.3f}")
print(f"La valeur optimale théorique de omega est : {om_theo:.3f}")
print(f"Pourcentage d'erreur entre valeur théorique et valeur exp : {abs(o[-1]- om_theo)*100:.2f}")
display(t,v,o,c,p)