import numpy as np
from matplotlib import pyplot as plt

_J = 10
_C_r = 1.5
_R = 0.53
_k = 0.9
_V = 10
_C_e = 20
_lamb = 1

eps = 0.05
deltaT = 

om_theo = (_lamb*_V)/_R - 1/_R * np.sqrt((_C_r + _C_e)/_k)

def generate_V_with_noises(N_samples, V = _V, sigma2 = 1):
    V_ini = np.array([V]*N_samples)
    noise = np.random.normal(0, sigma2, N_samples)
    return V_ini + noise

## ===== Implémentation du schéma d'Euler =====
# on écrit l'équation sous la forme u'(t) = f(t, u(t))
# dans notre cas, u = omega

def f(t, u, J, C_r, R, k, V, C_e, lamb):
    C_v = k*np.sign(lamb*V - R*u)*(lamb*V - R*u)**2
    return (C_v - C_e - C_r)/J

def euler(fun, t_ini, t_fin, N_samples, u_origin, V_array,
        J = _J, C_r = _C_r, R = _R,
        k = _k, C_e_array,
        lamb = _lamb):
    """
    Paramètres :
    'fun' (fonction) : the function of the Euler scheme,
    't_ini' (float) : beginning time of the scheme,
    't_fin' (float) : ending time of the scheme,
    'N_samples' (int) : number of time's samples,
    'u_origin' (float) : origin value of 'fun' at time t=t_ini,
    'V_array' (float np.array) : array containing wind's speed. Must be of size 'N_samples',
    'J' (float) : moment of inertia,
    'C_r' (float) : couple ,
    'R' (float) : radius of the wind turbine,
    'k' (float) : coefficient k,
    'C_e_array' (float np.array) : couple. Must be of size 'N_samples',
    'lamb' (float) : coefficient lambda

    Return :
    't_array' (float np.array) : array of time sampled (size : N_samples),
    'V_array' (float np.array) : array of wind's speed sampled (size : N_samples),
    'om_array' (float np.array) : array of rotation speed sampled (size : N_samples),
    'C_e_array' (float np.array) : array of C_e sampled (size : N_samples),
    'P_array' (float np.array) : array of the power sampled (size : N_samples)
    """

    dt = (t_fin - t_ini)/N_samples
    t_array = np.linspace(t_ini, t_fin, N_samples)
    om_array = np.zeros(N_samples)
    om_array[0] = u_origin

    for n in range(0, N_samples-1):
        cur_time = t_array[n]
        om_array[n+1] = om_array[n] + dt*fun(cur_time, om_array[n],
                                        J, C_r, R, k, V_array[n], C_e_array[n], lamb)
        if (om_array[n+1] < 0):
            om_array[n+1] = 0
            #print("ATTENTION : omega est devenu négatif. Il a donc été mis à 0 au lieu de sa valeur'")    
  
    P_array = om_array*C_e_array

    return t_array, V_array, om_array, C_e_array, P_array


## ===== Calcul des observables =====

def compute_C_e_opt_theo(C_r, lamb, V, k):
    """
    Paramètres :
    'C_r' (float) : couple ,
    'V' (float) : wind speed,
    'lamb' (float) : coefficient lambda,
    'k' (float) : coefficient k


    Renvoie la valeur théorique du couple résistif C_e,
    lorsque le vent 'V' est constant et connu, et que C_r est connu.
    """
    first_comp = 3*C_r + (lamb*V)**2 * k
    second_comp = lamb*V*np.sqrt(k*(k*(lamb*V)**2 - 3*C_r))
    return 2/9*(first_comp + second_comp)


def compute_variation_P(fun, C_e_min, C_e_max, N_samples_C_e,
                        t_ini, t_fin, N_samples_t,
                        u_origin, J = _J, C_r = _C_r, R = _R,
                        k = _k, V = _V, lamb = _lamb):
    """
    Paramètres :
    'fun' (fonction) : the function of the Euler scheme,
    'C_e_min' (float) : beginning value of C_e,
    'C_e_max' (float) : ending time of C_e_,
    'N_samples_C_e' (int) : number of samples for C_e,
    't_ini' (float) : beginning time of the scheme,
    't_fin' (float) : ending time of the scheme,
    'N_samples_t' (int) : number of time's samples,
    'u_origin' (float) : origin value of 'fun' at time t=t_ini,
    'J' (float) : moment of inertia,
    'C_r' (float) : couple ,
    'R' (float) : radius of the wind turbine,
    'k' (float) : coefficient k,
    'V' (float) : constant part of wind speed,
    'lamb' (float) : coefficient lambda

    Return :
    'C_e_array_samp' (float array) : array containing the different values of C_e_
        (size : N_samples_C_e),
    'om_average_array' (float array) : array containing the average value
        (for a set value of C_e_) of the rotation speed omega (size : N_samples_C_e),
    'P_average_array' (float array) : array containing the average value
        (for a set value of C_e_) of the power omega (size : N_samples_C_e)
    """

    C_e_array_samp = np.linspace(C_e_min, C_e_max, N_samples_C_e)
    om_average_array = np.zeros(N_samples_C_e)
    P_average_array = np.zeros(N_samples_C_e)

    for n in range(0, N_samples_C_e):
        V_array = generate_V_with_noises(N_samples_t, V)
        
        # a chaque nouvelle valeur de couple C_e, on crée un nouveau vent
        C_e_array = [C_e_array_sampl[n]]*N_samples_t
        t, v, o, c, p = euler(fun, t_ini, t_fin, N_samples_t, u_origin, V_array,
                                J, C_r, R, k, C_e_array, lamb)
        om_average_array[n] = np.average(o)
        P_average_array[n] = np.average(p)

    return C_e_array_samp, om_average_array, P_average_array

def compute_relaxation_time(fun, t_ini, t_fin, N_samples, u_origin, V_array, C_e_array, eps,
        J = _J, C_r = _C_r, R = _R,
        k = _k,
        lamb = _lamb):
    """
    Paramètres :
    'fun' (fonction) : the function of the Euler scheme,
    't_ini' (float) : beginning time of the scheme,
    't_fin' (float) : ending time of the scheme,
    'N_samples' (int) : maximum number of time's samples,
    'u_origin' (float) : origin value of 'fun' at time t=t_ini,
    'V_array' (float np.array) : array containing wind's speed. Must be of size 'N_samples',
    'J' (float) : moment of inertia,
    'C_r' (float) : couple ,
    'R' (float) : radius of the wind turbine,
    'k' (float) : coefficient k,
    'C_e_array' (float np.array) : couple. Must be of size 'N_samples',
    'lamb' (float) : coefficient lambda

    Return :
    't_array' (float np.array) : array of time sampled (size : N_samples),
    'V_array' (float np.array) : array of wind's speed sampled (size : N_samples),
    'om_array' (float np.array) : array of rotation speed sampled (size : N_samples),
    'C_e_array' (float np.array) : array of C_e sampled (size : N_samples),
    'P_array' (float np.array) : array of the power sampled (size : N_samples)
    """

    dt = (t_fin - t_ini)/N_samples
    t_array = np.linspace(t_ini, t_fin, N_samples)
    om_array = np.zeros(N_samples)
    om_array[0] = u_origin
    n_min = int(deltaT/dt);

    for n in range(0, N_samples-1):
        
        if n > n_min and (om_array[n] - om_array[n - n_min]) / om_array[n - n_min] < eps:
            return n - n_min

        cur_time = t_array[n]
        om_array[n+1] = om_array[n] + dt*fun(cur_time, om_array[n],
                                        J, C_r, R, k, V_array[n], C_e_array[n], lamb)
        if (om_array[n+1] < 0):
            om_array[n+1] = 0
            #print("ATTENTION : omega est devenu négatif. Il a donc été mis à 0 au lieu de sa valeur'")    
    
    print("No equilibrium found")
    return N_samples


## ===== Fonctions d'affichage =====

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


def display_variation_P_omega_moyenne(C_e_array, om_ave_array, P_ave_array):
    fig, axs = plt.subplots(2)

    P_max = np.max(P_ave_array)
    ind_max = np.argmax(P_ave_arr)
    om_P_max = om_ave_arr[ind_max]

    # ===========| Plot moyenne de omega |================
    axs[0].plot(C_e_array, om_ave_array)
    axs[0].scatter([C_e_array[ind_max]], [om_P_max], c='r' , s = 10)
    axs[0].set_xlabel("C_e (N.m)")
    axs[0].set_ylabel("Moyenne de omega")
    # ===========| Plot moyenne de P |================
    axs[1].plot(C_e_array, P_ave_array)
    axs[1].scatter([C_e_array[ind_max]], [P_max], c='r' , s = 10)
    axs[1].set_xlabel("C_e (N.m)")
    axs[1].set_ylabel("Moyenne de la puissance")

    print(f"""
    La valeur maximale de P (moyen) est {P_max:.2f}.
    Elle est atteinte pour C_e = {C_e_array[ind_max]:.3f}.
    La valeur de omega (moyen) en ce point est {om_P_max:.3f}
    """)

    fig.tight_layout()
    plt.show()



# ============| Affichage des données |=================
# _N_samples = 1000
# _V_array = np.array([_V]*_N_samples)
# t, v, o, c, p = euler(f, 0, 20, _N_samples, 0, _V_array)
# print(f"La valeur finale expérimentale de omega est : {o[-1]:.3f}\nCelle de P est : {p[-1]:.3f}")
# print(f"La valeur finale théorique de omega est : {om_theo:.3f}")
# print(f"Pourcentage d'erreur entre valeur théorique et valeur exp : {abs(o[-1]- om_theo)*100:.2f}")
#
# print(f"Puissance moyenne : {np.average(p):.2f}")
# display(t,v,o,c,p)


# ===========| Affichage des données moyennées |=============
C_e_arr, om_ave_arr, P_ave_arr = compute_variation_P(f, 0, 200, 500, 0, 20, 1000, 0)
display_variation_P_omega_moyenne(C_e_arr, om_ave_arr, P_ave_arr)


# coder un algorithme pour piloter la meilleure puissance moyenne,
# la meilleure puissance
# rajouter des variations dans le vent (V_array avec les composantes non constantes)
# rajouter un bruit gaussien
