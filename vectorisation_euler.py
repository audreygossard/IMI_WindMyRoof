# tout est dans le nom, afin de créer le tableau de manière plus efficace

import wind_my_roof_25_02_2021 as wmf
import numpy as np
import time as time

_J, _C_r, _R, _k, _V, _C_e, _lamb = wmf.set_parameters_default()


def f_vect(u, J, C_r, R, k, V, C_e, lamb):
    C_v = k*np.sign(lamb*V - R*u)*(lamb*V - R*u)**2
    return (C_v - C_e - C_r)/J


# version 1
# def euler(fun, t_ini, t_fin, N_samples, u_origin, V_array,
#           J=np.array([_J]), C_r=np.array([_C_r]), R=np.array([_R]),
#           k=np.array([_k]), C_e=np.array([_C_e]), lamb=np.array([_lamb])):
#     """
#     Paramètres :
#     'fun' (fonction) : the function of the Euler scheme,
#     't_ini' (float) : beginning time of the scheme,
#     't_fin' (float) : ending time of the scheme,
#     'N_samples' (int) : number of time's samples,
#     'u_origin' (array) : origin value of 'fun' at time t=t_ini,
#     'V_array' (float array) : array containing wind's speed. Must be of size 'N_samples',
#     'J' (float) : moment of inertia,
#     'C_r' (array) : couple ,
#     'R' (array) : radius of the wind turbine,
#     'k' (array) : coefficient k,
#     'C_e' (array) : couple ,
#     'lamb' (array) : coefficient lambda
#
#     Return :
#     't_array' (float array) : array of time sampled (size : N_samples),
#     'V_array' (float array) : array of wind's speed sampled (size : n_config * N_samples),
#     'om_array' (float array) : array of rotation speed sampled (size : n_config * N_samples),
#     'C_e_array' (float array) : array of C_e sampled (size : n_config * N_samples),
#     'P_array' (float array) : array of the power sampled (size : n_config * N_samples)
#     """
#
#     # on commence par crée les vecteurs qui, mis a coté, formeront tous les couples de paramètres que l'on veut
#     # on a ainsi plus qu'a appliquer euler pour chaque coordonnée de ces vecteurs
#     n_u, n_J, n_Cr, n_R, n_k, n_Ce, n_lamb = len(u_origin), len(J), len(C_r), len(R), len(k), len(C_e), len(lamb)
#     n_config = n_u * n_J * n_Cr * n_R * n_k * n_Ce * n_lamb  # nombre de configurations à tester
#
#     lamb_big = np.concatenate(tuple([lamb for i in range(n_u * n_J * n_Cr * n_R * n_k * n_Ce)]))
#     C_e_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_lamb) for el in C_e])) for i in
#                                     range(n_u * n_J * n_Cr * n_R * n_k * n_Ce)]))
#     k_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_Ce * n_lamb) for el in k])) for i in
#                                   range(n_u * n_J * n_Cr * n_R * n_k)]))
#     R_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_k * n_Ce * n_lamb) for el in R])) for i in
#                                   range(n_u * n_J * n_Cr * n_R)]))
#     C_r_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_R * n_k * n_Ce * n_lamb) for el in C_r]))
#                                     for i in range(n_u * n_J * n_Cr)]))
#     J_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_Cr * n_R * n_k * n_Ce * n_lamb) for el in J]))
#                                   for i in range(n_u * n_J)]))
#     u_origin_big = np.concatenate(tuple([np.concatenate(tuple([el * np.ones(n_J * n_Cr * n_R * n_k * n_Ce * n_lamb)
#                                                                for el in u_origin]))
#                                   for i in range(n_u)]))
#
#     dt = (t_fin - t_ini) / N_samples
#     t_array = np.linspace(t_ini, t_fin, N_samples)
#     om_array = np.zeros(shape=(n_config, N_samples))
#     om_array[:, 0] = u_origin_big
#
#     for n in range(0, N_samples - 1):
#         om_array[:, n + 1] = om_array[:, n] + dt * fun(om_array[:, n], J_big, C_r_big, R_big, k_big, V_array[n],
#                                                        C_e_big, lamb_big)
#         if not (om_array[:, n + 1] < 0).all():  # si yen a un négatif
#             for k in np.where(om_array[:, n + 1] < 0):  # on remet les négatifs à 0
#                 om_array[k, n + 1] = 0
#                 # print("ATTENTION : omega est devenu négatif. Il a donc été mis à 0 au lieu de sa valeur'")
#
#     C_e_array = np.outer(C_e_big, np.ones(N_samples))
#     C_e_array[:, 0] = 0
#     print("Calcul P_array", end=" ")
#     # c'est ici qu'on perde le plus de temps
#     P_array = np.zeros(shape=(n_config, N_samples))
#     for i in range(n_lamb):
#         index = i + n_lamb * np.arange(n_u * n_J * n_Cr * n_R * n_k * n_Ce)
#         P_array[index] = om_array[index] * C_e[i]
#     print(" Fait")
#
#     return t_array, V_array, om_array, C_e_array, P_array


def make_big(u_origin: np.ndarray, J: np.ndarray, C_r: np.ndarray, R: np.ndarray, k: np.ndarray, C_e: np.ndarray,
             lamb: np.ndarray):
    n_u, n_J, n_Cr, n_R, n_k, n_Ce, n_lamb = len(u_origin), len(J), len(C_r), len(R), len(k), len(C_e), len(lamb)
    n_config = n_u * n_J * n_Cr * n_R * n_k * n_Ce * n_lamb  # nombre de configurations à tester

    lamb_big = np.resize(lamb, (n_config,))
    C_e_big = np.resize(np.repeat(C_e, n_lamb), (n_config,))
    k_big = np.resize(np.repeat(k, n_lamb * n_Ce), (n_config,))
    R_big = np.resize(np.repeat(R, n_lamb * n_Ce * n_k), (n_config,))
    C_r_big = np.resize(np.repeat(C_r, n_lamb * n_Ce * n_k * n_R), (n_config,))
    J_big = np.resize(np.repeat(J, n_lamb * n_Ce * n_k * n_R * n_Cr), (n_config,))
    u_origin_big = np.resize(np.repeat(u_origin, n_lamb * n_Ce * n_k * n_R * n_Cr * n_J), (n_config,))

    return u_origin_big, J_big, C_r_big, R_big, k_big, C_e_big, lamb_big


# version 2 (200000 en 35 secondes, 280000 en 70 secondes, 400000 en 3000 secondes)
# le problème devient ici la mémoire ram nécéssité : 7 * n_config * 32 (ou 64 pour les lfaot) et ajouter 3 * N_samples
# * n_config * 32 pour les tableau de valeurs numériques
# fait des pointes à 5 Go de RAM sur mon PC pour 400 000 configurations et prends très longtemps
def vect_euler(fun, t_ini, t_fin, N_samples, u_origin, V_array,
          J=np.array([_J]), C_r=np.array([_C_r]), R=np.array([_R]),
          k=np.array([_k]), C_e=np.array([_C_e]), lamb=np.array([_lamb])):
    """
    Paramètres :
    'fun' (fonction) : the function of the Euler scheme,
    't_ini' (float) : beginning time of the scheme,
    't_fin' (float) : ending time of the scheme,
    'N_samples' (int) : number of time's samples,
    'u_origin' (array) : origin value of 'fun' at time t=t_ini,
    'V_array' (float array) : array containing wind's speed. Must be of size 'N_samples',
    'J' (float) : moment of inertia,
    'C_r' (array) : couple ,
    'R' (array) : radius of the wind turbine,
    'k' (array) : coefficient k,
    'C_e' (array) : couple ,
    'lamb' (array) : coefficient lambda

    Return :
    't_array' (float array) : array of time sampled (size : N_samples),
    'V_array' (float array) : array of wind's speed sampled (size : n_config * N_samples),
    'om_array' (float array) : array of rotation speed sampled (size : n_config * N_samples),
    'C_e_array' (float array) : array of C_e sampled (size : n_config * N_samples),
    'P_array' (float array) : array of the power sampled (size : n_config * N_samples)
    """

    # on commence par crée les vecteurs qui, mis a coté, formeront tous les couples de paramètres que l'on veut
    # on a ainsi plus qu'a appliquer euler pour chaque coordonnée de ces vecteurs
    n_u, n_J, n_Cr, n_R, n_k, n_Ce, n_lamb = len(u_origin), len(J), len(C_r), len(R), len(k), len(C_e), len(lamb)
    n_config = n_u * n_J * n_Cr * n_R * n_k * n_Ce * n_lamb  # nombre de configurations à tester

    print("Nombre config à tester : ", n_config)
    if n_config >= 300000:
        print("Attention, le nombre de configuration est trop important et le calcul risque de prendre très longtemps")

    u_origin_big, J_big, C_r_big, R_big, k_big, C_e_big, lamb_big = make_big(u_origin, J, C_r, R, k, C_e, lamb)

    print("Vecteurs big créés")
    print("Tailles : l = {}, C_e = {}, k = {}, R = {}, C_r = {}, J = {}, u = {}".format(len(lamb_big), len(C_e_big),
                                                                                        len(k_big), len(R_big),
                                                                                        len(C_r_big), len(J_big),
                                                                                        len(u_origin_big)))

    dt = (t_fin - t_ini) / N_samples
    t_array = np.linspace(t_ini, t_fin, N_samples)
    om_array = np.zeros(shape=(n_config, N_samples))
    P_array = np.zeros(shape=(n_config, N_samples))
    om_array[:, 0] = u_origin_big

    for n in range(0, N_samples - 1):
        om_array[:, n + 1] = om_array[:, n] + dt * fun(om_array[:, n], J_big, C_r_big, R_big, k_big, V_array[n],
                                                       C_e_big, lamb_big)
        om_array[np.where(om_array[:, n + 1] < 0), n + 1] = 0  # fait la même chose que les lignes en dessous
        # print("ATTENTION : omega est devenu négatif. Il a donc été mis à 0 au lieu de sa valeur'")
        P_array[:, n + 1] = om_array[:, n + 1] * C_e_big

    print("Schéma sur les {} pas de temps effectué \n".format(N_samples))

    C_e_array = np.outer(C_e_big, np.ones(N_samples))
    C_e_array[:, 0] = 0

    return t_array, V_array, om_array, C_e_array, P_array


if __name__ == '__main__':
    n_test = 20
    u = np.array([i for i in range(n_test)])
    c = np.array([1.5+i/10 for i in range(n_test)])
    r = np.array([0.53+i/100 for i in range(n_test)])
    k = np.array([0.9])
    C_e = np.array([20+i for i in range(n_test)])
    la = np.array([1])
    test = f_vect(u, _J, c, r, k, _V, c, la)
    print(test.shape)
    print(n_test**4)

    N_samples = 1000
    V_wind = 10
    V_wind_array = np.array([V_wind] * N_samples)

    t1 = time.time()
    t, v, o, c, p = vect_euler(f_vect, 0, 20, N_samples, u, V_wind_array, C_r=c, R=r, k=k, C_e=C_e, lamb=la)
    t2 = time.time()

    print("Temps pour résoudre Euler : {} secondes ".format(int(10*(t2-t1))/10))

    # wmf.display(t, v, o[0], c[0], p[0])
