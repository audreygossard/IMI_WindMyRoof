# fichier pour créer le gros tableau de variation de C_e, faire la fonction d'interpolation autour de ses valeurs,
# l'enregistrer et extraire celui-ci d'un fichier texte

import wind_my_roof_25_02_2021 as wmf
import numpy as np


def ind_reg_perm(om_array, om_theo, precision=0.1):
    """
    On veut trouver le premier instant tel que la valeur finale est attainte (à une précision prêt)
    L'ereur sur la valeur finale tourne aux alentours de 0.05% donc dès qu'on est à 0.1% de la valeur
    théorique c'est bon
    """
    ind = np.argmax(om_array > (1-precision/100)*om_theo)
    return ind


def val_sauv_om_t(om_array, t_array, ind_perm, nb_pts):
    """
    On veut garder que un certain nombre de points (nb_pts) entre les indices 0 et ind_perm (exclu)
    """
    # om_sauv = np.zeros(nb_pts)
    # t_sauv = np.zeros(nb_pts)
    q, r = ind_perm//nb_pts, ind_perm % nb_pts
    # # on prend un points tous les q indices sauf pour les r derniers ou c'est tous les q+1
    # ind = 0
    # for ind_sauv in range(nb_pts):
    #     om_sauv[ind_sauv] = om_array[ind]
    #     t_sauv[ind_sauv] = t_array[ind]
    #     if ind_sauv >= nb_pts-r-1:
    #         ind += q+1
    #     else:
    #         ind += q
    om_sauv_1 = om_array[:(nb_pts-r)*q:q]
    om_sauv_2 = om_array[(nb_pts-r)*q+1:ind_perm:q+1]
    om_sauv = np.concatenate((om_sauv_1, om_sauv_2))

    t_sauv_1 = t_array[:(nb_pts - r) * q:q]
    t_sauv_2 = t_array[(nb_pts - r) * q + 1:ind_perm:q + 1]
    t_sauv = np.concatenate((t_sauv_1, t_sauv_2))

    return om_sauv, t_sauv


# à vectoriser bien pour que se soit plus rapide, ie paraleliser les calculs pour euler (peut se faire sur presque
# tous les params mais besoin d'adapter les fonctions de Euler et Objectif)
def create_tab_var_Ce(u_array: np.ndarray, V_array: np.ndarray, Cr_array: np.ndarray, R_array: np.ndarray,
                      k_array: np.ndarray, Ce_array: np.ndarray, lamb_array: np.ndarray):
    """
    'parameters_array' (array) : array des valeurs choisis pour le paramètre

    On crée le tableau donnant les valuers de oméga en fonction du temps, en fonction de u initial, du C_e imposé et
    du vent (supposé constant ici) incident à l'éolienne
    """
    # etre a un niveau de C_e correspond à etre a un niveau de u en vent permanent donc pour l'instant plus simple de
    # faire comme ca, quand la modélisation du vent changera on réadaptera la fonction

    n_u, n_V, n_Cr, n_R, n_k, n_Ce, n_lamb = u_array.shape[0], V_array.shape[0], Cr_array.shape[0], R_array.shape[0],\
                                             k_array.shape[0], Ce_array.shape[0], lamb_array.shape[0]
    n_pts_om = 100  # le nombre de points que l'on garde dans le tableau pour l'évolution temporel de oméga
    tab = np.zeros((n_u, n_V, n_Cr, n_R, n_k, n_Ce, n_lamb, n_pts_om, 2))  # tableau final
    # de la forme [u, V, Cr, R, k, Ce, lamb, (t, omega(t))]
    # on récupère les couples (t, omega(t)) en prenant toutes les valeurs sur les 2 dernières dimensions
    N_samples = 1000
    for u, u_origin in enumerate(u_array):
        for v_ind, V_wind in enumerate(V_array):
            for cr, C_r in enumerate(Cr_array):
                for r, R in enumerate(R_array):
                    for k_ind, k in enumerate(k_array):
                        for ce, C_e in enumerate(Ce_array):
                            for l, lamb in enumerate(lamb_array):
                                V_wind_array = np.array([V_wind] * N_samples)
                                t, v, o, c, p = wmf.euler(wmf.f, 0, 20, N_samples, u_origin, V_wind_array,
                                                          C_r=C_r, R=R, k=k, C_e=C_e, lamb=lamb)
                                om_t = wmf.om_theo(C_r, R, k, V_wind, C_e, lamb)
                                k = ind_reg_perm(o, om_t)
                                save_o, save_t = val_sauv_om_t(o, t, k, n_pts_om)
                                tab[u, v_ind, cr, r, k_ind, ce, l] = np.transpose(np.vstack((save_t, save_o)))
                                # on empile verticalement puis on transpose pour avoir une colonne de t et une de w(t)
    return tab


# fonctions qui enregistre et qui lit le fichier

def array_to_string(tab: np.ndarray):
    res = ""
    for el in tab:
        res += str(el) + " "
    return res


def save_tab_Ce(tab: np.ndarray, u_array: np.ndarray, V_array: np.ndarray, Cr_array: np.ndarray, R_array: np.ndarray,
                k_array: np.ndarray, Ce_array: np.ndarray, lamb_array: np.ndarray, new: bool = True):
    """
    'tab'               (array) : tableau à enregistrer
    'parameters_array'  (array) : array des valeurs choisies pour le paramètre

    Enregistre le tableau dans un fichier .txt (dans le dossier du fichier python)
    """
    if new:
        file = open("Variations_Omega.txt", mode='x')  # pour créer le fichier
    with open("Variations_Omega.txt", "w") as file:
        file.write(str(tab.shape) + "\n")  # première ligne pour savoir quel array créer à la lecture
        # on parcours ensuite le tableau et pour chaque simulation enregistrée on écrit 3 lignes : les paramètres de
        # la simulation, les velurs de t, les valeurs de omega
        for u, u_origin in enumerate(u_array):
            for v_ind, V_wind in enumerate(V_array):
                for cr, C_r in enumerate(Cr_array):
                    for r, R in enumerate(R_array):
                        for k_ind, k in enumerate(k_array):
                            for ce, C_e in enumerate(Ce_array):
                                for l, lamb in enumerate(lamb_array):
                                    # j'ai pas trouvé de fonctions transformant un array en un string qui peut etre lu
                                    # par np.fromstring()
                                    param = str(u_origin) + " " + str(V_wind) + " " + str(C_r) + " " + str(R) + " " +\
                                            str(k) + " " + str(C_e) + " " + str(lamb)
                                    time_omega = tab[u, v_ind, cr, r, k_ind, ce, l]
                                    time = array_to_string(time_omega[:, 0])
                                    omega = array_to_string(time_omega[:, 1])
                                    file.write(param + "\n")
                                    file.write(time + "\n")
                                    file.write(omega + "\n")
    return 0


def load_tab_Ce():
    """
    Lis et rentre le tableau et les paramètres dans un array (besoin du format d'enregistrement de save_tab_Ce)
    """
    with open("Variations_Omega.txt", "r") as file:
        lines = file.readline()
        data = lines[1:]
        shape = np.fromstring(lines[0], dtype=int)
        tab = np.zeros(shape)
        n_u, n_V, n_Cr, n_R, n_k, n_Ce, n_lamb, n_pts_om, x = shape
        u_set = set({})
        V_set = set({})
        Cr_set = set({})
        k_set = set({})
        Ce_set = set({})
        lamb_set = set({})
        for n in range(len(data)//3):
            param = np.fromstring(data[3*n])
            time = np.fromstring(data[3*n+1])
            omega = np.fromstring(data[3*n+2])
            u_set.add(param[0])
            V_set.add(param[1])
            Cr_set.add(param[2])
            k_set.add(param[3])
            Ce_set.add(param[4])
            lamb_set.add(param[5])
            time_omega = np.transpose(np.vstack((time, omega)))
            tab[len(u_set)-1, len(V_set)-1, len(Cr_set)-1, len(k_set)-1, len(Ce_set)-1, len(lamb_set)-1] = time_omega
    return tab, u_array, V_array, Cr_array, k_array, Ce_array, lamb_array


if __name__ == '__main__':
    C_e_command = 20
    _J, _C_r, _R, _k, _V, _C_e, _lamb = wmf.set_parameters_default()
    om_t = wmf.om_theo(_C_r, _R, _k, 9, C_e_command, _lamb)

    # ============| Affichage des données |=================
    _N_samples = 1000
    _V_array = np.array([9] * _N_samples)

    t, v, o, c, p = wmf.euler(wmf.f, 0, 20, _N_samples, 0, _V_array, C_e=C_e_command)

    k = ind_reg_perm(o, om_t)
    print(f"Indice du régime permanent : {k}")
    print(f"Temps auquel regime permanent atteint : {t[k]:.3f}")
    print(f"Comparaison des valeurs : {om_t:.4f} vs {o[k]:.4f}")

    save_o, save_t = val_sauv_om_t(o, t, k, 100)
    print("Nombre de valeurs retenues : ", len(save_o))
    print(save_o[-1])
    print(save_t[-1])

    # ============| Test création tableau |=================
    u_array = np.array([0, 1, 2, 3, 4])
    V_array = np.array([10, 11, 12, 13, 14])
    Cr_array = np.array([1.5, 1.6, 1.7, 1.8, 1.9])
    R_array = np.array([0.53, 0.54, 0.55, 0.56, 0.57])
    k_array = np.array([0.9])
    Ce_array = np.array([20])
    lamb_array = np.array([1])

    test = create_tab_var_Ce(u_array, V_array, Cr_array, R_array, k_array, Ce_array, lamb_array)
    print("\nDimensions du test : {}".format(test.shape))

    couple_t_om = test[0, 0, 0, 0, 0, 0, 0]
    temps = couple_t_om[:, 0]
    om = couple_t_om[:, 1]
    # print("\nVerification : ")
    # print(temps)
    # print(om)

    print("Test enregistrement : ")
    save_tab_Ce(test, u_array, V_array, Cr_array, R_array, k_array, Ce_array, lamb_array)
