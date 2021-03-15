import numpy as np
from display import *
from simulation import *
from parameters import *

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
C_e_arr, om_ave_arr, P_ave_arr = compute_variation_P(
    f, 0, 200, 500, 0, 20, 1000, 0)
display_variation_P_omega_moyenne(C_e_arr, om_ave_arr, P_ave_arr)


# coder un algorithme pour piloter la meilleure puissance moyenne,
# la meilleure puissance
# rajouter des variations dans le vent (V_array avec les composantes non constantes)
# rajouter un bruit gaussien
