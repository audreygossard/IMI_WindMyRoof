import numpy as np
from matplotlib import pyplot as plt

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
    axs[0].scatter([C_e_array[ind_max]], [om_P_max], c='r', s=10)
    axs[0].set_xlabel("C_e (N.m)")
    axs[0].set_ylabel("Moyenne de omega")
    # ===========| Plot moyenne de P |================
    axs[1].plot(C_e_array, P_ave_array)
    axs[1].scatter([C_e_array[ind_max]], [P_max], c='r', s=10)
    axs[1].set_xlabel("C_e (N.m)")
    axs[1].set_ylabel("Moyenne de la puissance")

    print(f"""
    La valeur maximale de P (moyen) est {P_max:.2f}.
    Elle est atteinte pour C_e = {C_e_array[ind_max]:.3f}.
    La valeur de omega (moyen) en ce point est {om_P_max:.3f}
    """)

    fig.tight_layout()
    plt.show()
