def compute_variation_omega_theo(t_ini, t_fin, N_samples_t, om_origin, C_e = _C_e, J = _J, C_r = _C_r, R = _R, k = _k, V = _V, lamb = _lamb):
    time = np.linspace(t_ini, t_fin, N_samples_t)
    omega = np.zeros(N_samples_t)
    omega[0] = om_origin

    omega_eq = lamb * V/R - 1/R*np.sqrt((C_r + C_e)/k)
    beta = (C_r + C_e)*R/J
    alpha = R*k*beta/J
    z_0 = (lamb*V - R*om_origin)/beta
    om_ori = om_origin


    t_ini = 0
    for n in range(1, N_samples_t):
        coef_cons = lamb*V/R
        coef_c_r = np.sqrt((C_r + C_e)/k)/R
        coef_ponde = np.sqrt(alpha)*z_0

        coef_tanh = np.tanh(np.sqrt(alpha)*(time[n] - t_ini))
        om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)

        coef_tan = np.tan(np.sqrt(alpha)*(time[n] - t_ini))
        om_neg = coef_cons - coef_c_r * (coef_ponde + coef_tan)/(1 - coef_ponde * coef_tan)

        # cas om_pos <= 0 pour accepter les cas où la tangente hyperbolique explose
        if om_neg >= lamb*V/R and (om_pos >= lamb*V/R or om_pos <= 0):
            #print(om_pos, om_neg, n, time[n])
            omega[n] = om_neg
            continue

        if om_pos <= lamb*V/R and om_pos >= 0:
            #print(om_pos, om_neg, n, time[n])
            if (omega[n-1] >= lamb*V/R):
                t_ini = t[n-1]
                om_ini = omega[n-1]
                z_0 = (lamb*V - R*om_ini)/beta
                coef_ponde = np.sqrt(alpha)*z_0
                coef_tanh = np.tanh(np.sqrt(alpha)*(time[n] - t_ini))

                om_pos = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)

            omega[n] = om_pos
            #print(om_pos, time[n], n)

        #elif (om_neg > lamb*V/R):
        else:
            #print("ici")
            if (om_neg <= lamb*V/R):
                #print("bonjour, je suis ", n)
                t_ini = t[n-1]
                om_ori = omega[n-1]
                z_0 = (lamb*V - R*om_ori)/beta
                coef_ponde = np.sqrt(alpha)*z_0
                coef_tanh = np.tanh(np.sqrt(alpha)*(time[n] - t_ini))

                # c'est en fait om_pos, calculé à partir du temps précédent
                om_neg = coef_cons - coef_c_r * (coef_ponde + coef_tanh)/(1 + coef_ponde * coef_tanh)
            omega[n] = om_neg
            #print(om_neg, time[n], n)
        #print(om_pos, om_neg, n, time[n])
    return time, omega