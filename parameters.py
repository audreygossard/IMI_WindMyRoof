# List of the parameters hardcoded in this project

import numpy as np

_J = 10
_C_r = 1.5
_R = 0.53
_k = 0.9
_V = 10
_C_e = 20
_lamb = 1

eps = 0.05
deltaT = 3

om_theo = (_lamb*_V)/_R - 1/_R * np.sqrt((_C_r + _C_e)/_k)