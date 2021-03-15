import numpy as np
from parameters import *


def interpolation(simu_values, simu_parameters, parameters):
    """
    Parameters :
    'simu_values' (numpy array) : multi-dimensionnal array of all of the simulated values,
    'simu_parameters' (array) : 2D-array of all of the parameters used for the simulation (the nth element is the array of the values used for the nth parameter),
    'parameters' (array) : 1D-array of the values of the parameters we're aiming at.

    Return :
    'value' (float) : Interpolated value.
    """
    interpolated_parameters = []
    for idx in range(len(parameters)):
        for k in range(len(simu_parameters)):
            if parameters[idx] > simu_parameters[k]:
                interpolated_parameters.append(k)
                break

    interpolated_values = np.copy(simu_values)
    for param_idx in interpolated_parameters:
        interpolated_values = (interpolated_values[param_idx] + interpolated_values[param_idx]) / 2

    return interpolated_values