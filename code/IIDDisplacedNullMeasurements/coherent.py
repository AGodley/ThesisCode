# Alfred Godley
# Estimating coherent states through our triangulation strategy.
# These coherent states emerge from our iid qubit model in the asymptotic setting.
# Coherent states are eigenstates of the annihilation operator a.
# Therefore, they're characterised by their corresponding eigenvalue alpha.
# This is a complex number, so it converts the problem into a 2D estimation problem.
# Measuring the number operator on coherent states results in pulls from a Poisson distribution.
# This is what the code relies on.
# Instead of measuring the number operator directly we measure displaced number operators.
# These correspond to other coherent states with parameters zi
# The result is then a pull from Poisson dist. with parameter |z1-alpha|^2

###########
# Imports #
###########
import numpy as np  # Standard numpy and matplotlib
import matplotlib.pyplot as plt
from numpy import pi, sqrt, cos, sin    # Useful constants/functions from numpy
from numpy.random import poisson    # Poisson distribution
import time     # for timing the code
from multiprocessing import Pool    # For parallelizing the trajectory generation
from itertools import repeat    # For use with starmap
from pathlib import Path   # For file directories
import csv  # For comma separated values
from scipy.optimize import minimize    # For mle


# My Poisson function; Python struggles to handle alpha exponent, so use the numpy one
def my_poisson(u_poi, v_poi, alpha_0_poi, zi_poi, ki_poi):
    alpha_poi = alpha_0_poi + u_poi + 1j*v_poi
    amp_poi = abs(alpha_poi - zi_poi)
    p_n = np.exp(-(amp_poi**2)) * abs(amp_poi)**(2*ki_poi) / np.math.factorial(ki_poi)
    return p_n


# Function for maximum likelihood
def mle(uv_mle, alpha_0_poi, zs_mle, ks_mle):
    # Initializes the log-likelihood
    log_p = 0

    # Adds on contribution from first measurement
    log_p += np.log(my_poisson(uv_mle[0], uv_mle[1], alpha_0_poi, zs_mle[0], ks_mle[0]))

    # Adds on contribution from second measurement
    log_p += np.log(my_poisson(uv_mle[0], uv_mle[1], alpha_0_poi, zs_mle[1], ks_mle[1]))
    return -log_p


# Trajectory generation; for repeating N times with parallelized generation
def trajectory(id_val, n_val, alpha_val, alpha_0_val, angs_val, eps_val):
    # Starts loop timer
    t0 = time.time()

    # Amplitude of the measurements
    amp_zs = sqrt(2) * n_val**(3*eps_val)

    # Simulating measurement of the first coherent state corresponding to the first n1 qubits
    ang_1 = angs_val[0]     # Angle of first measurement
    z1 = amp_zs*np.exp(1j*ang_1)    # Complex number corresponding to the first measurement
    k1 = poisson(abs(alpha_val - z1) ** 2)   # Simulates measurement as a pull from the distribution

    # Simulating measurement of the next coherent state corresponding to the second n2 qubits
    ang_2 = angs_val[1]  # Angle of second measurement
    z2 = amp_zs * np.exp(1j * ang_2)  # Complex number corresponding to the first measurement
    k2 = poisson(abs(alpha_val - z2) ** 2)  # Simulates measurement as a pull from the distribution

    # # Minimization
    # res = minimize(mle, x0=np.array([0, 0]), args=(alpha_0_val, [z1, z2], [k1, k2]))
    # u, v = res.x
    # approx_alpha = alpha_0_val + u + v*1j
    #
    # # Visualization
    # plot = True
    # if plot:
    #     # Radius of localized region
    #     Rs = n_val ** (-0.5 + eps_val)
    #     # Values to plot
    #     us = np.linspace(-Rs, Rs)     # u values
    #     vs = np.linspace(-Rs, Rs)     # v values
    #     Us, Vs = np.meshgrid(us, vs)    # grid of values
    #     mles = mle([Us, Vs], alpha_0_val, [z1, z2], [k1, k2])
    #
    #     # Initializes figure
    #     fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #     # Plots the surface
    #     surf = ax.plot_surface(Us, Vs, mles)
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf)
    #     # Axes
    #     ax.set_xlabel('u')
    #     ax.set_ylabel('v')
    #     plt.show()

    # Madalin's new estimator
    uv_estimator = [n_val**(3*eps_val) - (k1 + k2)/(4 * n_val**(3*eps_val)), (k2-k1)/(4 * n_val**(3*eps_val))]
    approx_alpha = uv_estimator[0] + uv_estimator[1]*1j  # + alpha_0_val

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(f'Sample {id_val+1} took {total_time:.2f} seconds. \n'
          f'Alpha was {alpha_val} \n'
          f'and ML resulted in: {approx_alpha}')
    return approx_alpha


#########
# Setup #
#########
setup = {
    'N': 100,   # Number of repetitions
    'n': 100,   # Number of qubits per repetition; localizes theta
    'alpha_0': 1+1j,    # Value that alpha is localized around
    'angs': [pi/4, 3*pi/4],     # Measurement angles
    'eps': 0.1  # Error parameter epsilon
}

########
# Main #
########
if __name__ == '__main__':
    # Unpacks variables from setup
    N = setup['N']
    n = setup['n']
    alpha_0 = setup['alpha_0']
    angs = setup['angs']
    eps = setup['eps']

    # Choosing true alpha; assumes alpha has been localized around alpha_0_val
    R = n ** (-0.5 + eps)  # Radius of localized region
    r = R * np.random.random()  # Generates an amplitude for alpha within this region
    phase = 2 * pi * np.random.random()  # Generates a corresponding phase
    # True value of alpha for simulations
    alpha = alpha_0 + r * np.exp(1j * phase)

    # Process ids
    ids = np.arange(setup['N'])
    # Pool object to run the multiprocessing
    pool = Pool(8)
    results = pool.starmap(trajectory, zip(ids, repeat(n), repeat(alpha), repeat(alpha_0), repeat(angs), repeat(eps)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    # Unpacks results
    estimates = [None]*len(results)
    for i in np.arange(len(results)):
        estimates[i] = results[i]

    #print(estimates)
    print(f'Alpha was {alpha}')
    # Calculates the sample variance
    sampleVar = (1 / N) * np.sum(abs((estimates - alpha) ** 2))
    sampleFI = 1 / sampleVar
    print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))

    # ===================
    ## Saving utility ##
    # ===================
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    countingMeasurements = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')
    # Saves the data
    save = False
    print('Saving FI: {}'.format(save))
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(countingMeasurements.joinpath('counting.csv'), 'a', newline='') as file:
            data = []
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)
