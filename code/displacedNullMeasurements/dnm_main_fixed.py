#############
# Main file #
#############
# Implementing the new scheme based on displacing the absorber in the quantum Markov chain.
# This causes patterns of excitations in the expected vacuum output.
# Patterns are detected and used to estimate the value of the parameter.
import sys

# Standard imports
from numpy import pi, sqrt
import numpy as np      # For arrays and matrices
import matplotlib.pyplot as plt    # For plotting graphs
from qutip import *     # Package for simulating quantum objects

# Parallelization imports
from multiprocessing import Pool    # For parallelizing the trajectory generation; allows multiple traj. to run at once
from itertools import repeat    # Used to set up the parallel processing

# System imports
import time     # For timing the trajectory generation loop
from pathlib import Path   # For file directories; used to save the trajectories
import csv   # Comma separated values; format used to save data

# Initial estimation
from initial import initial_est

# Kraus operators
from kraus import k, k_dot, k_abs, k_abs_dot, true_ss

# Absorber functions
from absorber import uV

# QFI calculation
from qfi import qfi_calc

# Pattern checking function
from patterns import pattern_check, expected


def measurement_choice():
    # Sets the measurement basis in the second stage estimation
    meas = [fock(2, 0), fock(2, 1)]
    return meas


def trajectory(id, theta_rough, n_final, setup):
    # Main function of the file; generates sample trajectories

    # Unpacks setup
    N = setup['N']  # No. of samples to generate
    n = setup['n']  # No. of ancillary systems
    theta = setup['theta']  # True value of theta for trajectory generation
    lmbd = setup['lambda']  # Known parameter
    phi = setup['phi']  # Known phase parameter
    eps = setup['eps']  # Proportion of trajectory used in initial estimate

    # Starts loop timer
    t0 = time.time()

    # We then offset the absorber outside the localized region
    offset = n ** (-0.5 + 1.25 * eps)
    absorber = theta_rough + offset

    # Stage 2: measurement in the fixed basis specified by the function
    M = measurement_choice()

    # Finds kraus operators
    K, *_ = k_abs(theta, absorber, lmbd, phi, M)

    # True stationary state of V(tht_r+offset)U(tht) in order to calculate the mpn and expected pattern no.s
    rho_t0 = true_ss(theta, absorber, lmbd, phi, M)
    # For checking the true stationary state
    # print(rho_t0)
    # print(K[0]*rho_t0*K[0].dag() + K[1]*rho_t0*K[1].dag())

    # Mean photon number
    mpn = np.real_if_close((rho_t0 * K[1].dag() * K[1]).tr())

    # Initializes the state of the s+a; estimate of true ss
    a, rho_0 = uV(theta_rough, lmbd, phi)
    rho = rho_0

    # List to record which outcomes occurred
    x = [None] * n_final

    # Trajectory generation
    for j in np.arange(n_final):
        # Defines probability that outcome 0 occurs
        p0 = np.real_if_close( (K[0] * rho * K[0].dag()).tr() )

        # Choice of {0, 1} with prob {p(0), p(1)}
        x_j = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Records the outcome in x
        x[j] = x_j

        # Updates the state by applying the measurement projection and normalising
        rho = K[x_j] * rho * K[x_j].dag()
        rho = rho / rho.tr()

    # Useful print statements for checking the percentage of 1s in the trajectory
    # print(f'Mean photon number: {mpn:.5f}')
    # print(f'Average photons in trajectory: {np.sum(x) / len(x)}')
    # if id == 0:
    #     print(x)

    # Analyses the patterns in x
    patterns, wsp = pattern_check(x)

    # Saves the trajectory analysis
    save = False
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    displaced_null_markov = adaptiveMeasurementSimulation.joinpath('data').joinpath('displacedNullMeasurements')
    # Toggles saving
    if save:
        # Opens the file in append mode
        with open(displaced_null_markov.joinpath('trajectories.csv'), 'a', newline='') as file:
            data = [n, lmbd, phi, eps, theta_rough, offset, mpn, np.sum(x)/len(x), wsp] + [val for val in patterns.values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Ensures this is only done once
    expected_x = 0
    if id == 0:
        # Calculates the expected no. of each pattern
        expected_x, mpn_pat, FI_pat = expected(x, rho_t0, K, abs(theta - absorber))
        print(f'mpn_pat={mpn_pat:.5f}; FI_pat={FI_pat:.5f}')
        # Save expected values
        if save:
            # Opens the file in append mode
            with open(displaced_null_markov.joinpath('expected.csv'), 'a', newline='') as file:
                data = [n, lmbd, phi, eps, theta_rough, offset, mpn, mpn_pat, FI_pat] + [val for val in expected_x.values()]
                z = csv.writer(file)
                # Appends data onto the file
                z.writerow(data)

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(
        '''Sample {} finished; tht_0 was {}. It took {} minutes and {:.1f} seconds. 
        Other statistics: mpn={:.5f}; Traj. avg.={:.5f}; Wsp={:.5f}'''.format(
            id+1, theta_rough, total_time // 60, total_time % 60, mpn, np.sum(x)/len(x), wsp))
    ## Sample generation and argmax ##
    # End of sample generation
    return patterns, expected_x


###########
## Setup ##
###########
# Global parameter values
setup = {
    'N': 1,  # No. of samples to generate
    'n': 50000,  # No. of ancillary systems
    'theta': 0.2,  # True value of theta for trajectory generation
    'lambda': 0.8,  # Transition parameter, 0<=lmbd<=1
    'phi': pi/4,  # Phase parameter
    'eps': 0.1,  # Prop. of traj. to use in initial est.
}
###########
## Setup ##
###########

if __name__ == '__main__':
    # Unpacks the setup
    theta = setup['theta']
    lmbd = setup['lambda']
    phi = setup['phi']
    N = setup['N']
    n = setup['n']
    eps = setup['eps']

    # Process ids
    ids = np.arange(setup['N'])

    # Splits the samples into samples used in the initial and final estimation
    n_init = int(np.floor(n ** (1 - eps)))
    n_final = n - n_init

    # Performs the initial estimation, localizing theta to a region n**(-0.5+eps) using n**(1-eps) of the samples
    # theta_rough, *_ = initial_est(theta, lmbd, phi, n_init)
    # For no initial estimation
    theta_rough = 0.2

    # Confirms we want to continue with the initial estimate
    print(f'Initial estimate: {theta_rough}')
    confirm = input('Continue (Y/N): ')
    if confirm == 'N':
        sys.exit()

    # Useful information
    print(f'Running with N={N}, n={n} and epsilon={setup["eps"]};\n'
          f'')

    # Pool object to run the multiprocessing
    pool = Pool(1)
    results = pool.starmap(trajectory, zip(ids, repeat(theta_rough), repeat(n_final), repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    # Asymptotic qfi per step
    qfi = qfi_calc(theta, lmbd, phi)
    print('QFI per step: {}'.format(np.real_if_close(qfi)))

    # Expected counts of patterns
    result = results[0]
    result_expected = result[1]
    print(f'Expected counts of pattern 1:\n'
          f'{result_expected}')
    # Shows the actual results in the first 5 trajectories
    if len(results) > 1:
        for i in range(5):
            result = results[i]
            print(f'Actual counts in trajectory {i+1}:\n'
                  f'{result[0]}')
    else:
        result = results[0]
        print(f'Actual counts in trajectory {1}:\n'
              f'{result[0]}')

    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    displaced_null_markov = adaptiveMeasurementSimulation.joinpath('data').joinpath('displaced_null_markov')
    # Save expected values
    save2 = False
    print(f'Saving expected counts: {save2}')
    # Toggles whether to save
    if save2:
        result = results[0]
        # Opens the file in append mode and saves the headers
        with open(displaced_null_markov.joinpath('expected.csv'), 'w', newline='') as file:
            rows = ['n', 'lambda', 'phi', 'epsilon', 'theta_rough', 'offset', 'mpn', 'calculated_mpn', 'calculated_FI'] + [key for key in result[1].keys()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(rows)
