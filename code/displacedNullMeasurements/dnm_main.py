#############
# Main file #
#############
# Implementing the new scheme based on displacing the absorber in the quantum Markov chain.
# This causes patterns of excitations in the expected vacuum output.
# Patterns are detected and used to estimate the value of the parameter.

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
from patterns import pattern_check, expected, alternative, stochastic_patterns


def measurement_choice():
    # Sets the measurement basis in the second stage estimation
    meas = [fock(2, 0), fock(2, 1)]
    return meas


def trajectory(id, setup):
    # Main function of the file; generates sample trajectories

    # Unpacks setup
    N = setup['N']  # No. of samples to generate
    n = setup['n']  # No. of ancillary systems
    theta = setup['theta']  # True value of theta for trajectory generation
    lmbd = setup['lambda']  # Known parameter
    phi = setup['phi']  # Known phase parameter
    eps = setup['eps']  # Proportion of trajectory used in initial estimate
    gamma = setup['gamma']   # Absorber offset

    # Starts loop timer
    t0 = time.time()

    # Controls whether initial estimation is implemented
    if not setup['initial']:
        n_final = n
        theta_rough = 0.2
        # Sets eps and gamma to reflect that they're not included
        eps = 'NA'
        gamma = 'NA'
        # Arbitrary offset based off common values
        # offset = 0.07
        # Offset based off what we want for the order of local parameter u=O(n**(a.eps))
        offset = n**(-0.5) * 6.6
    else:
        # Splits the samples into samples used in the initial and final estimation
        n_init = int(np.floor(n ** (1 - eps)))
        n_final = n - n_init
        theta_rough, *_ = initial_est(theta, lmbd, phi, n_init)
        # We then offset the absorber outside the localized region; if statement can change direction of offset
        if id % 2 == 0:
            offset = n ** (-0.5 + gamma * eps)
        else:
            # For changing offset direction
            # offset = -1*n ** (-0.5 + gamma * eps)
            # For keeping it the same
            offset = n ** (-0.5 + gamma * eps)
    absorber = theta_rough + offset

    # Stage 2: measurement in the fixed basis specified by the function
    M = measurement_choice()

    # Finds kraus operators
    K, *_ = k_abs(theta, absorber, lmbd, phi, M)

    # True stationary state of V(tht_r+offset)U(tht) in order to calculate the mpn and expected pattern no.s
    rho_t0 = true_ss(theta, absorber, lmbd, phi, M)

    # Mean photon number
    mpn = np.real_if_close((rho_t0 * K[1].dag() * K[1]).tr())
    # Looking for errors
    if np.abs(mpn) > 1:
        print(f'Ss: {rho_t0}')
        print(f'K_1: {K[1]}')

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

        # Records the outcome in x_s2
        x[j] = x_j

        # Updates the state by applying the measurement projection and normalising
        rho = K[x_j] * rho * K[x_j].dag()
        rho = rho / rho.tr()

    # Useful print statements for checking the percentage of 1s in the trajectory
    # print(f'Mean photon number: {mpn:.5f}')
    # print(f'Average photons in trajectory: {np.sum(x) / len(x)}')
    # print(x)

    # Analyses the patterns in x
    patterns, wsp = pattern_check(x, 6)
    # Alternative Stochastic method
    # patterns, wsp = stochastic_patterns(x, 1/20, 10)

    # Calculates the expected counts for the pattern
    expected_counts, mpn_pat = expected(rho_t0, K, abs(theta - absorber), n_final)

    # Alternative stationary state used in Taylor expansion formula for mu_alphas
    rho_ss = true_ss(theta_rough, theta_rough, lmbd, phi, M)
    # Purity check; should be equal to 1
    # print((rho_ss**2).tr())

    # Checking K_dot against numerical derivative
    # K_dot, *_ = k_abs_dot(theta, absorber, lmbd, phi, M)
    # K_1, *_ = k_abs(theta, absorber, lmbd, phi, M)
    # K_2, *_ = k_abs(theta+10**-6, absorber, lmbd, phi, M)
    # print(f'K_dot check\nK_dot:\n{K_dot[0]}')
    # print(f'[K(theta,theta_rough)-K(theta,theta)]/deta.theta:\n{(K_1[0] - K_2[0]) / np.abs(10**-6)}')

    # Best guess at kraus operators for calculating expected values with alternative formula
    K, *_ = k_abs(theta_rough, theta_rough, lmbd, phi, M)
    K_dot, *_ = k_abs_dot(theta_rough, theta_rough, lmbd, phi, M)  # Used in the more complex formula for mus

    # Calculates the mus for the pattern using alternative formula
    alt_mus, FI_pat, alt_expected = alternative(rho_ss, K, K_dot, abs(theta - absorber), n_final)

    # Saves the trajectory analysis
    save = True
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    displaced_null_markov = adaptiveMeasurementSimulation.joinpath('data').joinpath('displacedNullMeasurements')
    # Toggles saving
    if save:
        # Opens the file in append mode
        with open(displaced_null_markov.joinpath('varying').joinpath('trajectories_varying.csv'), 'a', newline='') as file:
            data = [n, lmbd, phi, eps, gamma, theta_rough, offset, mpn, np.sum(x)/len(x), wsp] + [val for val in patterns.values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Save expected intensities mu_squared
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(displaced_null_markov.joinpath('varying').joinpath(
                'expected_varying.csv'), 'a', newline='') as file:
            data = [n, lmbd, phi, eps, gamma, theta_rough, offset, mpn, mpn_pat, FI_pat] + [val for val in
                                                                                            expected_counts.values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Save expected counts from old formula
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(displaced_null_markov.joinpath('varying').joinpath(
                'expected_mus_varying.csv'), 'a', newline='') as file:
            data = [n, lmbd, phi, eps, gamma, theta_rough, offset, mpn, mpn_pat, FI_pat] + [val for val in
                                                                                            alt_mus.values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)
    # print(f'Patterns: \n{patterns}\nMus: \n{alt_mus}\nExpected: \n{alt_expected}')
    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(
        '''Sample {} finished; tht_0 was {}. It took {} minutes and {:.1f} seconds. 
        Other statistics: mpn={:.5f}; Traj. avg.={:.5f}; Wsp={:.5f}; mpn_pat={:.5f}; FI_pat={:.5f}'''.format(
            id+1, theta_rough, total_time // 60, total_time % 60, mpn, np.sum(x)/len(x), wsp, mpn_pat, FI_pat))
    ## Sample generation and argmax ##
    # End of sample generation
    return patterns, alt_mus, alt_expected


###########
## Setup ##
###########
# Global parameter values
setup = {
    'N': 3000,  # No. of samples to generate
    'n': 6 * 10**5,  # No. of ancillary systems
    'theta': 0.2,  # True value of theta for trajectory generation
    'lambda': 0.8,  # Transition parameter, 0<=lmbd<=1
    'phi': pi / 4,  # Phase parameter
    'initial': False,  # Controls initial estimation; eps and gamma are irrelevant if True
    'eps': 0.065,  # Prop. of traj. to use in initial est.
    'gamma': 2.1   # Controls how far the absorber is offset, gamma>1
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
    gamma = setup['gamma']

    # Process ids
    ids = np.arange(setup['N'])

    # Useful information
    print(f'Running with N={N}, n={n}')
    if setup['initial']:
        print(f'gamma={gamma} and epsilon={eps};\n'
              f'Proportion used in initial estimation: {n ** (-eps):.1%};\n'
              f'Offset: {n**(-0.5 + gamma*eps)}')
    else:
        print(f'No initial estimation')

    # Pool object to run the multiprocessing
    pool = Pool(8)
    results = pool.starmap(trajectory, zip(ids, repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    # Asymptotic qfi per step
    qfi = qfi_calc(theta, lmbd, phi)
    print('QFI per step: {}'.format(np.real_if_close(qfi)))

    ## Expected counts for trajectory 1
    result = results[0]
    result_expected = result[2]
    print(f'Expected counts of pattern 1:\n'
          f'{result_expected}')

    ## Actual counts in up to first trajectory
    # Shows the actual results in the first trajectory
    if len(results) >= 1:
        for i in range(1):
            result = results[i]
            print(f'Actual counts in trajectory {i + 1}:\n'
                  f'{result[0]}')
    else:
        result = results[0]
        print(f'Actual counts in trajectory {1}:\n'
              f'{result[0]}')

    ## Average expected counts
    result = results[0]
    avg_expected = result[2].copy()
    for i in range(len(results) - 1):
        for key in avg_expected.keys():
            result = results[i+1]
            expected = result[2]
            avg_expected[key] = avg_expected[key] + expected[key]
    # Divides cumulative expected counts by N
    for key in avg_expected.keys():
        avg_expected[key] = avg_expected[key] / N
    print(f'Average expected counts:\n'
          f'{avg_expected}')

    ## Average actual results
    result = results[0]
    avg_actual = result[0].copy()
    for i in range(len(results) - 1):
        for key in avg_expected.keys():
            result = results[i + 1]
            actual = result[0]
            avg_actual[key] = avg_actual[key] + actual[key]
    # Divides cumulative expected counts by N
    for key in avg_expected.keys():
        avg_actual[key] = avg_actual[key] / N
    print(f'Average actual counts:\n'
          f'{avg_actual}')
