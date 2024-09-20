# Alfred Godley
# Implementing the new scheme based on counting measurements in different bases in the old input-output setting.
# Full version includes some preliminary estimation based on the expected total number of counts detected.

# Standard imports
import sys
from math import pi, sqrt  # For general maths
import numpy as np      # For arrays and matrices
import matplotlib.pyplot as plt    # For plotting graphs
from scipy.optimize import minimize, brute     # For argmax/maximum likelihood estimation
from qutip import *     # Package for quantum objects

# Imports from the initial estimation file
from initial_estimation_excitations import initial_est
from initial_estimation_excitations import kraus as kraus_2by2
from initial_estimation_excitations import kraus_dot as kraus_dot_2by2

# Absorber imports
from AbsorberFunctions import unitaryV  # For including the absorber
from AbsorberFunctions import ss, Id_T_heisenberg   # QFI calculation

# Parallelization imports
from multiprocessing import Pool    # For parallelizing the trajectory generation
from itertools import repeat    # For use with starmap

# System imports
import time     # For timing the trajectory generation loop
from pathlib import Path   # For file directories
import csv   # Comma separated values, useful for saving data


def kraus(theta_u, theta_ru, lmbd_u, phi_u, meas_u):
    # Function for calculating the Kraus operators
    # Also returns the corresponding pseudo-unitary
    # Kraus operators are used to update the state
    # Inputs: param of interest, a rough estimate of this param, lambda, phi and a list containing the measurements
    # When sent one measurement, it calculates the corresponding single Kraus

    # Not a true unitary as only it's action on |000>, |100>, |010> and |110> states are important
    unitry = Qobj([[np.cos(theta_u)*sqrt(1-theta_u**2), 0, 1j*np.sin(theta_u)*sqrt(1-lmbd_u), 0],
                   [0, 0, sqrt(lmbd_u)*np.exp(1j*phi_u), 0],
                   [1j*np.sin(theta_u) * sqrt(1-theta_u**2), 0, np.cos(theta_u)*sqrt(1-lmbd_u), 0],
                   [theta_u, 0, 0, 0]],
                  dims=[[2, 2], [2, 2]])

    # SOI+adsorber unitary
    V, *_ = unitaryV(theta_ru, lmbd_u, phi_u, False)

    # Permutation operator used to make U act on syst+out instead of syst+absorber
    P23 = tensor(fock(2, 0), fock(2, 0)) * tensor(fock(2, 0), fock(2, 0)).dag() + \
          tensor(fock(2, 1), fock(2, 0)) * tensor(fock(2, 0), fock(2, 1)).dag() + \
          tensor(fock(2, 0), fock(2, 1)) * tensor(fock(2, 1), fock(2, 0)).dag() + \
          tensor(fock(2, 1), fock(2, 1)) * tensor(fock(2, 1), fock(2, 1)).dag()

    # Total unitary
    VU = tensor(qeye(2), V) * (tensor(qeye(2), P23)*tensor(unitry, qeye(2))*tensor(qeye(2), P23))

    # Checks whether the function was sent 2 measurements or not
    if len(meas_u) == 2:
        # Calculates Kraus operators as Tr_e(Io|0><meas|*U)
        K_0 = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_u[0].dag()) * VU).ptrace([0, 1])
        K_1 = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_u[1].dag()) * VU).ptrace([0, 1])
        # print(f'k_1 is {K_1}')
        # print(K_1.dag()*K_1)

        # Kraus operators are returned in a list
        K = [K_0, K_1]
        # For checking they're proper Kraus operators
        # print('Kraus check:')
        # print(K[0].dag()*K[0] + K[1].dag()*K[1])
        # print(K)
    else:
        # Calculates a single Kraus operator as Tr_e(Io|0><meas|*U)
        # Used in fischer_cont() where measurement choice is already known
        K = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_u[0].dag()) * VU).ptrace([0, 1])
    return K, VU, V, unitry


def kraus_dot(theta_U, theta_RU, lmbd_U, phi_U, meas_U):
    # Function for the derivatives of the Kraus operators and unitary
    # Needed for calculation of the classical FI
    # When sent one measurement, it calculates the corresponding single Kraus derivative

    unitry_dot = Qobj([[-np.sin(theta_U)*sqrt(1-theta_U**2) - theta_U*np.cos(theta_U)*(1-theta_U**2)**(1/2), 0,
                        1j*np.cos(theta_U)*sqrt(1-lmbd_U), 0],
                       [0, 0, 0, 0],
                       [1j*np.cos(theta_U)*sqrt(1-theta_U**2) - 1j*np.sin(theta_U)*(1-theta_U**2)**(1/2), 0,
                        -np.sin(theta_U)*sqrt(1-lmbd_U), 0],
                       [1, 0, 0, 0]],
                     dims=[[2, 2], [2, 2]])

    # SOI+adsorber unitary
    V, *_ = unitaryV(theta_RU, lmbd_U, phi_U, False)

    # Permutation operator makes U act on syst+out instead of syst+absorber
    P23 = tensor(fock(2, 0), fock(2, 0))*tensor(fock(2, 0), fock(2, 0)).dag() + \
        tensor(fock(2, 1), fock(2, 0)) * tensor(fock(2, 0), fock(2, 1)).dag() + \
        tensor(fock(2, 0), fock(2, 1)) * tensor(fock(2, 1), fock(2, 0)).dag() + \
        tensor(fock(2, 1), fock(2, 1)) * tensor(fock(2, 1), fock(2, 1)).dag()

    # Total
    VU_diff = tensor(qeye(2), V)*(tensor(qeye(2), P23)*tensor(unitry_dot, qeye(2))*tensor(qeye(2), P23))

    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_U[0].dag()) * VU_diff).ptrace([0, 1])
        K_1_dot = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_U[1].dag()) * VU_diff).ptrace([0, 1])
        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot]
    else:
        # Calculates K_dot operator as Tr_e(Io|0><meas|*U_dot)
        # Used in fischer_cont() where measurement choice is already known
        K_dot = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_U[0].dag()) * VU_diff).ptrace([0, 1])

    return K_dot, VU_diff, V, unitry_dot


def measurement_choice():
    # Stage 2 (after initial estimation)
    meas = [fock(2, 0), fock(2, 1)]
    return meas


# Main function of the file
def Trajectory(id, setup):
    # Trajectory generation function. Change setup to alter the trajectory generation.

    # Unpacks setup
    N = setup['N']  # No. of samples to generate
    n = setup['n']  # No. of ancillary systems
    theta = setup['theta']  # True value of theta for trajectory generation
    lmbd = setup['lambda']  # Transition parameter, 0<=lmbd<=1
    phi = setup['phi']  # Phase parameter
    eps = setup['eps']  # Proportion of trajectory used in initial estimate

    # Variable that stores cumulative CFI from sampling method
    F = 0   # (1/N)*F gives the average FI

    # F_sqrd stores the cumulative sum of each contribution squared for calculating variance
    F_sqrd = 0

    # Variable for FI as 2nd derivative of the log-likelihood
    F_dm = 0

    ## Sample generation and argmax ##
    # Starts loop timer
    t0 = time.time()

    # Assume we've already localized a parameter to a region of the following size
    local = n**(-0.5 + eps)

    # Offset the absorber to slightly outside this region
    absorber = n**(-0.5 + 3*eps)

    # Splits the samples into samples used in the initial and final estimation
    n_init = int(np.floor(n ** (1-eps)))
    n_final = int(np.ceil(n ** (1-eps)))

    # Localizes theta using n**(1-eps) of the samples
    theta_rough, *_ = initial_est(theta, lmbd, phi, n_init)

    # Stage 2: measurement in the first fixed basis
    M = measurement_choice()   # Finds adaptive choice of next measurement
    # Finds the corresponding Kraus operators
    K, VU, *_ = kraus(theta, theta_rough, lmbd, phi, M)

    # Calculates the new stationary state based off the transition operator
    T = np.zeros((16, 16), dtype=complex)
    for i in range(16):
        for j in range(16):
            # Runs through the basis vectors in vector form
            r_j = np.zeros(16)
            r_j[j] = 1
            r_j = Qobj(r_j.transpose(), type='operator-ket')
            # Converts to operator form
            R_j = np.zeros((4, 4))
            R_j[j // 4, j % 4] = 1
            # Don't think this is correct
            # R_j = Qobj(R_j, dims=[[2, 2], [2, 2]])
            # Applies transition operator
            t = (K[0].data * R_j * (K[0].dag()).data + K[1].data * R_j * (K[1].dag()).data)
            # Converts back to vector
            Tj = np.hstack([t[l, :] for l in range(4)])
            # Takes ith component for the ij component of the Transition matrix
            T[i, j] = Tj[i]
    # Calculates its eigenvalues and eigenvectors; expect an evec with eval 1 corresponding to ss
    ev, vs = np.linalg.eig(T)
    r_ss = -vs[:, 0]
    # Converts that evec into operator form (the ss)
    rho_0 = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            rho_0[i, j] = r_ss[i*4 + j]
    rho_0 = Qobj(rho_0, dims=[[2, 2], [2, 2]])
    # Normalizes the trace
    rho_0 = rho_0 / rho_0.tr()

    # print(rho_0)
    # print(K[0]*rho_0*K[0].dag() + K[1]*rho_0*K[1].dag())

    # Mean photon number
    mpn = np.real_if_close((rho_0 * K[1].dag() * K[1]).tr())

    # Initializes the state of the s+a
    a, rho_0 = unitaryV(theta_rough, lmbd, phi, False)
    rho = rho_0

    # List to record which outcomes occurred
    x = [None] * n_final
    for j in np.arange(n_final):
        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()
        # p1 = (K[1] * rho * K[1].dag()).tr()

        # Choice of {0, 1} with prob {p(0), p(1)}
        x_j = np.random.choice([0, 1], p=[p0, 1 - p0])
        # print(p1)
        # print(K[1])

        # Records the outcome in x_s2
        x[j] = x_j

        # Updates the state by applying the measurement projection and normalising
        rho = K[x_j] * rho * K[x_j].dag()
        rho = rho / rho.tr()

    # Useful print statements for checking the percentage of 1s in the trajectory
    # print(f'Mean photon number: {mpn:.5f}')
    # print(f'Average photons in trajectory: {np.sum(x) / len(x)}')
    # print(x)

    # Calculates the mpn for 1 patterns based off the derivative formula
    expected_derivative = {}
    # Stationary state at theta_0 required for both this and the pattern_check function
    if True:
        # Finds the Kraus operators
        K_tht0, VU, *_ = kraus(theta, theta_rough, lmbd, phi, M)
        # Calculates the stationary state at theta=theta_0
        T = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            for j in range(16):
                # Runs through the basis vectors in vector form
                r_j = np.zeros(16)
                r_j[j] = 1
                r_j = Qobj(r_j.transpose(), type='operator-ket')
                # Converts to operator form
                R_j = np.zeros((4, 4))
                R_j[j // 4, j % 4] = 1
                # Don't think this is correct
                # R_j = Qobj(R_j, [[2, 2], [2, 2]])
                # Applies transition operator
                t = (K_tht0[0].data * R_j * (K_tht0[0].dag()).data + K_tht0[1].data * R_j * (K_tht0[1].dag()).data)
                # Converts back to vector
                Tj = np.hstack([t[l, :] for l in range(4)])
                # Takes ith component for the ij component of the Transition matrix
                T[i, j] = Tj[i]
        # Calculates its eigenvalues and eigenvectors; expect an evec with eval 1 corresponding to ss
        ev, vs = np.linalg.eig(T)
        r_ss = -vs[:, 0]
        # Converts that evec into operator form (the ss)
        rho_0_tht0 = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                rho_0_tht0[i, j] = r_ss[i * 4 + j]
        rho_0_tht0 = Qobj(rho_0_tht0, dims=[[2, 2], [2, 2]])
        # Normalizes the trace
        rho_0_tht0 = rho_0_tht0 / rho_0_tht0.tr()

    # Handles the 1s patterns
    if id == 0:
        # Calculates the numerical derivatives for the mu calculation
        h = 1e-6    # Small difference
        # Finds the Kraus' at theta+h
        K_small_diff, VU, *_ = kraus(theta+h, theta_rough, lmbd, phi, M)
        # Uses this to calculate the numerical derivative
        K_dot = [(1/h) * (K_small_diff[l] - K_tht0[l]) for l in range(len(K_tht0))]
        # Need to find the stationary state at a small difference from theta
        T = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            for j in range(16):
                # Runs through the basis vectors in vector form
                r_j = np.zeros(16)
                r_j[j] = 1
                r_j = Qobj(r_j.transpose(), type='operator-ket')
                # Converts to operator form
                R_j = np.zeros((4, 4))
                R_j[j // 4, j % 4] = 1
                # Don't think this is correct
                # R_j = Qobj(R_j, [[2, 2], [2, 2]])
                # Applies transition operator
                t = (K_small_diff[0].data * R_j * (K_small_diff[0].dag()).data +
                     K_small_diff[1].data * R_j * (K_small_diff[1].dag()).data)
                # Converts back to vector
                Tj = np.hstack([t[l, :] for l in range(4)])
                # Takes ith component for the ij component of the Transition matrix
                T[i, j] = Tj[i]
        # Calculates its eigenvalues and eigenvectors; expect an evec with eval 1 corresponding to ss
        ev, vs = np.linalg.eig(T)
        r_ss = -vs[:, 0]
        # Converts that evec into operator form (the ss)
        rho_0_small_diff = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                rho_0_small_diff[i, j] = r_ss[i * 4 + j]
        rho_0_small_diff = Qobj(rho_0_small_diff, dims=[[2, 2], [2, 2]])
        # Normalizes the trace
        rho_0_small_diff = rho_0_small_diff / rho_0_small_diff.tr()
        # Calculates the numerical derivative
        rho_dot = (rho_0_small_diff - rho_0_tht0)/h

        # Calculates the mu for each pattern of only 1s up to order 'order'
        order = 4
        mu = [None] * order
        for i in range(order):
            mu_terms = 0
            # Handles derivative on jth term; only two relevant terms
            for j in range(2):
                terms_product = qeye([2, 2])
                # Term held as a list
                term = [K_tht0[0]] * (i+1) + [rho_0_tht0] + [K_tht0[1].dag()] * (i+1)
                # Replaces an element with the derivative
                if j == 0:
                    term[i+1] = rho_dot
                elif j == 1:
                    term[i+2] = K_dot[1].dag()
                # Generates the multiplication of Kraus' and rho
                for k in range(len(term)):
                    terms_product = terms_product * term[k]
                # Adds that contribution to the sum
                mu_terms += terms_product.tr()
            # Stores the result in mu
            mu[i] = mu_terms

        # Calculates pattern 1 manually to check the loop is working properly
        mu_1 = (K_tht0[0]*rho_dot*K_tht0[1].dag() + K_tht0[0]*rho_0_tht0*K_dot[1].dag()).tr()

        # Calculates the expected values and stores them in a dictionary
        one_patterns = ['1', '11', '111', '1111']
        for i in range(len(one_patterns)):
            expected_derivative[one_patterns[i]] = absorber ** 2 * n * np.abs(mu[i]) ** 2
        print(f'Expected counts of 1 from the derivative formula: {absorber**2 * n * np.abs(mu_1)**2}')
        print(f'Expected counts of 1 from my loop: {expected_derivative["1"]}')
        print(f'Expected counts of 11 from my loop {expected_derivative["11"]}')
        print(f'Expected counts of 111 from my loop {expected_derivative["111"]}')
        print(f'Expected counts of 1111 from my loop {expected_derivative["1111"]}')

    # Analyses the patterns in x
    patterns, expected, wsp = pattern_check(x, rho_0_tht0, K, id, abs(theta-theta_rough))
    # test = np.abs((K[0]*rho_0*K[1].dag()).tr())**2 * n
    # print(f'Test: {test}')

    # Dummy value for the estimate
    theta_est = 1

    # Saves the trajectory analysis
    save3 = False
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    Model2 = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')
    # Toggles saving
    if save3:
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_excitations_full_analysis.csv'), 'a', newline='') as file:
            data = [setup['n'], eps, lmbd, phi, theta_rough] + [val for val in patterns.values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)
    # ===================
    ## Saving utility ##
    # ===================

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(
        '''Sample {} finished; tht_0 was {}. It took {} minutes and {:.1f} seconds. 
        Other statistics: mpn={:.5f}; Traj. avg.={:.5f}; Wsp={:.5f}'''.format(id+1, theta_rough, total_time // 60, total_time % 60, mpn, np.sum(x)/len(x), wsp))
    ## Sample generation and argmax ##
    # End of sample generation
    return F, F_sqrd, F_dm, theta_est, patterns, expected, expected_derivative


# Checks for patterns in the output list
def pattern_check(ones_list, stationary_state, ks, trajectory_id, local_u):
    # Creates the dictionary that stores the analysis
    analysis_x = {}

    # Initializes the first two patterns
    analysis_x['1'] = 0
    analysis_x['11'] = 0

    # Generates all possible patterns up to order n+2
    order_patterns = 3
    for i in range(0, 2 ** (order_patterns + 1)):
        analysis_x['1' + bin(i)[2:] + '1'] = 0

    # Copies the above dictionary to quickly initialize it
    expected_x = analysis_x.copy()

    # # Finds the indices of all ones
    # ones_loc = []
    # for i in range(len(ones_list)):
    #     if ones_list[i] == 1:
    #         ones_loc.append(i)

    # Fisher information calculation
    FI_patterns = 0

    # Calculates the mpn from a sum over expected patters
    expected_mpn = 0

    # Loop runs over each pattern in the dictionary above
    for key in analysis_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Ensures this is only done once
        if trajectory_id == 0:
            # Calculates the number of times we expect to see the pattern
            mu_pattern = stationary_state
            # Creates a superoperator for the transition operator
            T = sprepost(ks[0], ks[0].dag()) + sprepost(ks[1], ks[1].dag())
            # Creates a superoperator for the 2nd kraus
            J = sprepost(ks[0], ks[1].dag())
            # Runs through the pattern applying either T or J depending on the outcome 1 or 0
            for i in pattern:
                if i == 0:
                    mu_pattern = T(mu_pattern)
                elif i == 1:
                    mu_pattern = J(mu_pattern)
                else:
                    raise "How'd you get here?"

            expected_x[key] = np.abs(mu_pattern.tr())**2 * len(ones_list)

            # Adds the number of photons to the total sum of photons in detected patterns
            expected_mpn += (expected_x[key]/len(ones_list)) * np.sum(pattern)

            FI_patterns += 4 * (expected_x[key]/len(ones_list)) / (abs(local_u))**2

    # Ensures this is only printed once
    if trajectory_id == 0:
        print(f'Analytical result for the m.p.n: {expected_mpn}')
        print(f'Fisher information from patterns: {FI_patterns}')

    # Counts the 1s in all observed patterns
    weighted_sum_patterns = 0
    # Loop runs over each pattern in the dictionary above
    for key in analysis_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Then adds a number of padding zeroes either side of the pattern
        pad = 10
        for i in range(pad):
            pattern.reverse()
            pattern.append(0)
            pattern.reverse()
            pattern.append(0)

        # Loops over the trajectory and identifies how many times the pattern occurs; this misses the tails of the list
        for i in range(len(ones_list) - len(pattern)):
            if ones_list[i:i+len(pattern)] == pattern:
                analysis_x[key] += 1
                weighted_sum_patterns += np.sum(pattern)/len(ones_list)

    return analysis_x, expected_x, weighted_sum_patterns


#==========
## Setup ##
#==========
# Global parameter values
# Check Trajectory() for 2nd derivative FI step size
setup = {
    'N': 1000,  # No. of samples to generate
    'n': 5000,  # No. of ancillary systems
    'theta': 0.2,  # True value of theta for trajectory generation
    'lambda': 0.8,  # Transition parameter, 0<=lmbd<=1
    'phi': pi / 4,  # Phase parameter
    'eps': 0.1,  # Prop. of traj. to use in initial est.
}
#==========
## Setup ##
#==========

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

    # Useful information
    print(f'Running with N={N}, n={n} and epsilon={setup["eps"]};\n'
          f'')

    # Pool object to run the multiprocessing
    pool = Pool()
    results = pool.starmap(Trajectory, zip(ids, repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    #===================#
    ## QFI calculation ##
    #===================#
    if True:
        # Initial measurement choice
        M = [(fock(2, 0) + fock(2, 1)).unit(), (fock(2, 0) - fock(2, 1)).unit()]
        # Kraus operators for calculating the QFI
        K, U = kraus_2by2(theta, lmbd, phi, M)
        K_dot, U_dot = kraus_dot_2by2(theta, lmbd, phi, M)
        # Finds the 2by2 ss
        r = ss(theta, lmbd, phi, False)
        rho_ss = (1/2) * (qeye(2) + r[0, 0]*sigmax() + r[1, 0]*sigmay() + r[2, 0]*sigmaz())
        # print(rho_ss)

        # Finds the Moore-Penrose inverse of Id-T=R
        Id_T = Id_T_heisenberg(theta, lmbd, phi, False)
        R = np.linalg.pinv(Id_T)
        # print(R)

        # Finds the term that R is applied to when the gauge condition is satisfied
        A_sum = K_dot[0].dag()*K[0] + K_dot[1].dag()*K[1]
        A = Qobj((A_sum-A_sum.dag()) / (2*1j))
        # print("A")
        # print(A)
        # print(np.matrix([[(A*sigmax()).tr()],
        #                  [(A*sigmay()).tr()],
        #                  [(A*sigmaz()).tr()],
        #                  [A.tr()]]))
        # print("A check: {}".format((rho_ss*A).tr()))

        # Finds a - the mean of A, important since the gauge condition isn't naturally satisfied by A
        a = (rho_ss*A).tr()
        # print("a: {}".format(a))

        # Finds the new term to apply R to
        A_tilde = (K_dot[0] + 1j*a*K[0]).dag()*K[0] + (K_dot[1] + 1j*a*K[1]).dag()*K[1]
        A_tilde = Qobj((A_tilde - A_tilde.dag()) / (2 * 1j))
        # print('A_tilde')
        # print(A_tilde)
        # print("A_tilde check: {}".format((rho_ss*A_tilde).tr()))

        # Vectorizes A_tilde for applying R
        A_x = (A_tilde*sigmax()).tr()
        A_y = (A_tilde*sigmay()).tr()
        A_z = (A_tilde*sigmaz()).tr()
        A_I = A_tilde.tr()
        A_tilde = np.matrix([[A_x],
                             [A_y],
                             [A_z],
                             [A_I]])
        # Applies R
        RA = R*A_tilde
        # print(RA)

        # Converts this back from the vector from
        RA = (1/2)*(RA[0, 0]*sigmax() + RA[1, 0]*sigmay() + RA[2, 0]*sigmaz() + RA[3, 0]*qeye(2))
        # Initialises the QFI
        Q = 0
        # Adds contribution from each Kraus operator
        for i in np.arange(len(K)):
            # print((rho_ss*(K_dot[i] + 1j*a*K[i]).dag()*(K_dot[i] + 1j*a*K[i])).tr())
            # print(2 * (Qobj(np.imag(K[i] * rho_ss * (K_dot[i] + 1j*a*K[i]).dag())) * RA).tr())
            Q += 4 * ( (rho_ss*(K_dot[i] + 1j*a*K[i]).dag()*(K_dot[i] + 1j*a*K[i])).tr() +
                       2*( (Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()) -
                            Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()).dag()) / (2*1j) * RA).tr() )
        print('QFI per step: {}'.format(np.real_if_close(Q)))
    #===================#
    ## QFI Calculation ##
    #===================#

    #====================
    ## Approximate FIs ##
    #====================
    # Initializes variables to unpack from Pool
    F = 0
    F_sqrd = 0
    F_dm = np.zeros(N)
    theta_est = np.zeros(N)
    # Unpacks them
    for i in np.arange(len(results)):
        result = results[i]
        F += result[0]
        F_sqrd += result[1]
        F_dm[i] = result[2]
        theta_est[i] = result[3]

    # Calculates the FI as the average FI of the samples
    F_chains = (1/N) * F
    # The variance of FI of samples
    F_var = (1/N)*F_sqrd - F_chains**2
    # The error associated with this variance
    error = sqrt(F_var/N)
    # Calculates the FI as the average of derivative method FIs
    F_derivative_method = (1/N) * np.sum(F_dm)
    # print('FI successfully calculated')
    # print('Sample FI is {}'.format(F_chains))
    # print('Error is {}'.format(error))
    # print('2nd derivative FI is {}'.format(F_derivative_method))
    #
    # # Calculates FI from MLE estimates
    # print('Argmax successfully completed')
    # FI inversely proportional to the sample variance
    # Using true parameter value to calculate sample variance
    sampleVar = (1/N) * np.sum((theta_est-theta)**2)
    sampleFI = 1/sampleVar
    print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))
    #====================
    ## Approximate FIs ##
    #====================

    # Theta estimates
    print('Theta estimates: {}'.format(theta_est))

    # Expected counts of patterns
    result = results[0]
    result_expected = result[5]
    print(f'Expected counts of patterns:\n'
          f'{result_expected}')
    # Shows the actual results in the first 5 trajectories
    if len(results) > 5:
        for i in range(5):
            result = results[i]
            print(f'Actual counts in trajectory {i+1}:\n'
                  f'{result[4]}')
    else:
        result = results[0]
        print(f'Actual counts in trajectory {1}:\n'
              f'{result[4]}')

    #===================
    ## Saving utility ##
    #===================
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    Model2 = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')

    # Saves the data
    save = False
    print('Saving FI: {}'.format(save))
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_excitations_full.csv'), 'a', newline='') as file:
            data = [sampleFI, setup['N'], setup['n'], setup['eps'], error, lmbd, phi]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Saves the theta estimates
    save = False
    print('Saving thetas: {}'.format(save))
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_excitations_full_thetas.csv'), 'a', newline='') as file:
            data = [sampleFI, setup['N'], setup['n'], setup['eps'], error, lmbd, phi]
            thetas = theta_est
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)
            z.writerow(thetas)

    # Save expected values
    save = False
    print(f'Saving expected counts: {save}')
    # Toggles whether to save
    if save:
        result = results[0]
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_excitations_full_expected.csv'), 'w', newline='') as file:
            rows = ['n', 'lambda', 'phi', 'epsilon'] + [key for key in result[5].keys()]
            data = [setup['n'], lmbd, phi, setup['eps']] + [val for val in result[5].values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(rows)
            z.writerow(data)

    # Save expected values of ones
    save = False
    print(f'Saving expected counts of 1s: {save}')
    # Toggles whether to save
    if save:
        result = results[0]
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_excitations_expected_1s.csv'), 'w', newline='') as file:
            rows = ['n', 'lambda', 'phi', 'absorber offset'] + [key for key in result[6].keys()]
            data = [setup['n'], lmbd, phi, setup['absorber offset']] + [val for val in result[6].values()]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(rows)
            z.writerow(data)

