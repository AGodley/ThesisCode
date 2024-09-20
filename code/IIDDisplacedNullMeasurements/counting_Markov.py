# Alfred Godley
# Implementing the new scheme based on counting measurements in different bases in the old input-output setting

# Standard imports
import sys
from math import pi, sqrt  # For general maths
import numpy as np      # For arrays and matrices
import matplotlib.pyplot as plt    # For plotting graphs
from scipy.optimize import minimize, brute     # For argmax/maximum likelihood estimation
from qutip import *     # Package for quantum objects

# Imports from the initial estimation file
from initial_estimation_2 import initial_est
from initial_estimation_2 import kraus as kraus_2by2
from initial_estimation_2 import kraus_dot as kraus_dot_2by2

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
        # Kraus operators are returned in a list
        K = [K_0, K_1]
        # For checking they're proper Kraus operators
        # print('Kraus check:')
        # print(K[0].dag()*K[0] + K[1].dag()*K[1])
    else:
        # Calculates a single Kraus operator as Tr_e(Io|0><meas|*U)
        # Used in fischer_cont() where measurement choice is already known
        K = (tensor(qeye(2), qeye(2), fock(2, 0)*meas_u[0].dag()) * VU).ptrace([0, 1])

    return K, VU, V, unitry


def kraus_dot(theta_U, theta_RU, lmbd_U, phi_U, meas_U, gauge_U):
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

    # Kraus and unitary operator for gauge condition
    K, VU, *_ = kraus(theta_U, theta_RU, lmbd_U, phi_U, meas_U)
    # Including gauge condition
    VU_diff = VU_diff + 1j*gauge_U*VU

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
    # Determines which measurement to apply

    # Stage 2 (after initial estimation)
    meas = [fock(2, 0), fock(2, 1)]
    return meas


def mle_initial(tau, lmbd_val, phi_val, x_init):
    # Function for ML estimation
    # Essentially the same process as sample generation, but theta may now be non-zero
    # measurements = [measure_system, system_outcome]

    # # Useful print statement
    # print(tau, tht_rmle)

    # Initial estimation based on rough estimation
    # Initiates MLE probability and state
    log_p_mle = 0
    rho_mle = fock_dm(2, 0)
    # Possible measurement choices
    M = [(fock(2, 0) + fock(2, 1)).unit(), (fock(2, 0) - fock(2, 1)).unit()]
    for i_mle in np.arange(len(x_init)):
        # Finds which measurement was selected
        meas_mle = M[x_init[i_mle]]

        # Calculates the corresponding Kraus operators
        K_mle, U_mle = kraus_2by2(tau, lmbd_val, phi_val, [meas_mle])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # Updates log prob. with log prob. of this Kraus occurring
        log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

    # Returns minus value as the argmax finds minima
    return -log_p_mle


def mle_stage2(tau, tht_rmle, lmbd_val, phi_val, rho_init, x_init, x_s2):
    # Function for ML estimation
    # Essentially the same process as sample generation, but theta may now be non-zero
    # measurements = [measure_system, system_outcome]

    # # Useful print statement
    # print(tau, tht_rmle)

    # Takes into account the initial estimation
    log_p_mle = -mle_initial(tau, lmbd_val, phi_val, x_init)

    # 2nd stage estimation
    # Initial state
    rho_mle = rho_init
    # Finds the measurements used in this stage
    measurements_mle = measurement_choice()
    for i_mle in np.arange(len(x_s2)):
        # Selects the measurement that occurred
        meas_mle = measurements_mle[x_s2[i_mle]]

        # Calculates the corresponding Kraus operators
        K_mle, VU_mle, *_ = kraus(tau, tht_rmle, lmbd_val, phi_val, [meas_mle])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # log(0) raises a ZeroDivision error; this ignores it
        with np.errstate(divide='ignore'):
            # Updates log prob. with log prob. of this Kraus occurring
            log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

    # Returns minus value as the argmax finds minima
    return -log_p_mle


def Trajectory(id, setup):
    # Trajectory generation function. Change setup to alter the trajectory generation.

    # Unpacks setup
    N = setup['N']  # No. of samples to generate
    n = setup['n']  # No. of ancillary systems
    theta = setup['theta']  # True value of theta for trajectory generation
    lmbd = setup['lambda']  # Transition parameter, 0<=lmbd<=1
    phi = setup['phi']  # Phase parameter
    eps = setup['eps']  # Proportion of trajectory used in initial estimate

    # Splits n as specified by eps
    n_init = int(np.floor(n ** (1-eps)))
    n_s2 = n - n_init   # Samples used in second stage estimation

    # Variable that stores cumulative CFI from sampling method
    F = 0   # (1/N)*F gives the average FI

    # F_sqrd stores the cumulative sum of each contribution squared for calculating variance
    F_sqrd = 0

    # Variable for FI as 2nd derivative of the log-likelihood
    F_dm = 0

    ## Sample generation and argmax ##
    # Starts loop timer
    t0 = time.time()

    # Initial estimation
    theta_rough, x_rough = initial_est(theta, lmbd, phi, n_init)

    # Initial state; approximation of s+a stationary state using the rough estimate
    a, rho_0 = unitaryV(theta_rough, lmbd, phi, False)

    # Initializes the state of the s+a
    rho = rho_0

    # Initial estimation localizes theta to a region of the size below around the rough estimate
    local = n ** (-1 / 2 + eps)

    # We apply the absorber at a theta value sufficiently far from the rough estimate
    absorber = theta_rough + n ** (-1 / 2 + 3 * eps)

    # Stage 2: measurement in the first fixed basis
    M = measurement_choice()   # Finds adaptive choice of next measurement
    # Finds the corresponding Kraus operators
    K, VU, *_ = kraus(theta, absorber, lmbd, phi, M)
    # List to record which outcomes occurred
    x_s2 = [None] * n_s2
    for j in np.arange(n_s2):
        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()

        # Choice of {0, 1} with prob {p(0), p(1)}
        x = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Records the outcome in x_s2
        x_s2[j] = x

        # Updates the state by applying the measurement projection and normalising
        rho = K[x] * rho * K[x].dag()
        rho = rho / rho.tr()

    print(x_s2)
    # Brute force mle
    bnds = ([(theta_rough - 1.5*local, theta + 1.5*local)])  # Interval the argmax investigates
    result, fval, grid, jout = brute(mle_stage2, ranges=bnds, Ns=50, full_output=True,
                                     args=(theta_rough, lmbd, phi, rho_0, x_rough, x_s2), finish=None)
    theta_est = result

    # # plots MLE function on a grid
    # fig = plt.figure()
    # plt.plot(grid, jout)    # Uses the values from the brute force estimation
    # plt.show()

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0  # In seconds
    print(
        '''Sample {} finished. It took {} minutes and {} seconds. 
        The initial and final estimates of theta were {} and {} respectively'''.format(id+1, total_time // 60, total_time % 60, theta_rough, theta_est))
    ## Sample generation and argmax ##
    # End of sample generation
    return F, F_sqrd, F_dm, theta_est


#==========
## Setup ##
#==========
# Global parameter values
# Check Trajectory() for 2nd derivative FI step size
setup = {
    'N': 50,  # No. of samples to generate
    'n': 2000,  # No. of ancillary systems
    'theta': 0.2,  # True value of theta for trajectory generation
    'lambda': 0.8,  # Transition parameter, 0<=lmbd<=1
    'phi': pi / 4,  # Phase parameter
    'eps': 0.05,  # Prop. of traj. to use in initial est.
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
    print(f'Running with N={N}, n={n} and eps={eps}.\n'
          f'Using {int(np.floor(n ** (1-eps)))} samples in the initial estimation;\n'
          f'{n-int(np.floor(n ** (1-eps)))} samples in the second stage.')

    # Pool object to run the multiprocessing
    pool = Pool(9)
    results = pool.starmap(Trajectory, zip(ids, repeat(setup)))
    pool.close()
    pool.join()
    print('Samples successfully generated')

    #========
    ## QFI ##
    #========
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

    '''
    # Alternative method for calculating the Moore-Penrose inverse
    # Uses a singular value decomposition approach
    R = np.linalg.svd(Id_T)
    U1 = R[0]
    V1 = R[2]
    D = R[1]
    R = V1.I*np.diag([1/D[0], 1/D[1], 1/D[2], 0])*U1.I
    '''

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
    print('QFI per step: {}'.format(Q))
    #========
    ## QFI ##
    #========

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
    print('FI successfully calculated')
    print('Sample FI is {}'.format(F_chains))
    print('Error is {}'.format(error))
    print('2nd derivative FI is {}'.format(F_derivative_method))

    # Calculates FI from MLE estimates
    print('Argmax successfully completed')
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

    #===================
    ## Saving utility ##
    #===================
    # Finds directory for the project
    adaptiveMeasurementSimulation = (Path.cwd()).parents[1]
    Model2 = adaptiveMeasurementSimulation.joinpath('data').joinpath('countingMeasurements')

    # Saves the data
    save = True
    print('Saving FI: {}'.format(save))
    # Toggles whether to save
    if save:
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov.csv'), 'a', newline='') as file:
            data = [sampleFI, setup['N'], setup['n'], setup['eps'], error, lmbd, phi]
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)

    # Saves the theta estimates
    save2 = True
    print('Saving thetas: {}'.format(save))
    # Toggles whether to save
    if save2:
        # Opens the file in append mode
        with open(Model2.joinpath('counting_Markov_thetas.csv'), 'a', newline='') as file:
            data = [sampleFI, setup['N'], setup['n'], setup['eps'], error, lmbd, phi]
            thetas = theta_est
            z = csv.writer(file)
            # Appends data onto the file
            z.writerow(data)
            z.writerow(thetas)
    #===================
    ## Saving utility ##
    #===================