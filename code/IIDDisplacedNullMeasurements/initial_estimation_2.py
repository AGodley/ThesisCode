# Alfred Godley, 23/07/2022
# v2: Slight changes to include the part of the trajectory used in initial estimation in the argmax later on

from math import pi, sqrt, sin  # For general maths
import numpy as np      # For arrays and matrices
from scipy.optimize import *     # For argmax
import matplotlib.pyplot as plt    # For plotting graphs
from qutip import *     # For quantum objects
import csv   # Comma separated values
from AbsorberFunctions import ss, TO, unitaryV


# Kraus operator function without absorber, used for calculating QFI
def kraus(theta_u, lmbd_u, phi_u, meas_u):
    # Not a true unitary as only it's action on |000>, |100>, |010> and |110> states are important
    unitry = Qobj([[np.cos(theta_u)*sqrt(1-theta_u**2), 0, 1j*np.sin(theta_u)*sqrt(1-lmbd_u), 0],
                   [0, 0, sqrt(lmbd_u)*np.exp(1j*phi_u), 0],
                   [1j*np.sin(theta_u) * sqrt(1-theta_u**2), 0, np.cos(theta_u)*sqrt(1-lmbd_u), 0],
                   [theta_u, 0, 0, 0]],
                  dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_u) == 2:
        # Calculates Kraus operators as Tr_e(Io|0><meas|*U)
        K_0 = (tensor(qeye(2), fock(2, 0)*meas_u[0].dag()) * unitry).ptrace([0])
        K_1 = (tensor(qeye(2), fock(2, 0)*meas_u[1].dag()) * unitry).ptrace([0])
        # Kraus operators are returned in a list
        K = [K_0, K_1]
        # For checking they're proper Kraus operators
        # print('Kraus check:')
        # print(K[0].dag()*K[0] + K[1].dag()*K[1])
    else:
        # Used in fischer_cont() where measurement choice is already known
        K = (tensor(qeye(2), fock(2, 0)*meas_u[0].dag()) * unitry).ptrace([0])
    return K, unitry


# Derivative of Kraus operator function without absorber, used for calculating QFI
def kraus_dot(theta_U, lmbd_U, phi_U, meas_U):
    unitry_dot = Qobj([[-np.sin(theta_U)*sqrt(1-theta_U**2) - theta_U*np.cos(theta_U)*((1-theta_U**2)**(-1/2)), 0,
                        1j*np.cos(theta_U)*sqrt(1-lmbd_U), 0],
                       [0, 0, 0, 0],
                       [1j*np.cos(theta_U)*sqrt(1-theta_U**2) - 1j*np.sin(theta_U)*((1-theta_U**2)**(-1/2)), 0,
                        -np.sin(theta_U)*sqrt(1-lmbd_U), 0],
                       [1, 0, 0, 0]],
                      dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace([0])
        K_1_dot = (tensor(qeye(2), fock(2, 0)*meas_U[1].dag()) * unitry_dot).ptrace([0])
        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot] #
    else:
        # Used in fischer_cont() where measurement choice is already known
        K_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace([0])
    return K_dot, unitry_dot


# Creates a function for ML estimation
# Essentially the same process as sample generation, but theta may now be non-zero
# args = (theta, lmbd, phi, x[i]-a sample, rho_0)
# measure_system_mle = [measure_system, system_outcome]
def mle(tau, lmbd_val, phi_val, measurements, rho_init):
    # Initiates MLE probability and state
    log_p_mle = 0
    rho_mle = rho_init

    for i_mle in np.arange(len(measurements)):
        # Finds adaptive choice of next measurement
        meas_mle = measurements[i_mle]
        # Calculates the corresponding Kraus operators
        K_mle, U_mle = kraus(tau, lmbd_val, phi_val, [meas_mle])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # Updates log prob. with log prob. of this Kraus occurring
        log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

    # Returns minus value as there is only a minimise argmax function
    return -log_p_mle


def initial_est(theta, lmbd, phi, n):
    # Initial state
    rho_0 = fock_dm(2, 0)

    # Measurement choice
    M = [(fock(2, 0) + fock(2, 1)).unit(), (fock(2, 0) - fock(2, 1)).unit()]

    # Estimates of theta from the argmax
    theta_est = 0  # Vector of estimates

    # Sample generation and argmax #
    # Resets to initial state
    rho = rho_0

    # Stores measurements for calculating FI and MLE
    M_store = [None] * n

    x = np.zeros(n, dtype=int)
    for j in np.arange(n):
        K, U = kraus(theta, lmbd, phi, M)

        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()
        # Choice of {0, 1} with prob {p(0), p(1)}
        x[j] = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Stores which measurement occurred for FI calculation and measurement angles plot
        M_store[j] = M[x[j]]
        # print('Measurement choice {}'.format(j + 1))
        # print(M_store[j])

        # Updates the state by applying the measurement projection and normalising
        rho = K[x[j]] * rho * K[x[j]].dag()
        rho = rho / rho.tr()

    # # Calculates the MLE for the sample
    # # niter controls the number of attempts at estimation it tries
    # # niter_success stops the process if it gets the same result the specified no. of times in a row
    # # T - temperature parameter; should be comparable to step between minima
    # # MLE
    # bnds = [(theta - 0.2, theta + 0.2)]  # Interval the argmax investigates
    # result = minimize(mle, x0=np.array(theta), bounds=bnds, method='Nelder-Mead',
    #                        args=(lmbd, phi, M_store, rho_0))
    # theta_est = result.x[0]
    # # brute force on outliers
    # if abs(result.x-theta) > 0.03:
    #     result, fval, grid, jout = brute(mle, ranges=bnds, Ns=400, full_output=True,
    #                                      args=(lmbd, phi, M_store, rho_0))
    #     theta_est = result[0]

    # Pure Brute Force
    bnds = [(theta - 0.1, theta + 0.1)]  # Interval the argmax investigates
    result, fval, grid, jout = brute(mle, ranges=bnds, Ns=50, full_output=True,
                                     args=(lmbd, phi, M_store, rho_0), finish=None)
    theta_est = result
    # End of sample generation
    return theta_est, x
