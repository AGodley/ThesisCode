# Alfred Godley, 08/06/2023
# Given n total samples used, we use n**(1-eps) in the initial estimation
# This should localize the initial estimate to a region of size n**(-0.5+eps)
# Use this to inform the grid spacing in thts
# At n=100000 & eps=0.1, expect a region of size 0.01 for instance

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


# Calculates the expected number of 1s
def expected(theta, lmbd, phi, n, M, rho):
    K, U = kraus(theta, lmbd, phi, M)
    exp = (rho * K[1].dag() * K[1]).tr() * n
    return exp


def initial_est(theta, lmbd, phi, n):
    # Initial state
    rho_0 = fock_dm(2, 0)

    # Measurement choice
    M = [fock(2, 0), fock(2, 1)]

    # Sample generation and argmax #
    # Resets to initial state
    rho = rho_0

    # Stores measurements for calculating FI and MLE
    M_store = [None] * n

    K, U = kraus(theta, lmbd, phi, M)

    x = np.zeros(n, dtype=int)
    for j in np.arange(n):
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

    # Forces it into the stationary state
    i = 0
    while i < 8:
        rho = K[0] * rho * K[0].dag()
        rho = rho / rho.tr()
        i += 1

    # Estimate of the expected number of 1s in the trajectory, calculated assuming we've converged to ss
    exp = expected(theta, lmbd, phi, n, M, rho)

    # Calculates the estimate at a range of values
    thts = np.linspace(0.15, 0.25, 100)
    expt = [expected(tht, lmbd, phi, n, M, rho) for tht in thts]


    # Recorded number of 1s in the trajectory
    actual = np.sum(x)


    # Finds the value closest to the actual value
    theta_est = thts[0]
    for i in range(len(thts) - 1):
        if abs(actual - expt[i+1]) < abs(actual - expt[i]):
            theta_est = thts[i+1]

    print(thts)
    print(exp, actual, x[-11:-1])
    print(actual, theta_est)

    # Simple plot
    fig, ax = plt.subplots()
    ax.plot(thts, expt)
    plt.show()

    # End of sample generation
    return theta_est, x


if __name__ == '__main__':
    samples = 100000
    eps = 0.1
    used = int(samples**(1-eps))
    initial_est(0.2, 0.8, pi/4, used)
