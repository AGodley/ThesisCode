# Alfred Godley
# 12/10/2021
# v2: Adapting v1 to run multiple times, allowing us to plot the prob. dist. of the estimator
# v3: Adapting v2 to calculate FI as well
# v4: After meeting with Madalin. Still working on calculating the FI of the samples
# adaptive: Adding in adaptive measurements, requires updating to use qutip
# adaptive v2: Trying to avoid using unitaries
# adaptive v3: Now trying to change where the Fischer calculation is, so I can store the measurements from a sample,
# uses them to calculate the FI and then do the same for the next measurement
# adaptive v4: In v3 I renormalized A_1 by storing all the renormalization constants in norm_fac. In this version,
# I'm just going to keep multiplying A by 2 to keep it normalized. This will take away the factor in front of A_1
# adaptive v5: Trying to fix problem with Fischer contributions
# adaptive v6: Optimizing and changing to log prob for argmax
# adaptive v7: Changing minimize function
# adaptive v8: Changing unitary and added plot of measurement angle
# adaptive v9: Changing how the Kraus operators are calculated and neatening up the code for Madalin
# adaptive v10: Making it so you can swap to regular measurements, added a new parameter strat for this
# adaptive v11: Adding in system measurement, fixing issue of where to store the system measurement
# adaptive v12: Fixed errors in system measurement, and changing measurement angle plot
# adaptive v13: Added sample variance calculation and runs adaptive/adaptive w system measure simultaneously
# angle_finder: Adapted to find optimal regular measurement angle

from math import pi, sqrt   , sin  # For general maths
import numpy as np      # For arrays and matrices
from scipy.optimize import *     # For argmax
import matplotlib.pyplot as plt    # For plotting graphs
from qutip import *     # For quantum objects
import csv   # Comma separated values

# Creates a function for calculating Kraus operators
# Also returns the corresponding pseudo-unitary
# Kraus operators are used to update the state, unitary is used to calculate next measurement
# Inputs are respectively, the param of interest, lambda, phi and a list containing the measurements
# adaptive_v3: added functionality to send 1 measurement for use when calculating classical FI
def kraus(theta_u, lmbd_u, phi_u, meas_U):
    # Not a true unitary as only it's action on |00> and |10> state are important
    unitry = Qobj([[np.cos(theta_u) * sqrt(1 - theta_u ** 2), 0, 1j*np.sin(theta_u) * sqrt(1 - lmbd_u), 0],
                   [0, 0, sqrt(lmbd_u) * np.exp(1j * phi_u), 0],
                   [1j*np.sin(theta_u) * sqrt(1 - theta_u**2), 0, np.cos(theta_u) * sqrt(1 - lmbd_u), 0],
                   [theta_u, 0, 0, 0]],
              dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates Kraus operators as Tr_e(Io|0><meas|*U)
        K_0 = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry).ptrace(0)
        K_1 = (tensor(qeye(2), fock(2, 0)*meas_U[1].dag()) * unitry).ptrace(0)
        # Kraus operators are returned in a list
        K = [K_0, K_1]
        # For checking they're proper Kraus operators
        # print(K[0].dag()*K[0]+K[1].dag()*K[1])
    else:
        # Used in fischer_cont() where measurement choice is already known
        K = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry).ptrace(0)
    return K, unitry


# Creates a function for the derivatives of the Kraus operators and unitary
# Only important at theta=0
# Needed for calculation of the classical FI and A_1
# adaptive_v3: added functionality to send 1 measurement and calc. the corresponding Kraus for calc. FI
def kraus_dot(lmbd_U, meas_U):
    unitry_dot = Qobj([[0, 0, 1j*sqrt(1-lmbd_U), 0],
                       [0, 0, 0, 0],
                       [1j, 0, 0, 0],
                       [1, 0, 0, 0]],
                      dims=[[2, 2], [2, 2]])
    # Checks whether the function was sent 2 measurements or not
    if len(meas_U) == 2:
        # Calculates K_dot operators as Tr_e(Io|0><meas|*U_dot)
        K_0_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace(0)
        K_1_dot = (tensor(qeye(2), fock(2, 0)*meas_U[1].dag()) * unitry_dot).ptrace(0)
        # Once again returned in a list
        K_dot = [K_0_dot, K_1_dot]
    else:
        # Used in fischer_cont() where measurement choice is already known
        K_dot = (tensor(qeye(2), fock(2, 0)*meas_U[0].dag()) * unitry_dot).ptrace(0)
    return K_dot, unitry_dot


# Generates next choice of measurement
# strat_m decides between adaptive and regular measurements
def measurement_choice(pi_mc, U_m, U_dot_m, A_1_m, j_m, strat_m):
    # Regular +/- measurements
    if strat_m == 'reg':
        meas = [(1 / sqrt(2)) * (basis(2, 0) + basis(2, 1)), (1 / sqrt(2)) * (basis(2, 0) - basis(2, 1))]

    # Adaptive measurements
    elif strat_m == 'adapt':
        # The pi_m the function receives is the <e|A|e> from the last step
        # Begins by tensoring this with a new |0><0|
        A = tensor(pi_mc, fock_dm(2, 0))
        # A has contribution from A_1 for j>0, which this statement adds
        if j_m == 0:
            # A_1 calculated outside this function as it doesn't vary
            A = A_1_m
        else:
            # Choice of renormalisation multiplies A by 2 at each stage
            # This removes (1/2)^n term that's normally in front of A_1 term
            A = 2 * U_m * A * U_m.dag()
            A = A_1_m + A
            # A_norm = A / (-(A*A).tr()/2) # Alternative renormalisation

        # Traces out the system of interest producing B
        B = A.ptrace(1)

        # Checks whether B=0, if so returns +/- basis measurements
        # Otherwise, calculates components of B's bloch vec. and uses these to determine an orthogonal measurement
        if np.sum(abs(B.data)) < 1e-8:
            meas = [(1 / sqrt(2)) * (basis(2, 0) + basis(2, 1)), (1 / sqrt(2)) * (basis(2, 0) - basis(2, 1))]
            # print('Measurement choice')
            # print(meas)
        else:
            # A/B are anti-hermitian, so the imaginary part of B.sigx gives rx
            rx = np.imag((B * sigmax()).tr())
            # print('rx')
            # print(rx)
            ry = np.imag((B * sigmay()).tr())
            # print('ry')
            # print(ry)
            # ~Possible change here. Madalin agreed that this is working, but he suggested a better way that calculates
            # the projections directly
            C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
            ev, meas = C.eigenstates()
        # function returns two measurements
        # loop then chooses one of these and sends <e|pi_m|e> to this function to determine the next measurement
        pi_mc = A
    return meas, pi_mc


# Calculates measurement choice for measuring the system of interest at the end
def system_Measurement(pi_n):
    # Can use the same method as above
    rx = np.imag((pi_n * sigmax()).tr())
    ry = np.imag((pi_n * sigmay()).tr())
    C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
    ev, meas = C.eigenstates()
    return meas


# Calculates the Fischer information of a sample based on the measurements that occurred
def fischer_contr(rho_0_fi, n_fi, lmbd_fi, phi_fi, meas_fi, measure_system_fi):
    f = 0
    for j in np.arange(n_fi):
        K_j, U_fi = kraus(0, lmbd_fi, phi_fi, [meas_fi[j]])
        # K_total added for system measurement
        K_total = K_j / (K_j * rho_0_fi).tr()
        # Initially calculates K_j_dot and finds associated renormalisation constant 1/Tr(K_j*rho)
        K_total_dot, U_fi = kraus_dot(lmbd_fi, [meas_fi[j]])
        K_total_dot = K_total_dot / (K_j * rho_0_fi).tr()
        # Then multiplies by the product K_n...K_(j-1) with their renormalisation constants
        for k in np.arange(j + 1, n_fi):
            K_k, U_fi = kraus(0, lmbd_fi, phi_fi, [meas_fi[k]])
            K_total_dot = K_k * K_total_dot / (K_k * rho_0_fi).tr()
            K_total = K_k * K_total / (K_k * rho_0_fi).tr()

        # Determines whether we're measuring the system
        if measure_system_fi[0]:
            msyst = measure_system_fi[1] * measure_system_fi[1].dag()
            # Adds the total contribution from derivative on jth Kraus operator to f
            f += ((msyst * K_total * rho_0_fi * K_total.dag()).tr())**(-1) \
                 * (2 * np.real((msyst * K_total_dot * rho_0_fi).tr()))
        else:
            # Adds the total contribution from derivative on jth Kraus operator to f
            f += (2 * np.real((K_total_dot * rho_0_fi).tr()))
    f = f**2
    return f


# Main
# Global parameter values
N = 10000        # No. of samples to generate
n = 20         # No. of ancillary systems
theta = 0       # True value of theta for simulation
lmbd = 0.1      # Transition parameter, 0<=lmbd<=1
phi = 3*pi/4     # Phase parameter

# Measurement strategy
strat = 'reg'     # should be either reg or adapt

# Measuring the system
# With new simultaneous adapt/adapt w measure leave True
# Set to false for regular measurement
measure_system = False

# Loop trys a range of angles in order to find the ideal angle
ang = np.linspace(0, pi, 50)
F_ang = np.zeros(len(ang))
for k in np.arange(len(ang)):
    print('On angle={}'.format(ang[k]))

    # Initial state
    rho_0 = fock_dm(2, 0)
    # Initial measurement choice
    M = [(fock(2, 0) + np.exp(1j*ang[k])*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j*ang[k])*fock(2, 1)).unit()]

    # Kraus and unitary operators for calculating A_1
    K, U = kraus(theta, lmbd, phi, M)

    # Initialise data matrix that holds samples, row i gives outcomes from sample i
    x = np.zeros((N, n), dtype=int)
    # Initialise FI store
    # F is the cumulative classical FI, so (1/N)*F gives the average FI
    F = 0
    # Same as above for system measurement
    F_syst = 0

    # F_sqrd stores the cumulative sum of each contribution squared for calculating variance
    F_sqrd = 0
    F_syst_sqrd = 0

    # M_angle stores the measurement angle of each measurement
    M_angle = np.zeros(n)

    # Generates multiple samples
    for i in np.arange(N):
        print('Generating sample {}'.format(i+1))
        # Resets to initial state
        rho = rho_0

        # Stores measurements for calculating FI
        M_store = [None]*n

        for j in np.arange(n):
            # Defines probability that outcome 0 occurs
            p0 = (K[0] * rho * K[0].dag()).tr()
            # Choice of {0, 1} with prob {p(0), p(1)}
            x[(i, j)] = np.random.choice([0, 1], p=[p0, 1 - p0])

            # Stores which measurement occurred for FI calculation and measurement angles plot
            M_store[j] = M[x[(i, j)]]
            # print('Measurement choice {}'.format(j + 1))
            # print(M_store[j])

            # Updates the state by applying the measurement projection and normalising
            rho = K[x[(i, j)]] * rho * K[x[(i, j)]].dag()
            rho = rho / rho.tr()

            # Extracts measurement angles of chain 0
            if i == 0:
                M_j = M_store[j].full()
                # Rotates measurement so M_j[0] is always positive
                M_j = M_j[0]/abs(M_j[0]) * M_j
                M_angle[j] = np.angle(M_j[1])

        # Uses the fischer_contr() function to add FI of this sample to the cumulative FI
        F_cont = fischer_contr(rho_0, n, lmbd, phi, M_store, [False, M_store[-1]])
        # Culmulative FI
        F += F_cont
        # Sqaure of contributions for calculating variance
        F_sqrd += F_cont**2

    # FI is the average of the culmulative FI
    F_ang[k] = (1/N) * F


# Saving data
save = True
print('Saving: {}'.format(save))
# Toggles whether to save
if save:
    # Opens the file in append mode
    with open('../../data/Model1/fig4_2.csv', 'a', newline='') as file:
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(['F'] + F_ang.tolist())
        z.writerow(['Angle'] + ang.tolist())
