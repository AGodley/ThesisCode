# Alfred Godley
# Main file for the adaptive measurement simulations; refer to the latex notes for more info.

from math import pi, sqrt  # For general maths
import numpy as np      # For arrays and matrices
from scipy.optimize import *     # For argmax/maximum likelihood estimation
import matplotlib.pyplot as plt    # For plotting graphs
from qutip import *     # Package for quantum objects
import csv   # Comma separated values, useful for saving data
# Additional functions required for the absorber grouped in a separate file
from AbsorberFunctions import ss, TO, Id_T_heisenberg, unitaryV
# Imports the function used for the rough initial estimation
from initial_estimation_2 import initial_est
# Imports the old Kraus function from before the introduction of the adsorber
# Used to calculated the initial estimation's contribution to the MLE
# Also used in the QFI calculation
from initial_estimation_2 import kraus as kraus_2by2
from initial_estimation_2 import kraus_dot as kraus_dot_2by2
import time     # For timing the trajectory generation loop


# Function for calculating Kraus operators
# Also returns the corresponding pseudo-unitary
# Kraus operators are used to update the state, unitary is used to calculate next measurement
# Inputs are respectively, the param of interest, our rough estimate, lambda, phi and a list containing the measurements
# adaptive_v3: added functionality to send 1 measurement for use when calculating classical FI
def kraus(theta_u, theta_ru, lmbd_u, phi_u, meas_u):
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
    VU = tensor(qeye(2), V)*(tensor(qeye(2), P23)*tensor(unitry, qeye(2))*tensor(qeye(2), P23))
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


# Function for the derivatives of the Kraus operators and unitary
# Needed for calculation of the classical FI and A_1
# adaptive_v3: added functionality to send 1 measurement and calc. the corresponding Kraus for calc. FI
def kraus_dot(theta_U, theta_RU, lmbd_U, phi_U, meas_U):
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


# Generates next choice of measurement
def measurement_choice(pi_mc, VU_m, VU_dot_m, A_1_m, j_m):
    # Adaptive measurements
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
        A = 2 * VU_m * A * VU_m.dag()
        A = A_1_m + A
        # A_norm = A / (-(A*A).tr()/2) # Alternative renormalisation

    # Traces out the system of interest and absorber producing B
    B = A.ptrace(2)

    # Checks whether B=0, if so returns +/- basis measurements
    # Otherwise, calculates components of B's bloch vec. and uses these to determine an orthogonal measurement
    if np.sum(abs(B.data)) < 1e-8:
        # +/- measurements
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
        C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
        ev, meas = C.eigenstates()
    # function returns two measurements
    # loop then chooses one of these and sends <e|pi_m|e> to this function to determine the next measurement
    pi_mc = A
    return meas, pi_mc


# Function for ML estimation
# Essentially the same process as sample generation, but theta may now be non-zero
# measurements = [measure_system, system_outcome]
def mle(tau, tht_rmle, lmbd_val, phi_val, measurements, rho_init, x_init):
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

    # 2nd stage estimation
    # Initial state
    rho_mle = rho_init
    for i_mle in np.arange(len(measurements)):
        # Finds adaptive choice of next measurement
        meas_mle = measurements[i_mle]
        # Calculates the corresponding Kraus operators
        K_mle, VU_mle, *_ = kraus(tau, tht_rmle, lmbd_val, phi_val, [meas_mle])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # Updates log prob. with log prob. of this Kraus occurring
        log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

    # Returns minus value as the argmax finds minima
    return -log_p_mle


# Calculates the Fischer information of a sample based on the measurements that occurred
def fischer_contr(rho_0_fi, n_fi, tht_fi, tht_rfi, lmbd_fi, phi_fi, meas_fi):
    # Initialises a variable for the FI
    f = 0
    # Each loop finds contribution from derivative of the jth Kraus
    for j in np.arange(n_fi):
        # Needed in K_total_dot
        K_j, VU_fi, *_ = kraus(tht_fi, tht_rfi, lmbd_fi, phi_fi, [meas_fi[j]])
        # Initially calculates K_j_dot and finds associated renormalisation constant 1/Tr(K_j*rho)
        K_total_dot, VU_fi, *_ = kraus_dot(tht_fi, tht_rfi, lmbd_fi, phi_fi, [meas_fi[j]])
        K_total_dot = K_total_dot / (K_j * rho_0_fi).tr()
        # Then multiplies by the product K_n...K_(j-1) with their renormalisation constants
        for k in np.arange(j + 1, n_fi):
            K_k, VU_fi, *_ = kraus(tht_fi, tht_rfi, lmbd_fi, phi_fi, [meas_fi[k]])
            K_total_dot = K_k * K_total_dot / (K_k * rho_0_fi).tr()

        # Adds the total contribution from derivative on jth Kraus operator to f
        f += (2 * np.real((K_total_dot * rho_0_fi).tr()))
    f = f**2
    return f


## Setup ##
# Global parameter values
N = 100        # No. of samples to generate
# Proportion of trajectory used in initial estimate
eps = 0     # Set to 1 for just rough estimation
rough = False   # Toggles rough estimation for calculating QFI, also change eps to 0
n = 300      # No. of ancillary systems
# Splits n as specified by eps
n_init = int(np.floor(n*eps))
n_adapt = int(np.ceil(n*(1-eps)))
theta = 0.2       # True value of theta for trajectory generation
lmbd = 0.8      # Transition parameter, 0<=lmbd<=1
phi = pi/4     # Phase parameter
opt = 0        # Optional optimal regular measurement angle

# Variable that stores cumulative CFI from sampling method
# (1/N)*F gives the average FI
F = 0

# F_sqrd stores the cumulative sum of each contribution squared for calculating variance
F_sqrd = 0

# M_angle stores the measurement angle of each measurement
M_angle = np.zeros(n_adapt)

# Estimates of theta from the argmax
theta_est = np.ones(N)     # Vector of estimates
## Setup ##


## Sample generation and argmax ##
# N total samples
for i in np.arange(N):
    # Starts loop timer
    t0 = time.time()

    # Rough estimate of theta - moved here since it changes for each trajectory
    if rough:
        theta_rough, x_rough = initial_est(theta, lmbd, phi, n_init)  # Rough initial estimate
    else:
        # Used for calculating FI
        theta_rough = 0.2
        x_rough = []
    print('Rough estimate of theta: {}'.format(theta_rough))

    # Initial state
    a, rho_0 = unitaryV(theta_rough, lmbd, phi, False)

    # Initial measurement choice
    M = [(fock(2, 0) + np.exp(1j * opt)*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j * opt)*fock(2, 1)).unit()]

    # Kraus and unitary operators for calculating A_1
    K, VU, *_ = kraus(theta, theta_rough, lmbd, phi, M)
    # Derivative of Kraus and unitary operators for calculating A_1
    K_dot, VU_dot, *_ = kraus_dot(theta, theta_rough, lmbd, phi, M)  # Used for calculating FI at theta=0
    # Initial A, needed for calculating future As
    A_1 = VU_dot * tensor(rho_0, fock_dm(2, 0)) * VU.dag() \
          - VU * tensor(rho_0, fock_dm(2, 0)) * VU_dot.dag()

    print('Generating sample {}'.format(i+1))

    # Resets to initial state
    rho = rho_0
    # Print statement for checking ss
    # print(K[0] * rho * K[0].dag() + K[1] * rho * K[1].dag() - rho)

    # Stores measurements for calculating FI and MLE
    M_store = [None]*n_adapt

    # Initial pi_m
    pi_m = rho_0

    for j in np.arange(n_adapt):
        # Finds adaptive choice of next measurement
        M, pi_m = measurement_choice(pi_m, VU, VU_dot, A_1, j)
        # Finds the corresponding Kraus operators
        K, VU, *_ = kraus(theta, theta_rough, lmbd, phi, M)

        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()

        # Choice of {0, 1} with prob {p(0), p(1)}
        x = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Stores which measurement occurred for FI calculation and measurement angles plot
        M_store[j] = M[x]

        # Updates the state by applying the measurement projection and normalising
        rho = K[x] * rho * K[x].dag()
        rho = rho / rho.tr()

        # Updates pi_m given measurement outcome
        pi_m = (tensor(tensor(qeye(2), qeye(2)), M[x] * M[x].dag()) * pi_m).ptrace([0, 1])

        # Extracts measurement angles of chain 0
        if i == 0:
            M_j = M_store[j].full()
            # Rotates measurement so M_j[0] is always positive
            M_j = M_j[0]/abs(M_j[0]) * M_j
            M_angle[j] = np.angle(M_j[1])

    # Calculates the FI of this sample using the fischer_contr() function
    F_cont = fischer_contr(rho_0, n_adapt, theta, theta_rough, lmbd, phi, M_store)
    F += F_cont    # Adds this to the culmulative FI
    # Square of contributions for calculating variance
    F_sqrd += F_cont**2

    # Calculates the MLE for the sample
    print('Argmax for sample {}'.format(i+1))
    # MLE
    bnds = ([(-0.4, 0.4)])          # Interval the argmax investigates
    result = minimize(mle, x0=np.array(0), bounds=bnds, method='L-BFGS-B',
                      args=(theta_rough, lmbd, phi, M_store, rho_0, x_rough))
    theta_est[i] = result.x
    print('Estimate of theta: {}'.format(result.x))
    # Brute force on outliers
    # Uses the true value to activate
    # This is only needed as minimize doesn't always pick the global minima
    # Could be fixed with a better minimize function?
    if abs(result.x-theta) > 0.1:
        print('Using brute force')
        result, fval, grid, jout = brute(mle, ranges=bnds, Ns=200, full_output=True,
                                         args=(theta_rough, lmbd, phi, M_store, rho_0, x_rough))
        theta_est[i] = result
        print('New estimate of theta: {}'.format(result))

    # Ends loop timer
    t1 = time.time()
    total_time = t1 - t0    # In seconds
    print('Sample {} took {} minutes and {} seconds'.format(i+1, np.floor(total_time/60), total_time%60))
# End of sample generation
print('Samples successfully generated')
## Sample generation and argmax ##

## QFI ##
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
A_tilde = A - a*qeye(2)
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
               2*((Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()) -
                   Qobj(K[i]*rho_ss*(K_dot[i] + 1j*a*K[i]).dag()).dag())/(2*1j) * RA).tr() )
print('QFI per step: {}'.format(Q))
## QFI ##

## Approximate FIs ##
# Calculates the FI as the average FI of the samples
F_chains = (1 / N) * F
# The variance of FI of samples
F_var = (1/N)*F_sqrd - F_chains**2
# The error associated with this variance
error = sqrt(F_var/N)
print('FI successfully calculated')
print('Sample FI is {}'.format(F_chains))
print('Error is {}'.format(error))

# Calculates FI from MLE estimates
print('Argmax successfully completed')
# FI inversely proportional to the sample variance
# Using true parameter value to calculate sample variance
sampleVar = (1/N) * np.sum((theta_est-theta)**2)
sampleFI = 1/sampleVar
print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))
## Approximate FIs ##

# Theta estimates
print('Theta estimates: {}'.format(theta_est))

## Saving Utility ##
# Saves the data
save = True
print('Saving FI: {}'.format(save))
# Toggles whether to save
if save:
    # Opens the file in append mode
    with open('../../data/Model2/full_adaptive_no_initial_est.csv', 'a', newline='') as file:
        data = [F_chains, sampleFI, N, n, eps, error, lmbd, phi]
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(data)
## Saving utility ##