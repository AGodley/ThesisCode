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
# adaptive v14: Added more saving options, found a new funciton for MLE

from math import pi, sqrt, sin  # For general maths
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
def measurement_choice(pi_mc, U_m, U_dot_m, A_1_m, j_m, strat_m, opt_m):
    # Regular +/- measurements
    if strat_m == 'reg':
        meas = [(1 / sqrt(2)) * (basis(2, 0) + np.exp(1j*opt_m)*basis(2, 1)), (1 / sqrt(2)) * (basis(2, 0) - np.exp(1j*opt_m)*basis(2, 1))]

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
def system_Measurement(pi_n, meas_syst_s_m, strat_s_m, opt_s_m):
    if strat_s_m == 'reg' and meas_syst_s_m:
        meas = [(fock(2, 0) + np.exp(1j*opt_s_m)*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j*opt_s_m)*fock(2, 1)).unit()]
    else:
        # Can use the same method as above
        rx = np.imag((pi_n * sigmax()).tr())
        ry = np.imag((pi_n * sigmay()).tr())
        C = (1 / 2) * (qeye(2) + (rx * sigmay() - ry * sigmax()) / sqrt(rx ** 2 + ry ** 2))
        ev, meas = C.eigenstates()
    return meas


# Creates a function for ML estimation
# Essentially the same process as sample generation, but theta may now be non-zero
# args = (theta, lmbd, phi, x[i]-a sample, rho_0)
# measure_system_mle = [measure_system, system_outcome]
def mle(tau, lmbd_val, phi_val, xdata, rho_init, strat_mle, measure_system_mle, opt_mle):
    # Initiates MLE probability and state
    log_p_mle = 0
    rho_mle = rho_init

    # Initiates the first measurement
    meas_mle = [(fock(2, 0)+fock(2, 1)).unit(), (fock(2, 0)-fock(2, 1)).unit()]
    # Initiates corresponding Kraus operators and unitaries for calculating A_1
    K_mle, U_mle = kraus(tau, lmbd_val, phi_val, meas_mle)
    K_mle_zero, U_mle_zero = kraus(0, lmbd_val, phi_val, meas_mle)
    K_dot_mle, U_dot_mle = kraus_dot(lmbd, meas_mle)
    # Initial A_1, needed for calculating adaptive measurements
    A_1_mle = U_dot_mle * tensor(rho_init, fock_dm(2, 0)) * U_mle_zero.dag() \
              - U_mle_zero * tensor(rho_init, fock_dm(2, 0)) * U_dot_mle.dag()

    # Initialises pi_m for measurement_choice()
    pi_mle = rho_mle

    for i_mle in np.arange(len(xdata)):
        # Finds adaptive choice of next measurement
        meas_mle, pi_mle = measurement_choice(pi_mle, U_mle, U_dot_mle, A_1_mle, i_mle, strat_mle, opt_mle)
        # Calculates the corresponding Kraus operators
        K_mle, U_mle = kraus(tau, lmbd_val, phi_val, [meas_mle[xdata[i_mle]]])

        # Updates the state based on which Kraus occurred in the sample
        rho_mle = K_mle * rho_mle * K_mle.dag()

        # Updates log prob. with log prob. of this Kraus occurring
        log_p_mle += np.log(rho_mle.tr())

        # Normalises the state
        rho_mle = rho_mle / rho_mle.tr()

        # Updates pi_m given measurement outcome
        if strat_mle == 'adapt':
            pi_mle = (tensor(qeye(2), meas_mle[xdata[i_mle]]*meas_mle[xdata[i_mle]].dag()) * pi_mle).ptrace(0)

    # Measures the system
    if measure_system_mle[0]:
        M_mle = system_Measurement(pi_mle, measure_system_mle, strat_mle, opt_mle)
        # Defines probability that outcome 0 occurs
        p0_mle = (rho_mle * M_mle[0] * M_mle[0].dag()).tr()
        p = [p0_mle, 1 - p0_mle]
        log_p_mle += np.log(p[measure_system_mle[1]])

    # Returns minus value as there is only a minimise argmax function
    return -log_p_mle


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
N = 1        # No. of samples to generate
n = 20         # No. of ancillary systems
theta = 0       # True value of theta for simulation
lmbd = 0.8      # Transition parameter, 0<=lmbd<=1
phi = pi/4     # Phase parameter

# Measurement strategy
strat = 'adapt'     # should be either reg or adapt

# Measuring the system
# With new simultaneous adapt/adapt w measure leave True
# Set to false for regular measurement
measure_system = False

# Initial state
rho_0 = fock_dm(2, 0)
# Optimal angle
# (33/50)pi for case 1, (14/49)pi for case 2
# Needs setting in measurement choice function too
opt = (33/50)*pi
# Initial measurement choice
M = [(fock(2, 0) + np.exp(1j*opt)*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j*opt)*fock(2, 1)).unit()]

# Kraus and unitary operators for calculating A_1
K, U = kraus(theta, lmbd, phi, M)
# Derivative of Kraus and unitary operators at theta=0 for calculating A_1
K_dot, U_dot = kraus_dot(lmbd, M)    # Used for calculating FI at theta=0
# Initial A, needed for calculating future As
A_1 = U_dot * tensor(rho_0, fock_dm(2, 0)) * U.dag() \
      - U * tensor(rho_0, fock_dm(2, 0)) * U_dot.dag()

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
# System Measurement Outcomes
syst_meas_outcome = np.zeros(N, dtype=int)

# Generates multiple samples
for i in np.arange(N):
    print('Generating sample {}'.format(i+1))
    # Resets to initial state
    rho = rho_0

    # Stores measurements for calculating FI
    M_store = [None]*n

    # Initial pi_m
    pi_m = rho_0

    for j in np.arange(n):
        # print('Calculating the {} measurement for sample {}'.format(j + 1, i+1))
        # Finds adaptive choice of next measurement
        M, pi_m = measurement_choice(pi_m, U, U_dot, A_1, j, strat, opt)
        K, U = kraus(theta, lmbd, phi, M)

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

        # Updates pi_m given measurement outcome
        if strat == 'adapt':
            pi_m = (tensor(qeye(2), M[x[(i, j)]] * M[x[(i, j)]].dag()) * pi_m).ptrace(0)

        # Extracts measurement angles of chain 0
        if i == 0:
            M_j = M_store[j].full()
            # Rotates measurement so M_j[0] is always positive
            M_j = M_j[0]/abs(M_j[0]) * M_j
            M_angle[j] = np.angle(M_j[1])

    # Measures the system
    if measure_system:
        M = system_Measurement(pi_m, measure_system, strat, opt)
        # Defines probability that outcome 0 occurs
        p0 = (rho * M[0]*M[0].dag()).tr()
        # Chooses a measurement with prob. defined above and appends onto x and M_store
        syst_meas_outcome[i] = np.random.choice([0, 1], p=[p0, 1 - p0])
        M_store.append(M[syst_meas_outcome[i]])

    # Uses the fischer_contr() function to add FI of this sample to the cumulative FI
    F_cont = fischer_contr(rho_0, n, lmbd, phi, M_store, [False, M_store[-1]])
    # Culmulative FI
    F += F_cont
    # Sqaure of contributions for calculating variance
    F_sqrd += F_cont**2

    # Same again for system measurement
    if measure_system:
        F_syst_cont = fischer_contr(rho_0, n, lmbd, phi, M_store, [measure_system, M_store[-1]])
        F_syst += F_syst_cont
        F_syst_sqrd += F_syst_cont ** 2
print('Samples successfully generated')
print('1st:')
print(x[0])
# The average FI of the samples
F_chains = (1 / N) * F
# The variance of FI of samples
F_var = (1/N)*F_sqrd - F_chains**2
# The error associated with this variance
error = sqrt(F_var/N)
print('FI successfully calculated')
print('Sample FI is {}'.format(F_chains))
print('Error is {}'.format(error))

if measure_system:
    # The average FI of the samples
    F_syst_chains = (1 / N) * F_syst
    # The variance of FI of samples
    F_syst_var = (1 / N) * F_syst_sqrd - F_syst_chains ** 2
    # The error associated with this variance
    error_syst = sqrt(F_syst_var / N)
    print('Sample FI with system measurement is {}'.format(F_syst_chains))
    print('Error with system measurement is {}'.format(error_syst))

# calculate exact QFI
QFI_exact = (1-(sqrt(1-lmbd))**n)**2/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd)**(n-1)\
            + (n-1)*lmbd/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd - (1-lmbd)**n)/(1-sqrt(1-lmbd))**2 \
            - 2*lmbd*(sqrt(1-lmbd)-(sqrt(1-lmbd))**n)/((1-sqrt(1-lmbd))**3) \
            + (1- (1-lmbd)**(n-1))/lmbd \
            -(1-lmbd-(1-lmbd)**n)/lmbd - 1 + n*(1-lmbd)**(n-1)\
            +n*(1- (1-lmbd)**(n-1))
print('Exact QFI is {}'.format(4*QFI_exact))

# calculate exact asymptotic QFI
QFI_asymptotic =n*8*(1+ sqrt(1-lmbd)/(1-sqrt(1-lmbd)))
print('Asymptotic QFI is {}'.format(QFI_asymptotic))

# Checks values from MLE
tht_vals = np.linspace(-0.9, 0.9, 200)
ests = np.zeros(len(tht_vals))
for i in np.arange(len(tht_vals)):
    tht = tht_vals[i]
    ests[i] = mle(tht, lmbd, phi, x[0], rho_0, strat, [measure_system, syst_meas_outcome[0]], opt)
print(ests)

# Plots results
fig_mle, ax_mle = plt.subplots()
ax_mle.plot(tht_vals, ests)
ax_mle.set_ylabel('MLE Value')
ax_mle.set_xlabel('theta')
plt.show()
