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
# adaptive v14: Added more saving options, found a new function for MLE
# adaptive v15: Changing it so argmax is done after each sample
# adaptive v16: Swapping to basin hopping MLE method
# adaptive v17: Brute force on outliers
# Integrated git, so I don't need these comments anymore

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
        print(K[0].dag()*K[0]+K[1].dag()*K[1])
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
        # print(B)

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
def mle(tau, lmbd_val, phi_val, measurements, rho_init, strat_mle, measure_system_mle, opt_mle):
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

    # Measures the system
    if measure_system_mle[0]:
        M_mle = measure_system_mle[1]
        # Defines probability that outcome 0 occurs
        p0_mle = (rho_mle * M_mle * M_mle.dag()).tr()
        log_p_mle += np.log(p0_mle)

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


# Setup #
# Global parameter values
N = 10        # No. of samples to generate
n = 2000         # No. of ancillary systems
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
opt = 0
# Initial measurement choice
M = [(fock(2, 0) + np.exp(1j*opt)*fock(2, 1)).unit(), (fock(2, 0) - np.exp(1j*opt)*fock(2, 1)).unit()]

# Kraus and unitary operators for calculating A_1
K, U = kraus(theta, lmbd, phi, M)
# Derivative of Kraus and unitary operators at theta=0 for calculating A_1
K_dot, U_dot = kraus_dot(lmbd, M)    # Used for calculating FI at theta=0
# Initial A, needed for calculating future As
A_1 = U_dot * tensor(rho_0, fock_dm(2, 0)) * U.dag() \
      - U * tensor(rho_0, fock_dm(2, 0)) * U_dot.dag()

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
syst_measurements = [None]*N

# Estimates of theta from the argmax
theta_est = np.ones(N)     # Vector of estimates
bnds = [(-0.5, 0.5)]          # Interval the argmax investigates
# Setup #

# Sample generation and argmax #
for i in np.arange(N):
    print('Generating sample {}'.format(i+1))
    # Resets to initial state
    rho = rho_0

    # Stores measurements for calculating FI and MLE
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
        x = np.random.choice([0, 1], p=[p0, 1 - p0])

        # Stores which measurement occurred for FI calculation and measurement angles plot
        M_store[j] = M[x]
        # print('Measurement choice {}'.format(j + 1))
        # print(M_store[j])

        # Updates the state by applying the measurement projection and normalising
        rho = K[x] * rho * K[x].dag()
        rho = rho / rho.tr()

        # Updates pi_m given measurement outcome
        if strat == 'adapt':
            pi_m = (tensor(qeye(2), M[x] * M[x].dag()) * pi_m).ptrace(0)

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
        # Chooses a measurement with prob. defined above and stores it in syst_measurements
        syst_meas_outcome = np.random.choice([0, 1], p=[p0, 1 - p0])
        syst_measurements[i] = M[syst_meas_outcome]

    # Uses the fischer_contr() function to add FI of this sample to the cumulative FI
    F_cont = fischer_contr(rho_0, n, lmbd, phi, M_store, [False, syst_measurements[i]])
    F += F_cont    # Culmulative FI
    # Square of contributions for calculating variance
    F_sqrd += F_cont**2

    # Same again for system measurement
    if measure_system:
        F_syst_cont = fischer_contr(rho_0, n, lmbd, phi, M_store, [measure_system, M_store[-1]])
        F_syst += F_syst_cont
        F_syst_sqrd += F_syst_cont ** 2

    # Calculates the MLE for the sample
    # niter controls the number of attempts at estimation it tries
    # niter_success stops the process if it gets the same result the specified no. of times in a row
    # T - temperature parameter; should be comparable to step between minima
    print('Argmax for sample {}'.format(i+1))
    # MLE
    bnds = ([(-0.5, 0.5)])          # Interval the argmax investigates
    result = minimize(mle, x0=np.array(0), bounds=bnds, method='L-BFGS-B',
                      args=(lmbd, phi, M_store, rho_0, strat, [measure_system, syst_measurements[i]], opt))
    theta_est[i] = result.x
    # brute force on outliers
    if abs(result.x) > 0.1:
        result, fval, grid, jout = brute(mle, ranges=bnds, Ns=200, full_output=True,
                                         args=(lmbd, phi, M_store, rho_0, strat, [measure_system, syst_measurements[i]], opt))
        theta_est[i] = result
        print('Brute')
# End of sample generation
print('Samples successfully generated')
# Sample generation and argmax #

# Approximate FIs #
# Calculates the FI as the average FI of the samples
F_chains = (1 / N) * F
# The variance of FI of samples
F_var = (1/N)*F_sqrd - F_chains**2
# The error associated with this variance
error = sqrt(F_var/N)
print('FI successfully calculated')
print('Sample FI is {}'.format(F_chains))
print('Error is {}'.format(error))

# Calculates the FI including system measurement
if measure_system:
    # The average FI of the samples
    F_syst_chains = (1 / N) * F_syst
    # The variance of FI of samples
    F_syst_var = (1 / N) * F_syst_sqrd - F_syst_chains ** 2
    # The error associated with this variance
    error_syst = sqrt(F_syst_var / N)
    print('Sample FI with system measurement is {}'.format(F_syst_chains))
    print('Error with system measurement is {}'.format(error_syst))

# Calculates FI from MLE estimates
print('Argmax successfully completed')
# FI inversely proportional to the sample variance
# Using true parameter value to calculate sample variance
sampleVar = (1/N) * np.sum((theta_est-theta)**2)
sampleFI = 1/sampleVar
print("The sample variance is {} and the empirical FI is {}".format(sampleVar, sampleFI))
# Approximate FIs #

# Exact QFIs #
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
# Exact QFIs #

# Figure plots #
# Measurement angle plot
fig0 = plt.figure()
ax0 = plt.subplot(1, 1, 1, projection='polar')
# Corrects the angles to 0<theta<pi range
M_angle_corrected = M_angle
for i in np.arange(n):
    if M_angle_corrected[i] < 0:
        M_angle_corrected[i] = M_angle_corrected[i] + pi
    # Factor of two makes polar plot cover the whole circle
    M_angle_corrected[i] = 2 * M_angle_corrected[i]
# Sets up radii for the polar plot
r = np.linspace(1, 10, n)
# Polar plot
ax0.plot(M_angle_corrected, r, 'o', linewidth=0.1)
# Arcs between points
for i in np.arange(n-1):
    points = 100
    if abs(M_angle_corrected[i+1] - M_angle_corrected[i]) < pi:
        arc = np.linspace(M_angle_corrected[i], M_angle_corrected[i+1], points)
    else:
        if M_angle_corrected[i] < pi:
            arc0 = np.linspace(M_angle_corrected[i], 0, int(np.ceil(0.5*points)))
            arc1 = np.linspace(2*pi, M_angle_corrected[i+1], int(np.floor(0.5*points)))
            arc = np.append(arc0, arc1)
        else:
            arc0 = np.linspace(M_angle_corrected[i], 2*pi, int(np.ceil(0.5*points)))
            arc1 = np.linspace(0, M_angle_corrected[i+1], int(np.floor(0.5*points)))
            arc = np.append(arc0, arc1)
    ax0.plot(np.append(arc, M_angle_corrected[i+1]), np.append(r[i]*np.ones(points), r[i+1]), 'r--')
ax0.set_xticks(pi/180*np.arange(0, 360, 30))
ax0.set_xticklabels(0.5*np.arange(0, 360, 30))
plt.show()

# Plots histogram of MLE estimates
plt.figure(2)
plt.hist(theta_est)
plt.xlabel(r'$\hat{\theta}$')
plt.ylabel('Frequency')
# Changing initial state or phi isn't recorded - don't change them
plt.savefig('figures\simulation_adaptive_v15_N{}_n{}_theta{}_lmbd{}.png'.format(N, n, str(theta).replace('.', ''),
                                                                                str(lmbd).replace('.', '')))
plt.show()
# Figure plots #

# Saving Utility #
# Saves the data
save = False
print('Saving FI: {}'.format(save))
# Toggles whether to save
if save:
    # Opens the file in append mode
    with open('../../data/Model1/fig1_regular_non_opt.csv', 'a', newline='') as file:
        data = [F_chains, N, n, error, lmbd, phi, strat, measure_system]
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(data)

# Same again with system measurement  FI instead
save2 = False
print('Saving FI with system measurement: {}'.format(save2))
if save2:
    # Opens the file in append mode
    with open('../../data/Model1/fig1_adaptive_measure_system.csv', 'a', newline='') as file:
        data = [F_syst_chains, N, n, error_syst, lmbd, phi, strat, measure_system]
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(data)

# Saves the measurement angles
save3 = False
print('Saving measurement angles: {}'.format(save3))
if save3:
    with open('../../data/Model1/fig2_2_measurement_angles.csv', 'a', newline='') as file:
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(M_angle_corrected)

# Saving the argmax
save4 = False
print('Saving the MLE estimates: {}'.format(save4))
if save4:
    with open('../../data/Model1/fig3_regular_non_opt.csv', 'a', newline='') as file:
        z = csv.writer(file)
        # Appends data onto the file
        z.writerow(['I_MLE', 'I', 'N', 'n', 'lmbd', 'phi', 'Strategy', 'Measuring system'])
        z.writerow([sampleFI, F_chains, N, n, lmbd, phi, strat, measure_system])
        z.writerow(['Data']+theta_est.tolist())
# Saving Utility #
