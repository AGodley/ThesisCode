######################
# Initial estimation #
######################
# Initial estimation protocol. Initialises a system in the |0> state, then evolves it by applying Kraus operators
# randomly. At each step the outcome corresponding to the chosen Kraus operator is recorded. To estimate the parameter,
# we calculate the stationary state on a grid of values of theta, then we calculate the expected number of counts for
# that stationary state with that theta, and finally select the theta with expected counts closest to the actual counts
# observed.

# Standard imports
import numpy as np
from numpy import pi
from qutip import *
import matplotlib.pyplot as plt

# Kraus operators
from kraus import k, k_dot

# Stationary state
from absorber import ss


def initial_est(tht, lmbd, phi, n):
    # Rudimentary initial estimation. We assume we converge to the stationary state, and attempt to force this by
    # applying K_0 a number of times. We then calculate the expected number of 1s as n.Tr(K_1rho_ssK_1*), generate
    # a trajectory of outcomes and compare the expected results to the actual results.

    # Initial state
    rho_0 = fock_dm(2, 0)

    # Measurement choice
    M = [(np.sqrt(9)*fock(2, 0) + fock(2, 1)).unit(), (fock(2, 0) - np.sqrt(9)*fock(2, 1)).unit()]
    # M = [fock(2, 0), fock(2, 1)]    # Counting measurements

    # Sample generation and argmax #
    # Resets to initial state
    rho = rho_0

    K, U = k(tht, lmbd, phi, M)

    x = np.zeros(n, dtype=int)
    for j in np.arange(n):
        # Defines probability that outcome 0 occurs
        p0 = (K[0] * rho * K[0].dag()).tr()
        # Choice of {0, 1} with prob {p(0), p(1)}
        x[j] = np.random.choice([0, 1], p=[p0, 1 - p0])
        # print(x[j])
        # Updates the state by applying the measurement projection and normalising
        rho = K[x[j]] * rho * K[x[j]].dag()
        rho = rho / rho.tr()

    # Calculates the expected no. of ones at a range of values of theta
    points = 2000
    thetas = np.linspace(0.18, 0.22, points)
    expected = np.zeros(points)
    for i in range(len(thetas)):
        # Calculates the Kraus operators at that theta
        K, U = k(thetas[i], lmbd, phi, M)
        # Calculates the stationary state at that theta
        r_ss = ss(thetas[i], lmbd, phi)
        rho = (1/2) * (qeye(2) + r_ss[0, 0]*sigmax() + r_ss[1, 0]*sigmay() + r_ss[2, 0]*sigmaz())
        # Calculates the corresponding expected no. of ones
        expected[i] = np.real_if_close((rho * K[1].dag() * K[1]).tr()) * n

    # Recorded number of 1s in the trajectory
    actual = np.sum(x)

    # Finds the value closest to the actual value
    theta_est = thetas[0]
    for i in range(len(expected) - 1):
        if abs(actual - expected[i+1]) < abs(actual - expected[i]):
            theta_est = thetas[i+1]

    # print(thetas)
    # print(exp, actual, x[-11:-1])
    # print(actual, theta_est)
    #
    # # Simple plot
    # fig, ax = plt.subplots()
    # ax.plot(thetas, expt)
    # plt.show()

    # End of sample generation
    return theta_est, x


if __name__ == '__main__':
    samples = 100000
    eps = 0.1
    used = int(samples**(1-eps))
    estimate, xs = initial_est(0.2, 0.8, pi/4, used)

    print(f'Estimating theta using {used} samples')
    print(f'We found theta={estimate}')