from math import pi, sqrt  # For general maths
import numpy as np      # For arrays and matrices
from qutip import *     # For quantum objects
from code.Model2.AbsorberFunctions import unitaryV


# Creates a function for calculating Kraus operators
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
    # Total unitary
    VU = tensor(qeye(2), V)*tensor(unitry, qeye(2))
    # Checks whether the function was sent 2 measurements or not
    return VU


a, sPsi = unitaryV(0.2, 0.8, pi/4, False)
sPsi = tensor(sPsi, fock_dm(2, 0))
print('sPsi: {}'.format(sPsi))
VU = kraus(0.2, 0.2, 0.8, pi/4, [fock_dm(2, 0)])
print('VU: {}'.format(VU))
print('VUsPsiUVdag: {}'.format(VU*sPsi*VU.dag()))


