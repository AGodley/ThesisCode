from qutip import *
from numpy import sin, cos,  sqrt
import numpy as np
import matplotlib.pyplot as plt


def ss(tht, lmb, phi):
    # Commonly occurring square root term
    sqrts = sqrt(1-tht**2) * sqrt(1-lmb)
    #
    A = (1 - tht**2*lmb + sqrts*( -1 - tht*sqrt(lmb)*cos(phi)*(1-cos(2*tht)) + cos(2*tht)*(sqrts-1) ) ) / \
        (sqrts + tht*sqrt(lmb)*cos(phi) - 1)
    print('A is {}'.format(A))
    #
    B = sin(2*tht)**2 * sqrts / (2*A)
    print('B is {}'.format(B))
    # Finding rz
    # Num out by about 0.5
    num = -(B + (cos(tht))**2) * (lmb - tht**2)
    # Also out
    denom = (B*(2-tht**2-lmb) + cos(2*tht) - cos(tht)**2*(tht**2+lmb) - 1)
    print('num is {}'.format(num))
    print('denom is {}'.format(denom))
    rz = num / denom
    # Finding ry
    ry = -sin(2*tht)*((2-tht**2-lmb)*rz + (lmb-tht**2)) / (2*A)
    # Finding rx
    rx = tht*sqrt(lmb)*sin(phi)*ry / (sqrts + tht*sqrt(lmb)*cos(phi) - 1)
    # Print statements
    print('rx is {}'.format(rx))
    print('ry is {}'.format(ry))
    print('rz is {}'.format(rz))
    # Grouping them together
    r = np.matrix([[rx],
                   [ry],
                   [rz],
                   [1]])
    return r



def T(tht, lmb, phi):
    tht_s = tht ** 2
    # Sqrt terms
    rt_lmb = np.sqrt(lmb)
    rt_tht_s = np.sqrt(1 - tht_s ** 2)
    rt_lmb_m = np.sqrt(1 - lmb)
    # Common combined sqrt terms
    a = rt_tht_s * rt_lmb_m
    b = tht * rt_lmb
    # Trig of phi
    cs_p = np.cos(phi)
    sn_p = np.sin(phi)
    # Trig of theta
    cs = np.cos(tht)
    cs2 = np.cos(2 * tht)
    sn = np.sin(tht)
    sn2 = np.sin(2 * tht)
    # Writing out T
    T = np.matrix([[a + b*cs_p, -b*sn_p, 0, 0],
                   [-b*sn_p, -b*cs_p + a*cs2, sn*cs*(2-tht_s-lmb), sn*cs*(lmb-tht_s)],
                   [0, -sn2*a, cs2 - cs**2*(tht_s + lmb), cs**2*(lmb - tht_s)],
                   [0, 0, 0, 1]])
    print('T:')
    print(T)
    return T

# Values
tht = 0.2
lmb = 0.8
phi = np.pi/4
# Vector r that corresponds to stationary state
r = ss(tht, lmb, phi)
# Extracting components
rx = r[0, 0]
ry = r[1, 0]
rz = r[2, 0]
# Norm of vector r
mod_r = np.sqrt(rx**2 + ry**2 + rz**2)
print('Modulus of r: {}'.format(mod_r))

# Transition matrix that acts on vector form r
# For checking s.s.
t = T(tht, lmb, phi)
print('T(r):')
print(t * r)
print('T^2(r):')
print(t**2 * r)
print('T^50(r):')
print(t**50 * r)
print('T^100(r):')
print(t**100 * r)
print('T^1000(r):')
print(t**1000 * r)

# Eigenstates
# rx=r[0], ry=r[1], rz=r[2], rI=r[3]
f_plus = Qobj([[rz + mod_r],
               [rx + 1j*ry]]).unit()
mod_f_plus = Qobj([[rz + mod_r],
                   [rx + 1j*ry]]).norm()
f_minus = Qobj([[rz - mod_r],
                [rx + 1j*ry]]).unit()
mod_f_minus = Qobj([[rz - mod_r],
                   [rx + 1j*ry]]).norm()
# Eigenvalues
lmb_plus = (1/2) * (1 + mod_r)
lmb_minus = (1/2) * (1 - mod_r)

# |Psi_0/1> states
a_plus = rz + mod_r
a_minus = rz - mod_r
b = rx + 1j*ry
A = (1/mod_f_plus**2) * np.sqrt(lmb_plus)
B = (1/mod_f_minus**2) * np.sqrt(lmb_minus)
# State 0
psi_0 = ((A*a_plus**2 + B*a_plus**2)*tensor(basis(2, 0), basis(2, 0)) +
        (A*a_plus*b + B*a_minus*b)*tensor(basis(2, 1), basis(2, 0))).unit()
# State 1
psi_1 = ((A*a_plus*b + B*a_minus*b)*tensor(basis(2, 0), basis(2, 0)) +
        b**2*(A + B)*tensor(basis(2, 1), basis(2, 0))).unit()
print('Psi Normalised check: {}'.format(psi_0.dag()*psi_0))
print('Psi orthogonal check : {}'.format(psi_0.dag()*psi_1))
# Finds orthogonal psi_1
psi_1o = (psi_1 - (psi_0.dag()*psi_1).tr()/(psi_0.dag()*psi_0).tr() * psi_0).unit()
print('Orthogonal psi_1 check: {}'.format(psi_1o.dag()*psi_0))
# Gets other 2 psi vectors for defining V
# Projection operator
proj_psi = psi_0*psi_0.dag() + psi_1o*psi_1o.dag()
# Orthogonal projection
orth_proj_psi = qeye([2, 2]) - proj_psi
# Eigenvalues and corresponding eigenstates
ev, psis = orth_proj_psi.eigenstates()
print('Eigenvalues: {}'.format(ev))
# Select states corresponding to an ev of 1
psi_2 = psis[2]
psi_3 = psis[3]

# |phi_0/1> states
# State 1
phi_0 = (( cos(tht)*sqrt(1-tht**2)*(A*a_plus**2 + B*a_minus**2) + 1j*b*sin(tht)*sqrt(1-lmb)*(A*a_plus + B*a_minus) )
        * tensor(basis(2, 0), basis(2, 0)) +
        b*sqrt(lmb)*np.exp(1j*phi)*(A*a_plus + B*a_minus) * tensor(basis(2, 0), basis(2, 1)) +
        ( b*cos(tht)*sqrt(1-tht**2)*(A*a_plus + B*a_minus) + 1j*b**2*sin(tht)*sqrt(1-lmb)*(A + B) )
        * tensor(basis(2, 1), basis(2, 0)) +
        b**2*sqrt(lmb)*np.exp(1j*phi)*(A + B) * tensor(basis(2, 1), basis(2, 1))).unit()
# State 2
phi_1 = (( cos(tht)*sqrt(1-lmb)*b*(A*a_plus + B*a_minus) + 1j*sin(tht)*sqrt(1-tht**2)*(A*a_plus**2 + B*a_minus**2) )
        * tensor(basis(2, 0), basis(2, 0)) +
        tht*(A*a_plus**2 + B*a_minus**2) * tensor(basis(2, 0), basis(2, 1)) +
        ( b**2*cos(tht)*sqrt(1-lmb)*(A+B) + 1j*b*sin(tht)*sqrt(1-tht**2)*(A*a_plus + B*a_minus) )
        * tensor(basis(2, 1), basis(2, 0)) +
        b*tht*(A*a_plus + B*a_minus) * tensor(basis(2, 1), basis(2, 1))).unit()
print('Phi Normalised check: {}'.format(phi_0.dag()*phi_0))
print('Phi orthogonal check : {}'.format(phi_0.dag()*phi_1))
# Finds orthogonal phi_1
phi_1o = (phi_1 - (phi_0.dag()*phi_1).tr()/(phi_0.dag()*phi_0).tr() * phi_0).unit()
print('Orthogonal phi_1 check: {}'.format(phi_1o.dag()*phi_0))
# Gets other 2 psi vectors for defining V
# Projection operator
proj_phi = phi_0*phi_0.dag() + phi_1o*phi_1o.dag()
# Orthogonal projection
orth_proj_phi = qeye([2, 2]) - proj_phi
# Eigenvalues and corresponding eigenstates
ev, phis = orth_proj_phi.eigenstates()
print('Eigenvalues: {}'.format(ev))
# Select states corresponding to an ev of 1
phi_2 = phis[2]
phi_3 = phis[3]

# Unitary V
V = psi_0*phi_0.dag() + psi_1o*phi_1o.dag() + psi_2*phi_2.dag() + psi_3*phi_3.dag()
print('Unitary V: {}'.format(V))