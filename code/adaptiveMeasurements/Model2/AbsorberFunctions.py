# Alfred Godley
# New functions required when we introduced the absorber system

from qutip import *
from numpy import sin, cos, sqrt
import numpy as np


# Function that finds the stationary state of the system of interest (SOI) using the analytical formulas we calculated.
# This stationary state is usually mixed, and we want to introduce a secondary unitary V acting on the SOI and an
# additional adsorber system that makes the SOI+adsorber state pure.
def ss(tht, lmb, phi, show):
    # Commonly occurring square root term
    sqrts = sqrt(1-tht**2) * sqrt(1-lmb)

    # Two constants defined in the calculation to simplify the expressions
    A = (1 - tht**2*lmb + sqrts*(-1 - tht*sqrt(lmb)*cos(phi)*(1-cos(2*tht)) + cos(2*tht)*(sqrts-1) ) ) \
        / (sqrts + tht*sqrt(lmb)*cos(phi) - 1)
    B = sin(2*tht)**2 * sqrts / (2*A)

    # Formula for rz component
    num = -(B + (cos(tht))**2) * (lmb - tht**2)
    denom = (B*(2-tht**2-lmb) + cos(2*tht) - cos(tht)**2*(tht**2+lmb) - 1)
    rz = num / denom

    # Formula for ry component
    ry = -sin(2*tht)*((2-tht**2-lmb)*rz + (lmb-tht**2)) / (2*A)

    # Formula for rx component
    rx = tht*sqrt(lmb)*sin(phi)*ry / (sqrts + tht*sqrt(lmb)*cos(phi) - 1)

    # Grouping them together. The 4th component is necessary to account for constant terms when multiplying by the
    # matrix form of the transition operator T
    r = np.matrix([[rx],
                   [ry],
                   [rz],
                   [1]])

    # Print statements for checking stationary state
    # Toggled by passing 'True' or 'False' to the function
    if show:
        print('r:')
        print(r)
        print('Mod r: {}'.format(sqrt(rx**2+ry**2+rz**2)))
    return r


# Finds the matrix form of the Schrodinger representation transition operator T using the analytical expressions we
# calculated. For the stationary state vector r above, we should have T*r=r, which can be checked in unitaryV. This
# is used in the unitaryV function.
def TO(tht, lmb, phi, show):
    # All these terms are defined to simplify the expressions in T below
    # theta squared term
    tht_s = tht ** 2

    # Square root terms
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

    # Print statement for T
    # Toggled by passing 'True' or 'False' to the function
    if show:
        print('T:')
        print(T)
    return T


# Finds the matrix form of Id-T_heisenberg using the analytical expressions we calculated. The Heisenberg representation
# of the transition operator isn't explicitly included as only Id-T is important. The Moore-Penrose inverse of this is
# used in the calculation of the QFI.
def Id_T_heisenberg(tht, lmb, phi, show):
    # Terms defined to simplify expressions in code
    # theta squared term
    tht_s = tht ** 2

    # Square root terms
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

    # Writing out ID_T
    Id_T = np.matrix([[1 - a - b*cs_p, b*sn_p, 0, 0],
                      [b*sn_p, 1 - cs2*a + b*cs_p, sn2*a, 0],
                      [0, -sn2*(1-(1/2)*(lmb+tht_s)), 1 - (cs2 - (1/2)*(tht_s + lmb)*(1 + cs2)), 0],
                      [0, -(1/2)*(lmb-tht_s)*sn2, -(1/2)*(lmb-tht_s)*(1+cs2), 0]])

    # Print statement for T
    # Toggled by passing 'True' or 'False' to the function
    if show:
        print('Id_T:')
        print(Id_T)
    return Id_T


# Calculates the unitary V which acts on the system of interest (SOI) and absorber system. V should reverse the
# evolution of the SOI so that the combined SOI+absorber has a pure stationary state. V is defined using two sets of
# states on the absorber+input systems. Part of these sets are calculated using analytical formulae. The other part
# represents a necessary, but arbitrary choice of states required to make V unitary.
#
# The first set, the phis or 'ph' states, come from purifying the ss of the SOI onto the SOI+adsorber system, tensoring
# with the input state |0><0| and then rewriting this expression so that it splits the overall state into a component
# with the ss in the state |0> and a component with the ss in the state |1>. The ph states are the parts tensored to
# this |0/1> respectively.
#
# Similarly, the psis or 'ps' states are the parts tensored to this |0/1> respectively after applying the evolution
# unitary U to the combined SOI+adsorber+inputs and then repeating this splitting process again.
#
# We then define V as the projection from these ph states to the ps states.
def unitaryV(tht, lmb, phi, show):
    # Uses the ss function above to find the stationary state vector for the SOI
    r = ss(tht, lmb, phi, show)

    # Extracts the rx, ry and rz components
    rx = r[0, 0]
    ry = r[1, 0]
    rz = r[2, 0]

    # Calculated the norm of r, which is |r|<1 for mixed states
    # This term is important
    mod_r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Uses the transition matrix function to find the transition matrix that acts on the vector form of r
    t = TO(tht, lmb, phi, show)

    # Uses print statements to check the ss is truly stationary
    if show:
        print('r:')
        print(r)
        print('T(r):')
        print(t * r)
        print('T^2(r):')
        print(t ** 2 * r)
        print('T^50(r):')
        print(t ** 50 * r)
        print('T^100(r):')
        print(t ** 100 * r)
        print('T^1000(r):')
        print(t ** 1000 * r)

    # Calculates the eigenstates of the ss for purification using analytical formulae
    # Also calculates these states norms, which is needed in the formulae for the states
    f_plus = Qobj([[rz + mod_r],
                   [rx + 1j*ry]]).unit()
    mod_f_plus = Qobj([[rz + mod_r],
                       [rx + 1j*ry]]).norm()
    f_minus = Qobj([[rz - mod_r],
                    [rx + 1j*ry]]).unit()
    mod_f_minus = Qobj([[rz - mod_r],
                        [rx + 1j*ry]]).norm()

    # Calculates the corresponding eigenvalues
    lmb_plus = (1 / 2) * (1 + mod_r)
    lmb_minus = (1 / 2) * (1 - mod_r)

    # Calculates the purified stationary state
    sPsi = (sqrt(lmb_plus)*tensor(f_plus, f_plus) + sqrt(lmb_minus)*tensor(f_minus, f_minus)) * \
           (sqrt(lmb_plus)*tensor(f_plus, f_plus) + sqrt(lmb_minus)*tensor(f_minus, f_minus)).dag()

    # Terms introduced to simplify expressions
    a_plus = rz + mod_r
    a_minus = rz - mod_r
    b = rx + 1j * ry
    A = (1 / mod_f_plus ** 2) * np.sqrt(lmb_plus)
    B = (1 / mod_f_minus ** 2) * np.sqrt(lmb_minus)

    # ps states calculated using analytical formulae
    # Psi state 0
    ps_0 = ((A*a_plus**2 + B*a_minus**2) * tensor(basis(2, 0), basis(2, 0)) +
            (A*a_plus*b + B*a_minus*b)   * tensor(basis(2, 1), basis(2, 0))).unit()
    # Psi state 1
    ps_1 = ((A*a_plus*b + B*a_minus*b) * tensor(basis(2, 0), basis(2, 0)) +
             b**2*(A + B) * tensor(basis(2, 1), basis(2, 0))).unit()

    # Uses Gram-Schmidt to find component of ps_1 that is orthogonal to ps_0
    ps_1o = (ps_1 - (ps_0.dag() * ps_1).tr() / (ps_0.dag() * ps_0).tr() * ps_0).unit()

    # Calculates the other two ps states by finding the eigenstates of Id-(|ps_0><ps_0|+|ps_1><ps_1|)
    # Projection operator for ps_0 and ps_1 states
    proj_psi = ps_0 * ps_0.dag() + ps_1o * ps_1o.dag()
    # Calculates the orthogonal projection
    orth_proj_psi = qeye([2, 2]) - proj_psi
    # Finds the eigenvalues and corresponding eigenstates
    ev, psis = orth_proj_psi.eigenstates()
    # The eigenvalues are ordered, so the two ev of 1 are at the end of the list. The same happens for the eigenstates.
    ps_2 = psis[2]
    ps_3 = psis[3]

    # ph states calculated using analytical formulae
    # Phi state 0
    ph_0 = ((cos(tht) * sqrt(1 - tht**2) * (A*a_plus**2 + B*a_minus**2) + 1j*b*sin(tht)*sqrt(1 - lmb)
            * (A*a_plus + B*a_minus)) * tensor(basis(2, 0), basis(2, 0)) +
            b*sqrt(lmb) * np.exp(1j*phi) * (A*a_plus + B*a_minus) * tensor(basis(2, 0), basis(2, 1)) +
            (b*cos(tht) * sqrt(1-tht**2) * (A*a_plus + B*a_minus) + 1j*b**2*sin(tht) * sqrt(1-lmb)*(A + B))
            * tensor(basis(2, 1), basis(2, 0)) +
            b**2*sqrt(lmb) * np.exp(1j*phi)*(A + B) * tensor(basis(2, 1), basis(2, 1))).unit()
    # Phi state 1
    ph_1 = ((cos(tht) * sqrt(1 - lmb) * b * (A * a_plus + B * a_minus) + 1j * sin(tht) * sqrt(1 - tht ** 2) * (
            A * a_plus ** 2 + B * a_minus ** 2))
            * tensor(basis(2, 0), basis(2, 0)) +
            tht * (A * a_plus ** 2 + B * a_minus ** 2) * tensor(basis(2, 0), basis(2, 1)) +
            (b ** 2 * cos(tht) * sqrt(1 - lmb) * (A + B) + 1j * b * sin(tht) * sqrt(1 - tht ** 2) * (
            A * a_plus + B * a_minus))
            * tensor(basis(2, 1), basis(2, 0)) +
            b * tht * (A * a_plus + B * a_minus) * tensor(basis(2, 1), basis(2, 1))).unit()

    # Uses Gram-Schmidt to find component of ph_1 that is orthogonal to ph_0
    ph_1o = (ph_1 - (ph_0.dag() * ph_1).tr() / (ph_0.dag() * ph_0).tr() * ph_0).unit()

    # Calculates the other two ph states by finding the eigenstates of Id-(|ph_0><ph_0|+|ph_1><ph_1|)
    # Projection operator for ps_0 and ps_1 states
    proj_phi = ph_0*ph_0.dag() + ph_1o*ph_1o.dag()
    # Orthogonal projection
    orth_proj_ph = qeye([2, 2]) - proj_phi
    # Finds the eigenvalues and corresponding eigenstates
    ev, phis = orth_proj_ph.eigenstates()
    # Select states corresponding to an ev of 1
    ph_2 = phis[2]
    ph_3 = phis[3]

    # Unitary V
    V = ps_0*ph_0.dag() + ps_1o*ph_1o.dag() + ps_2*ph_2.dag() + ps_3*ph_3.dag()

    # Print statements for checking this is truly unitary
    if show:
        print('Unitary V: {}'.format(V))
        print('V*V^dag:')
        print(V*V.dag())
        print('Stationary state:')
        print(sPsi)
        print('Stationary state ^2:')
        print(sPsi*sPsi)

    return V, sPsi


if __name__ == '__main__':
    # Checks
    tht = 0.2
    phi = np.pi/4
    lmb = 0.8
    print('ss:')
    ss(tht, lmb, phi, True)
    print('TO:')
    TO(tht, lmb, phi, True)
    print('Id-T:')
    Id_T_heisenberg(tht, lmb, phi, True)
    print('UnitaryV')
    unitaryV(tht, lmb, phi, True)

    # Function for calculating Kraus operators
    # Also returns the corresponding pseudo-unitary
    # Kraus operators are used to update the state, unitary is used to calculate next measurement
    # Inputs are respectively, the param of interest, our rough estimate, lambda, phi and a list containing the measurements
    # adaptive_v3: added functionality to send 1 measurement for use when calculating classical FI
    def kraus(theta_u, theta_ru, lmbd_u, phi_u, meas_u):
        # Not a true unitary as only it's action on |000>, |100>, |010> and |110> states are important
        unitry = Qobj([[np.cos(theta_u) * sqrt(1 - theta_u ** 2), 0, 1j * np.sin(theta_u) * sqrt(1 - lmbd_u), 0],
                       [0, 0, sqrt(lmbd_u) * np.exp(1j * phi_u), 0],
                       [1j * np.sin(theta_u) * sqrt(1 - theta_u ** 2), 0, np.cos(theta_u) * sqrt(1 - lmbd_u), 0],
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
        VU = tensor(qeye(2), V) * (tensor(qeye(2), P23) * tensor(unitry, qeye(2)) * tensor(qeye(2), P23))
        # Checks whether the function was sent 2 measurements or not
        if len(meas_u) == 2:
            # Calculates Kraus operators as Tr_e(Io|0><meas|*U)
            K_0 = (tensor(qeye(2), qeye(2), fock(2, 0) * meas_u[0].dag()) * VU).ptrace([0, 1])
            K_1 = (tensor(qeye(2), qeye(2), fock(2, 0) * meas_u[1].dag()) * VU).ptrace([0, 1])
            # Kraus operators are returned in a list
            K = [K_0, K_1]
            # For checking they're proper Kraus operators
            # print('Kraus check:')
            # print(K[0].dag()*K[0] + K[1].dag()*K[1])
        else:
            # Calculates a single Kraus operator as Tr_e(Io|0><meas|*U)
            # Used in fischer_cont() where measurement choice is already known
            K = (tensor(qeye(2), qeye(2), fock(2, 0) * meas_u[0].dag()) * VU).ptrace([0, 1])
        return K, VU, V, unitry

    K, *_ = kraus(tht, tht, lmb, phi, [fock(2, 0), fock(2, 1)])
    T = spre(K[0])*spost(K[0].dag()) + spre(K[1])*spost(K[1].dag())
    print(T)
