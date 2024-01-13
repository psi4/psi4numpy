# -*- coding: utf-8 -*-
"""
This script simulates vibrational circular dichroism (VCD) spectra
at the SCF level of thoery. This requires the calculation of nuclear
hessians, atomic polar tensors (APTs), and atomic axial tensors (AATs)
making use of one and two electron integrals and their associated
derivatives from PSI4.

References: 
1. "Coupled-cluster open shell analytic gradients: Implementation of the
direct product decomposition approach in energy gradient calculations",
J. Gauss, J. F. Stanton, R. J. Bartlett
DOI: 10.1063/1.460915

2. "Theory of vibrational circular dichroism", P. J. Stephens
DOI: 10.1021/j100251a006

3. "Efficient calculation of vibrational magnetic dipole transition 
moments and rotational strenths",
R. D. Amos, N. C. Handy, K. J. Jalkanen, P. J. Stephens
DOI: 10.1016/0009-2614(87)80046-5

4. "A new population analysis based on atomic polar tensors"
J. Cioslowski
DOI: 10.1021/ja00204a001
"""

__authors__ = "Kirk C. Pearce"
__credits__ = ["Kirk C. Pearce"]
__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2020-05-29"

import time
import numpy as np
import math
import copy
import psi4
from psi4 import *

# Setup NumPy options
np.set_printoptions(
    precision=10, 
    linewidth=200, 
    suppress=True, 
    threshold=sys.maxsize
)

psi4.set_memory(int(1e9), False)
psi4.core.set_output_file('output.dat', False)
psi4.core.set_num_threads(4)

# Useful Constants
psi_c = psi4.constants.get("Natural unit of velocity") * 100                                                    # speed of light in cm/s
psi_na = psi4.constants.na                                                                                      # Avogdro's number
psi_alpha = psi4.constants.get("fine-structure constant")                                                       # finite-structure constant
psi_h = psi4.constants.h                                                                                        # Planck's constant in J s
psi_hbar = psi_h / (2 * np.pi)                                                                                  # hbar in J s

# Unit Conversions
#
# Energy conversions
hartree2joule = psi4.constants.get("Atomic unit of energy")                                                     # Eh -> J
#
# Distance conversions
psi_bohr2m = psi4.constants.bohr2m                                                                              # bohr -> m
sqbohr2sqmeter = psi_bohr2m ** 2                                                                                # bohr^2 -> m^2
psi_bohr2angstroms = psi4.constants.bohr2angstroms                                                              # bohr -> Ang
psi_m2angstroms = 10**10                                                                                        # m -> Ang
#
# Mass conversions
amu2kg = psi4.constants.get("Atomic mass constant")                                                             # amu -> kg
psi_au2amu = psi4.constants.au2amu                                                                              # m_e -> amu
#
# Dipole moment type conversions
psi_dipmom_au2debye = psi4.constants.dipmom_au2debye                                                            # e a0 -> D (C m)
psi_dipmom_au2si = psi4.constants.dipmom_au2si                                                                  # e a0 -> C m
conv_dip_grad_au2DAamu = (psi_dipmom_au2debye / psi_bohr2angstroms) * (1 / np.sqrt(psi_au2amu))                 # (e a0 / a0)/(m_e^(1/2)) -> (D/A)/(amu^(1/2))
conv_cgs = 6.46047502185 * (10**(-36))                                                                          # (e a0)^2 -> (esu * cm)^2
#
# IR frequency conversions
omega2nu = 1./(psi_c*2*np.pi)                                                                                   # omega (s^-1)  -> nu (cm^-1)
psi4_hartree2wavenumbers = psi4.constants.hartree2wavenumbers                                                   # Eh -> cm^-1
conv_ir_au2DAamu = conv_dip_grad_au2DAamu ** 2                                                                  # (e a0 / a0)^2/m_e -> (D/A)^2/amu
conv_freq_au2wavenumbers = np.sqrt((1 / psi_au2amu) * hartree2joule / (sqbohr2sqmeter * amu2kg)) * omega2nu     # (Eh / (bohr^2 m_e)) -> cm^-1
conv_freq_wavenumbers2hartree = 1 / psi4_hartree2wavenumbers                                                    # cm^-1 -> Eh

# Convert IR intensities from (D/A)^2/amu to km/mol:
#
# Conversion factor for taking IR intensities from ((D/A)^2 / amu) to a.u. ((e a0 / a0)^2 / me):
conv_kmmol = (1 / conv_dip_grad_au2DAamu) ** 2
# Multiply by (1 / (4 pi eps_0)) in a.u. (1 / (e^2 / a0 Eh)) to get to units of (Eh a0 / me)
# Multiply by (Na pi / 3 c^2) in a.u. (Na = mol^-1; c = Eh / alpha^2 me) to get to units of a0/mol
conv_kmmol *= psi_na * np.pi * (1 / 3) * psi_alpha**2
# Convert to units of km/mol
conv_kmmol *= psi_bohr2m * (1 / 1000)
#
conv_ir_DAamu2kmmol = conv_kmmol                                                                                # (D/A)^2/amu -> km/mol


# Specify Molecule
mol = psi4.geometry("""
    O       0.0000000000        1.3192641900       -0.0952542913
    O      -0.0000000000       -1.3192641900       -0.0952542913
    H       1.6464858700        1.6841036400        0.7620343300
    H      -1.6464858700       -1.6841036400        0.7620343300
    symmetry c1
    units bohr
    noreorient
    no_com
""")

psi4.core.set_active_molecule(mol)

geom = psi4.core.Matrix.to_array(mol.geometry())

# Set Psi4 Options
options = {
    #'BASIS':'STO-3G_dalton',
    'SCF_TYPE':'PK',
    'E_CONVERGENCE':1e-12,
    'D_CONVERGENCE':1e-12}

# Define custom basis set to match STO-3G from DALTON
def define_custom_basis(mol, role):
    basstrings = {}
    mol.set_basis_all_atoms("sto-3g", role=role)
    basstrings['sto-3g'] = """ 
spherical
****
H 0
S 3 1.00
      3.4252509 0.15432897
      0.6239137 0.53532814
      0.1688554 0.44463454
****
O 0
S 3 1.00
    130.7093200 0.15432897
     23.8088610 0.53532814
      6.4436083 0.44463454
SP 3 1.00
      5.0331513 -0.09996723 0.15591627
      1.1695961 0.39951283 0.60768372
      0.3803890 0.70011547 0.39195739
****
"""
    return basstrings
qcdb.libmintsbasisset.basishorde['STO-3G_DALTON'] = define_custom_basis
core.set_global_option("BASIS", "STO-3G_dalton")

psi4.set_options(options)

# Perform SCF Energy Calculation
rhf_e, wfn = psi4.frequency('SCF', return_wfn=True)

# Relevant Variables
natoms = mol.natom()
nmo = wfn.nmo()
nocc = wfn.doccpi()[0]
nvir = nmo - nocc

# MO Coefficients
C = wfn.Ca_subset("AO", "ALL")
npC = psi4.core.Matrix.to_array(C)

# Integral generation from Psi4's MintsHelper
mints = psi4.core.MintsHelper(wfn.basisset())

# Build T, V, and S
T = mints.ao_kinetic()
npT = psi4.core.Matrix.to_array(T)
V = mints.ao_potential()
npV = psi4.core.Matrix.to_array(V)
S = mints.ao_overlap()
npS = psi4.core.Matrix.to_array(S)

# Transform S to MO Basis
npS_mo = 2.0 * np.einsum('uj,vi,uv', npC, npC, npS, optimize=True)

# Build AO Dipole Integrals
MU = mints.ao_dipole()
AM = mints.ao_angular_momentum()
npMU = []
npAM = []
npMU_mo = []
npAM_mo = []
for cart in range(3):
    npMU.append(psi4.core.Matrix.to_array(MU[cart]))
    npAM.append(psi4.core.Matrix.to_array(AM[cart]))
    npAM[cart] *= 0.5
    npMU_mo.append(np.einsum('uj,vi,uv', npC, npC, npMU[cart], optimize=True))
    npAM_mo.append(np.einsum('uj,vi,uv', npC, npC, npAM[cart], optimize=True))

# Build ERIs
ERI = mints.mo_eri(C, C, C, C)
npERI = psi4.core.Matrix.to_array(ERI)
# Physicist notation
npERI = npERI.swapaxes(1, 2)

# Build Core Hamiltonian in AO basis
H_core = npT + npV

# Transform H to MO basis
H = np.einsum('uj,vi,uv', npC, npC, H_core, optimize=True)

# Build Fock Matrix
F = H + 2.0 * np.einsum('pmqm->pq', npERI[:, :nocc, :, :nocc], optimize=True)
F -= np.einsum('pmmq->pq', npERI[:, :nocc, :nocc, :], optimize=True)

# Occupied and Virtual Orbital Energies
F_occ = np.diag(F)[:nocc]
F_vir = np.diag(F)[nocc:nmo]

# Build Reference OPDM
npD_ao = 2.0 * np.einsum('ui,vi->uv', npC[:, :nocc], npC[:, :nocc], optimize=True)
ref_opdm = np.zeros((nmo, nmo))
ref_opdm[:nocc, :nocc] = np.diag(np.repeat(2, nocc))

# Build Total OPDM
Ppq = np.zeros((nmo, nmo))
Ppq += ref_opdm

# Build Reference TPDM
ref_tpdm = np.zeros((nmo, nmo, nmo, nmo))
ref_tpdm += 2.0 * np.einsum("pr,qs->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm -= 1.0 * np.einsum("ps,qr->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm = -0.25 * ref_tpdm

# Build Total TPDM
Ppqrs = np.zeros((nmo, nmo, nmo, nmo))
Ppqrs += ref_tpdm


# Build I' intermediate
# where I'_pq = - (1/2) * [ fpp(Ppq + Pqp) + sum_rs (Prs * (4<rp|sq> - <rp|qs> - <rq|ps>)) * kronecker_delta(q,occ) + ...
#         ... + sum_rst (Pqrst <pr|st> + Prqst <rp|st> + Prsqt <rs|pt> + Prstq <rs|tp>) ]
Ip = np.zeros((nmo, nmo))

# I'pq += fpp(Ppq + Pqp)
Ip += np.einsum("pr,rq->pq", F, Ppq, optimize=True)
Ip += np.einsum("qr,rp->pq", Ppq, F, optimize=True)

# I'_pq += sum_rst (Pqrst <pr|st> + Prqst <rp|st> + Prsqt <rs|pt> + Prstq <rs|tp>)
Ip += np.einsum('qrst,prst->pq', Ppqrs, npERI, optimize=True)
Ip += np.einsum('rqst,rpst->pq', Ppqrs, npERI, optimize=True)
Ip += np.einsum('rsqt,rspt->pq', Ppqrs, npERI, optimize=True)
Ip += np.einsum('rstq,rstp->pq', Ppqrs, npERI, optimize=True)

# I'_pq += sum_rs Prs(4<rp|sq> - <rp|qs> - <rq|ps>) kronecker_delta(q,occ)
Ip[:, :nocc] += 4.0 * np.einsum('rs,rpsq->pq', Ppq , npERI[:, :, :, :nocc], optimize=True)
Ip[:, :nocc] -= 1.0 * np.einsum('rs,rpqs->pq', Ppq , npERI[:, :, :nocc, :], optimize=True)
Ip[:, :nocc] -= 1.0 * np.einsum('rs,rqps->pq', Ppq , npERI[:, :nocc, :, :], optimize=True)

Ip *= -0.5


# Build I'' ,
# where I''_pq = I'_qp    if (p,q) = (a,i)
#              = I'_pq    otherwise
Ipp = copy.deepcopy(Ip)
Ipp[nocc:, :nocc] = Ip[:nocc, nocc:].T

# Build X_ai = I'_ia - I'_ai
X = Ip[:nocc, nocc:].T - Ip[nocc:, :nocc]

# Build Idenity matrices in nocc/nvir dimensions
I_occ = np.diag(np.ones(nocc))
I_vir = np.diag(np.ones(nvir))

# Build epsilon_a - epsilon_i matrix
eps = np.asarray(wfn.epsilon_a_subset("AO", "ALL"))
eps_diag = eps[nocc:].reshape(-1, 1) - eps[:nocc]

# Build the electronic hessian, G
# where G = ((epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker_delta(i,j)) * (4<ij|ab> - <ij|ba> - <ia|jb>)

# G += 4<ij|ab> - <ij|ba> - <ia|jb>
G =  4.0 * npERI[:nocc, :nocc, nocc:, nocc:]
G -= 1.0 * npERI[:nocc, :nocc, nocc:, nocc:].swapaxes(2, 3)
G -= 1.0 * npERI[:nocc, nocc:, :nocc, nocc:].swapaxes(1, 2)

# Change shape of G from ij,ab to ia,jb
G = G.swapaxes(1, 2)

# G += (epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker delta(i,j)
G += np.einsum('ai,ij,ab->iajb', eps_diag, I_occ, I_vir, optimize=True)

# Take Transpose of G_iajb
G = G.T.reshape(nocc * nvir, nocc * nvir)

# Solve G^T(ai,bj) Z(b,j) = X(a,i)
X = X.reshape(nocc * nvir, -1)
Z = np.linalg.solve(G, X).reshape(nvir, -1)

# Relax OPDM
# Ppq(a,i) = Ppq(i,a) = - Z(a,i)
Ppq[:nocc, nocc:] = -Z.T
Ppq[nocc:, :nocc] = -Z

# Build Lagrangian, I, where
# I(i,j) = I''(i,j) + sum_ak ( Z(a,k) * [ 2<ai|kj> - <ai|jk>  + 2<aj|ki> - <aj|ik> ])
# I(i,a) = I''(i,a) + Z(a,i) * eps(i)
# I(a,i) = I''(a,i) + Z(a,i) * eps(i)
# I(a,b) = I''(a,b)
I = copy.deepcopy(Ipp)
# I(i,j) 
I[:nocc, :nocc] += 2.0 * np.einsum('ak,aikj->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] -= 1.0 * np.einsum('ak,aijk->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] += 2.0 * np.einsum('ak,ajki->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] -= 1.0 * np.einsum('ak,ajik->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
# I(a,i)
I[nocc:, :nocc] += Z * F_occ
# I(i,a)
I[:nocc, nocc:] += (Z * F_occ).T



# Build Integral Derivatives
cart = ['_X', '_Y', '_Z']
oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}

deriv1_mat = {}
deriv1_np = {}

# 1st Derivative of OEIs
for atom in range(natoms):
    for key in oei_dict:
        deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
        for p in range(3):
            map_key = key + str(atom) + cart[p]
            deriv1_np[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])

# 1st Derivative of TEIs
for atom in range(natoms):
    key = "TEI"
    deriv1_mat[key + str(atom)] = mints.mo_tei_deriv1(atom, C, C, C, C)
    for p in range(3):
        map_key = key + str(atom) + cart[p]
        deriv1_np[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])

# Build Fpq^x
F_grad = {}
for atom in range(natoms):
    for p in range(3):
        key = str(atom) + cart[p]
        F_grad[key] = copy.deepcopy(deriv1_np["T" + key])
        F_grad[key] += deriv1_np["V" + key]
        F_grad[key] += 2.0 * np.einsum('pqmm->pq', deriv1_np["TEI" + key][:, :, :nocc, :nocc], optimize=True)
        F_grad[key] -= 1.0 * np.einsum('pmmq->pq', deriv1_np["TEI" + key][:, :nocc, :nocc, :], optimize=True)


Hes = {};
deriv2_mat = {}
deriv2_np = {}

Hes["N"] = np.zeros((3 * natoms, 3 * natoms))
Hes["S"] = np.zeros((3 * natoms, 3 * natoms))
Hes["T"] = np.zeros((3 * natoms, 3 * natoms))
Hes["V"] = np.zeros((3 * natoms, 3 * natoms))
Hes["TEI"] = np.zeros((3 * natoms, 3 * natoms))
Hes["R"] = np.zeros((3 * natoms, 3 * natoms))
Hessian = np.zeros((3 * natoms, 3 * natoms))

# 2nd Derivative of Nuclear Repulsion
Hes["N"] = np.asarray(mol.nuclear_repulsion_energy_deriv2())

psi4.core.print_out("\n\n")
Mat = psi4.core.Matrix.from_array(Hes["N"])
Mat.name = "NUCLEAR HESSIAN"
Mat.print_out()

# 2nd Derivative of OEIs
for atom1 in range(natoms):
    for atom2 in range(natoms):
        for key in  oei_dict:
            string = key + str(atom1) + str(atom2)
            deriv2_mat[string] = mints.mo_oei_deriv2(oei_dict[key], atom1, atom2, C, C)
            pq = 0
            for p in range(3):
                for q in range(3):
                    map_key = string + cart[p] + cart[q]
                    deriv2_np[map_key] = np.asarray(deriv2_mat[string][pq])
                    pq += 1
                    row = 3 * atom1 + p
                    col = 3 * atom2 + q
                    if key == "S":
                        Hes[key][row][col] = np.einsum("pq,pq->", I, deriv2_np[map_key], optimize=True)
                    else:
                        Hes[key][row][col] = np.einsum("pq,pq->", Ppq, deriv2_np[map_key], optimize=True)

for key in Hes:
    Mat = psi4.core.Matrix.from_array(Hes[key])
    if key in oei_dict:
        Mat.name = oei_dict[key] + " HESSIAN"
        Mat.print_out()
        psi4.core.print_out("\n")


for atom1 in range(natoms):
    for atom2 in range(natoms):
        string = "TEI" + str(atom1) + str(atom2)
        deriv2_mat[string] = mints.mo_tei_deriv2(atom1, atom2, C, C, C, C)
        pq = 0
        for p in range(3):
            for q in range(3):
                map_key = string + cart[p] + cart[q]
                deriv2_np[map_key] = np.asarray(deriv2_mat[string][pq])
                pq = pq + 1
                row = 3 * atom1 + p
                col = 3 * atom2 + q

                Hes["TEI"][row][col] +=  2.0 * np.einsum("pq,pqmm->", ref_opdm, deriv2_np[map_key][:, :, :nocc, :nocc], optimize=True)
                Hes["TEI"][row][col] += -1.0 * np.einsum("pq,pmmq->", ref_opdm, deriv2_np[map_key][:, :nocc, :nocc, :], optimize=True)

                Hes["TEI"][row][col] +=  2.0 * np.einsum("pq,pqmm->", Ppq-ref_opdm, deriv2_np[map_key][:, :, :nocc, :nocc], optimize=True)
                Hes["TEI"][row][col] += -1.0 * np.einsum("pq,pmmq->", Ppq-ref_opdm, deriv2_np[map_key][:, :nocc, :nocc, :], optimize=True)

                Hes["TEI"][row][col] += np.einsum("pqrs,prqs->", Ppqrs, deriv2_np[map_key], optimize=True)

TEIMat = psi4.core.Matrix.from_array(Hes["TEI"])
TEIMat.name = " TEI HESSIAN"
TEIMat.print_out()


# Solve the first-order CPHF equations here,  G_aibj Ubj^x = Bai^x (Einstein summation),
# where G is the electronic hessian,
# G_aibj = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>,
# where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
# x refers to the perturbation, Ubj^x are the corresponsing CPHF coefficients
# and Bai^x = Sai^x * epsilon_ii - Fai^x + Smn^x  * (2<am|in> - <am|ni>),
# where, S^x =  del(S)/del(x), F^x =  del(F)/del(x).

B = {}
U = {}
# Build Bai^x
for atom in range(natoms):
    for p in range(3):
        key = str(atom) + cart[p]
        B[key] =  np.einsum("ai,ii->ai", deriv1_np["S" + key][nocc:, :nocc], F[:nocc, :nocc], optimize=True)
        B[key] -= F_grad[key][nocc:, :nocc]
        B[key] +=  2.0 * np.einsum("amin,mn->ai", npERI[nocc:, :nocc, :nocc, :nocc], deriv1_np["S" + key][:nocc, :nocc], optimize=True)
        B[key] += -1.0 * np.einsum("amni,mn->ai", npERI[nocc:, :nocc, :nocc, :nocc], deriv1_np["S" + key][:nocc, :nocc], optimize=True)

        # Compute U^x, where
        # U_ij^x = - 1/2 S_ij^a
        # U_ai^x = G^(-1)_aibj * B_bj^x
        # U_ia^x = - (U_ai^x + S_ai^x)
        # U_ab^x = - 1/2 S_ab^a
        U[key] = np.zeros((nmo, nmo))
        U[key][:nocc, :nocc] = - 0.5 * deriv1_np["S" + key][:nocc, :nocc]
        U[key][nocc:, :nocc] = (np.linalg.solve(G.reshape(nocc * nvir, -1), B[key].reshape(nocc * nvir, -1))).reshape(nvir, nocc)
        U[key][:nocc, nocc:] = - (U[key][nocc:, :nocc] + deriv1_np["S" + key][nocc:, :nocc]).T
        U[key][nocc:, nocc:] = - 0.5 * deriv1_np["S" + key][nocc:, nocc:]


# Build the response hessian now
for r in range(3 * natoms):
    for c in range(3 * natoms):
        atom1 = r // 3
        atom2 = c // 3

        p = r % 3
        q = c % 3

        key1  = str(atom1) + cart[p]
        key2  = str(atom2) + cart[q]

        # Ipq Contributions to the Hessian:
        # d^2E/dxdy += P+(xy) sum_pq Ipq ( Upt^x * Uqt^y - Spt^x * Sqt^y)
        #
        Hes["R"][r][c] +=  1.0 * np.einsum('pq,pt,qt->', I, U[key1], U[key2], optimize=True)
        Hes["R"][r][c] +=  1.0 * np.einsum('pq,pt,qt->', I, U[key2], U[key1], optimize=True)
        #
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,pt,qt->', I, deriv1_np["S" + key2], deriv1_np["S" + key1], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,pt,qt->', I, deriv1_np["S" + key1], deriv1_np["S" + key2], optimize=True)



        # Ppq Contributions to the Hessian:
        # d^2E/dxdt += P+(xy) sum_pq Ppq [ sum_t ( Upt^x * ftq^y + Utq^x * fpt^y ) ] 
        #
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tp,tq->', Ppq, U[key1], F_grad[key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tp,tq->', Ppq, U[key2], F_grad[key1], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tq,pt->', Ppq, U[key1], F_grad[key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tq,pt->', Ppq, U[key2], F_grad[key1], optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pq Ppq ( sum_tm [ Utm^x * ( <pm||qt>^y + <pt||qm>^y ) ] )
        #
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tm,pqmt->', Ppq, U[key1][:, :nocc], deriv1_np["TEI" + key2][:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tm,ptmq->', Ppq, U[key1][:, :nocc], deriv1_np["TEI" + key2][:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tm,pqtm->', Ppq, U[key1][:, :nocc], deriv1_np["TEI" + key2][:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tm,pmtq->', Ppq, U[key1][:, :nocc], deriv1_np["TEI" + key2][:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tm,pqmt->', Ppq, U[key2][:, :nocc], deriv1_np["TEI" + key1][:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tm,ptmq->', Ppq, U[key2][:, :nocc], deriv1_np["TEI" + key1][:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tm,pqtm->', Ppq, U[key2][:, :nocc], deriv1_np["TEI" + key1][:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tm,pmtq->', Ppq, U[key2][:, :nocc], deriv1_np["TEI" + key1][:, :nocc, :, :], optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pq Ppq ( sum_tv [ Utp^x * Uvq^y * ftv ] )
        #
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tp,vq,tv->', Ppq, U[key1], U[key2], F, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pq,tp,vq,tv->', Ppq, U[key2], U[key1], F, optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pq Ppq ( sum_tvm [ Utp^x * Uvm^y * ( <tm||qv> + <tv||qm> ) ] )
        #
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tp,vm,tmqv->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tp,vm,tmvq->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tp,vm,tvqm->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tp,vm,tvmq->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tp,vm,tmqv->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tp,vm,tmvq->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tp,vm,tvqm->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tp,vm,tvmq->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :, :nocc, :], optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pq Ppq ( sum_tvm [ Utq^x * Uvm^y * ( <pm||tv> + <pv||tm> ) ] )
        #
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tq,vm,pmtv->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tq,vm,pmvt->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tq,vm,pvtm->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tq,vm,pvmt->', Ppq, U[key1], U[key2][:, :nocc], npERI[:, :, :nocc, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tq,vm,pmtv->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tq,vm,pmvt->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :nocc, :, :], optimize=True)
        Hes["R"][r][c] += 2.0 * np.einsum('pq,tq,vm,pvtm->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :, :, :nocc], optimize=True)
        Hes["R"][r][c] -= 1.0 * np.einsum('pq,tq,vm,pvmt->', Ppq, U[key2], U[key1][:, :nocc], npERI[:, :, :nocc, :], optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pq Ppq ( sum_tvm [ 1/2 * Utm^x * Uvm^y * ( <pt||qv> + <pv||qt> ) ] )
        #
        Hes["R"][r][c] += 0.5 * 2.0 * np.einsum('pq,tm,vm,ptqv->', Ppq, U[key1][:, :nocc], U[key2][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] -= 0.5 * 1.0 * np.einsum('pq,tm,vm,ptvq->', Ppq, U[key1][:, :nocc], U[key2][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] += 0.5 * 2.0 * np.einsum('pq,tm,vm,pvqt->', Ppq, U[key1][:, :nocc], U[key2][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] -= 0.5 * 1.0 * np.einsum('pq,tm,vm,pvtq->', Ppq, U[key1][:, :nocc], U[key2][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] += 0.5 * 2.0 * np.einsum('pq,tm,vm,ptqv->', Ppq, U[key2][:, :nocc], U[key1][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] -= 0.5 * 1.0 * np.einsum('pq,tm,vm,ptvq->', Ppq, U[key2][:, :nocc], U[key1][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] += 0.5 * 2.0 * np.einsum('pq,tm,vm,pvqt->', Ppq, U[key2][:, :nocc], U[key1][:, :nocc], npERI, optimize=True)
        Hes["R"][r][c] -= 0.5 * 1.0 * np.einsum('pq,tm,vm,pvtq->', Ppq, U[key2][:, :nocc], U[key1][:, :nocc], npERI, optimize=True)


        
        # Ppqrs Contributions to the Hessian:
        # d^2E/dxdy += P+(xy) sum_pqrs ( sum_t [ Utp^x * <tq|rs>^y + Utq^x * <pt|rs>^y + Utr^x * <pq|ts>^y + Uts^x * <pq|rt>^y ] )
        #
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,trqs->', Ppqrs, U[key1], deriv1_np["TEI" + key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,prts->', Ppqrs, U[key1], deriv1_np["TEI" + key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tr,ptqs->', Ppqrs, U[key1], deriv1_np["TEI" + key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,ts,prqt->', Ppqrs, U[key1], deriv1_np["TEI" + key2], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,trqs->', Ppqrs, U[key2], deriv1_np["TEI" + key1], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,prts->', Ppqrs, U[key2], deriv1_np["TEI" + key1], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tr,ptqs->', Ppqrs, U[key2], deriv1_np["TEI" + key1], optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,ts,prqt->', Ppqrs, U[key2], deriv1_np["TEI" + key1], optimize=True)
        #
        # d^2E/dxdy += P+(xy) sum_pqrs ( sum_tv [ Utp^x * Uvq^y * <tv|rs> + Utp^x * Uvr^y * <tq|vs> + Utp^x * Uvs^y * <tq|rv> + ...
        #                                   ... + Utq^x * Uvr^y * <pt|vs> + Utq^x * Uvs^y * <pt|rv> + Utr^x * Uvs^y * <pq|tv> ] )
        #
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vq,tvrs->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vr,tqvs->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vs,tqrv->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,vr,ptvs->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,vs,ptrv->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tr,vs,pqtv->', Ppqrs, U[key1], U[key2], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vq,tvrs->', Ppqrs, U[key2], U[key1], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vr,tqvs->', Ppqrs, U[key2], U[key1], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tp,vs,tqrv->', Ppqrs, U[key2], U[key1], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,vr,ptvs->', Ppqrs, U[key2], U[key1], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tq,vs,ptrv->', Ppqrs, U[key2], U[key1], npERI, optimize=True)
        Hes["R"][r][c] += 1.0 * np.einsum('pqrs,tr,vs,pqtv->', Ppqrs, U[key2], U[key1], npERI, optimize=True)


# Build Total Hessian
for key in Hes:
    Hessian += Hes[key]

# Symmetrize Hessian
Hessian = (Hessian + Hessian.T)/2
print("\nMolecular Hessian (a.u.):\n", Hessian)

Mat = psi4.core.Matrix.from_array(Hessian)
Mat.name = " TOTAL HESSIAN"
Mat.print_out()

# Test Hessian
psi_H = wfn.hessian()
python_H = psi4.core.Matrix.from_array(Hessian)
psi4.compare_matrices(psi_H, python_H, 8, "RHF-HESSIAN-TEST")

# Mass Weight the Hessian Matrix:
masses = np.array([mol.mass(i) * (1 / psi_au2amu) for i in range(mol.natom())])
M = np.diag(1 / np.sqrt(np.repeat(masses, 3)))
mH = M.T.dot(Hessian).dot(M)
#print("\nMass-weighted Hessian (a.u.):\n", mH)
#print("\nMass-weighted Hessian (hartree/(bohr^2 amu)):\n", mH * (1 / psi_au2amu))


# Mass-weighted Normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH)
#print(k2) #eigenvalues in a.u. (hartree/bohr^2 m_e)
#print(Lxm) #eigenvectors are unitless
#print(k2 * (1 / psi_au2amu)) #eigenvalues in (hartree/(bohr^2 amu))
#print(k2 * (1 / psi_au2amu) * hartree2joule / (sqbohr2sqmeter * amu2kg)) #eigenvalues in (J/kg*m^2), effectively s^-2

normal_modes = []
mode = 3 * natoms - 1
print("\nNormal Modes (cm^-1):")
while mode >= 6:
    if k2[mode] >= 0.0:
        normal_modes.append(np.sqrt(k2[mode]))
        print("%.2f" % (np.sqrt(k2[mode]) * conv_freq_au2wavenumbers))
    else:
        normal_modes.append(np.sqrt(abs(k2[mode])))
        print("%.2fi" % (np.sqrt(abs(k2[mode])) * conv_freq_au2wavenumbers))
    mode -= 1

# Test frequencies
psi_freq = np.flip(wfn.frequencies().to_array(), 0)
for i in range(len(normal_modes)):
    psi4.compare_values(psi_freq[i], normal_modes[i] * conv_freq_au2wavenumbers, 6, "FREQ-TEST")

# Un-mass-weight the eigenvectors
Lx = M.dot(Lxm) #now has units of m_e**(-1/2)

# Normal Coordinates transformation matrix
S = np.flip(Lx, 1)[:,:len(normal_modes)]
#print("\nNormal Coordinates transformation matrix (a.u.):\n", S)
#print("\nNormal Coordinates transformation matrix (amu**-(1/2)):\n", S * (1 / np.sqrt(psi_au2amu)))


# Compute Dipole Derivative
Gradient = {};
Gradient["MU"] = np.zeros((3 * natoms, 3))

# Add 1st derivative of electric dipole integrals
for atom in range(natoms):
    deriv1_mat["MU_" + str(atom)] = mints.ao_elec_dip_deriv1(atom)
    for mu_cart in range(3):
        for atom_cart in range(3):
            map_key = "MU" + cart[mu_cart] + "_" + str(atom) + cart[atom_cart]
            deriv1_np[map_key] = np.asarray(deriv1_mat["MU_" + str(atom)][3 * mu_cart + atom_cart])
            Gradient["MU"][3 * atom + atom_cart, mu_cart] += np.einsum("uv,uv->", deriv1_np[map_key], npD_ao)

# Add nuclear contribution to dipole derivative
zvals = np.array([mol.Z(i) for i in range(mol.natom())])
Z = np.zeros((3 * natoms, 3))
for i in range(len(zvals)):
    np.fill_diagonal(Z[(3 * i) : (3 * (i+1)),:], zvals[i])
Gradient["MU"] += Z

# Add orbital relaxation piece to dipole derivative
for atom in range(natoms):
    for mu_cart in range(3):
        for atom_cart in range(3):
            key = str(atom) + cart[atom_cart]
            Gradient["MU"][3 * atom + atom_cart, mu_cart] += 4.0 * np.einsum('pj,pj', U[key][:, :nocc], npMU_mo[mu_cart][:, :nocc], optimize=True)
print("\nDipole Derivatives (a.u.):\n", Gradient["MU"])
#print("\nDipole Derivatives (D/A):\n", Gradient["MU"] * (psi_dipmom_au2debye / psi_bohr2angstroms)) 

# Test dipole derivatives with Psi4
dipder = wfn.variables().get("CURRENT DIPOLE GRADIENT", None)
if dipder is not None:
    dipder = np.asarray(dipder).T 
PSI4_dipder = psi4.core.Matrix.from_array(dipder.T)
python_dipder = psi4.core.Matrix.from_array(Gradient["MU"])
psi4.compare_matrices(PSI4_dipder, python_dipder, 6, "DIPOLE_DERIVATIVE_TEST")  # TEST

# APT Population Analysis
# where Q^A = 1/3 ( dmu_x/dx_A + dmu_y/dy_A + dmu_z/dz_A), 
# where A represents an atom; x_A, y_A, z_A are cartesian coordinates of atom A; and mu_x, mu_y, mu_z are cartesian components of the dipole
Q = np.zeros(natoms)
for atom in range(natoms):
    for atom_cart in range(3):
        for mu_cart in range(3):
            if atom_cart == mu_cart:
                Q[atom] += Gradient["MU"][3 * atom + atom_cart][mu_cart]
Q *= (1/3)
print("\nAPT Population Analysis:\n", Q)

# Tranform dipole derivative to normal coordinate babis
dipder = copy.deepcopy(Gradient["MU"].T)
dip_grad = np.einsum('ij,jk->ik', dipder, S, optimize=True)
#print("\nDipole Gradient in Normal Coordinate Basis (D/(A*amu**(1/2))):\n", dip_grad * conv_dip_grad_au2DAamu)
#print("\nDipole Gradient in Normal Coordinate Basis (D/(A*amu**(1/2)))\n", np.einsum('ij,jk->ik', S.T, dipder.T, optimize=True) * conv_dip_grad_au2DAamu)

# Compute IR Intensities
IR_ints = np.zeros(len(normal_modes))
for i in range(3):
    for j in range(len(normal_modes)):
        IR_ints[j] += dip_grad[i][j] * dip_grad[i][j]
#print("\nIR Intensities (a.u.):\n", IR_ints)
#print("\nIR Intensities ((D/A)^2/amu):\n", IR_ints * conv_ir_au2DAamu)
#print("\nIR Intensities km/mol:\n", IR_ints * conv_ir_au2DAamu * conv_ir_DAamu2kmmol)


# Calculate APT, i.e. dipole transition moment, <0|mu|1>_i
APT = copy.deepcopy(dip_grad)
for i in range(len(normal_modes)):
    APT[:,i] *= (1 / np.sqrt(2 * normal_modes[i]))
print("\nElectric dipole transition moment, <0|mu|1>_i (a.u.):\n", APT)

# Calculate dipole strength, D_i = |<0|mu|1>_i|^2
dip_str = np.zeros(len(normal_modes))
for i in range(len(normal_modes)):
    dip_str[i] = np.einsum('j,j', APT[:,i], APT[:,i], optimize=True)
print("\nDipole Strengths, |<0|mu|1>_i|^2 (esu^2 cm^2 * 10**(-40)):\n", dip_str * conv_cgs * (10**40))


print("\n\nVibrational Frequencies and IR Intensities:\n------------------------------------------\n")
print("  mode        frequency               IR intensity\n========================================================")
print("           cm-1      hartrees      km/mol  (D/A)**2/amu \n--------------------------------------------------------")
for i in range(len(normal_modes)):
    print("  %3d    %7.2f     %8.6f     %7.3f      %6.4f" % (i + 1, normal_modes[i] * conv_freq_au2wavenumbers, normal_modes[i] * conv_freq_au2wavenumbers * conv_freq_wavenumbers2hartree, IR_ints[i] * conv_ir_au2DAamu * conv_ir_DAamu2kmmol, IR_ints[i] * conv_ir_au2DAamu))


# Solve the first-order CPHF equations here,  G_aibj Ubj^x = Bai^x (Einstein summation),
# where G is the electronic hessian,
# G_aibj = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>,
# where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
# x refers to the *magnetic field* perturbation, Ubj^x are the corresponsing CPHF coefficients
# and Bai^x = - Fai^x n> = - m_ai, where, F^x =  del(F)/del(x).

# Build epsilon_a - epsilon_i matrix
eps_m = np.asarray(wfn.epsilon_a_subset("AO", "ALL"))
eps_diag_m = eps[nocc:].reshape(-1, 1) - eps[:nocc]

# Build the magnetic hessian, G
# where G_m = ((epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker_delta(i,j)) + (<ij|ba> - <ia|jb>)

# G_m += <ij|ba> - <ia|jb>
G_m  =  1.0 * npERI[:nocc, :nocc, nocc:, nocc:].swapaxes(2, 3)
G_m += -1.0 * npERI[:nocc, nocc:, :nocc, nocc:].swapaxes(1, 2)

# Change shape of G_m from ij,ab to ia,jb
G_m = G_m.swapaxes(1, 2)

# G_m += (-epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker delta(i,j)
G_m += np.einsum('ai,ij,ab->iajb', eps_diag_m, I_occ, I_vir, optimize=True)

# Take transpose of G_m_iajb
G_m = G_m.T.reshape(nocc * nvir, nocc * nvir)

B_m = {}
U_m = {}
# Build Bai^x and solve for CPHF coefficents
# where Bai^x = (mu_mag)_ia^x
for p in range(3):
    key = cart[p]

    B_m[key] = npAM_mo[p][nocc:, :nocc]

    # Compute U^x, where
    U_m[key] = np.zeros((nmo, nmo))
    # U_ai^x = G^(-1)_aibj * B_bj^x
    U_m[key][nocc:, :nocc] = (np.linalg.solve(G_m.reshape(nocc * nvir, -1), B_m[key].reshape(nocc * nvir, -1))).reshape(nvir, nocc)
    # U_ia^x = U_ai^x
    U_m[key][:nocc, nocc:] = U_m[key][nocc:, :nocc].T
    #print("U_m[",key,"]:\n", U_m[key])

# Get half-differentiated overlap integrals for AAT computation
for atom in range(natoms):
    deriv1_mat["S_LEFT_HALF_" + str(atom)] = mints.mo_overlap_half_deriv1("LEFT", atom, C, C)
    for atom_cart in range(3):
        map_key1 = "S_LEFT_HALF_" + str(atom) + cart[atom_cart]
        deriv1_np[map_key1] = np.asarray(deriv1_mat["S_LEFT_HALF_" + str(atom)][atom_cart])

# Compute AAT
AAT_elec = np.zeros((3 * natoms, 3))
for atom in range(natoms):
    for atom_cart in range(3):
        for mu_cart in range(3):
            key1 = str(atom) + cart[atom_cart]
            key2 = cart[mu_cart]
            AAT_elec[3 * atom + atom_cart][mu_cart] += 2.0 * np.einsum("ai,ai", U[key1][nocc:, :nocc], U_m[key2][nocc:, :nocc], optimize=True)
            AAT_elec[3 * atom + atom_cart][mu_cart] += 2.0 * np.einsum("ai,ia", U_m[key2][nocc:, :nocc], deriv1_np["S_LEFT_HALF_" + key1][:nocc, nocc:], optimize=True)

levi_civit = np.zeros((3,3,3))
levi_civit[0][1][2] = levi_civit[1][2][0] = levi_civit[2][0][1] = 1
levi_civit[0][2][1] = levi_civit[1][0][2] = levi_civit[2][1][0] = -1
AAT_nuc = np.zeros((3 * natoms, 3))
# Nuclear contribution to AAT
for atom in range(natoms):
    for atom_cart in range(3):
        for mu_cart in range(3):
            for gamma in range(3):
                AAT_nuc[3 * atom + atom_cart][mu_cart] += 1.0 * levi_civit[atom_cart][mu_cart][gamma] * geom[atom][gamma] * (zvals[atom] / 4)

print("\nElectronic contribution to AAT:\n", AAT_elec)
#print("\nElectronic contribution to AAT (T^-1 A^-1 * 10^-6):\n", AAT_elec * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6)
print("\nNuclear contribution to AAT:\n", AAT_nuc)
AAT = AAT_elec + AAT_nuc
print("\nTotal AAT:\n", AAT)

# Compare AATs to DALTON's result
AAT_dalton = psi4.core.Matrix.from_list([
[-0.16438927,   -0.10446393,   -2.00901728],
[ 0.09439940,    0.00574249,    0.11809387],
[ 2.19819309,   -0.11635324,    0.15601962],
[-0.16438927,   -0.10446393,    2.00901728],
[ 0.09439940,    0.00574249,   -0.11809387],
[-2.19819309,    0.11635324,    0.15601962],
[-0.11803349,    0.11314253,   -0.10726430],
[-0.19342146,    0.00350944,    0.36522207],
[ 0.25524026,   -0.20123957,    0.12058354],
[-0.11803349,    0.11314253,    0.10726430],
[-0.19342146,    0.00350944,   -0.36522207],
[-0.25524026,    0.20123957,    0.12058354]])
AAT_python_mat = psi4.core.Matrix.from_array(AAT)
psi4.compare_matrices(AAT_dalton, AAT_python_mat, 8, "SCF-VCD-TEST")

AAT = np.einsum("ij,jk->ik", AAT.T, S, optimize=True)
for i in range(len(normal_modes)):
    AAT[:,i] *= np.sqrt(normal_modes[i] / 2)
print("\nMagnetic dipole transition moment, <0|mu|1>_i (a.u.):\n", AAT)

# Calculate rotational strength, R_i = <0|mu_elec|1>_i <1|mu_mag|0>_i
rot_str = np.zeros(len(normal_modes))
for i in range(len(normal_modes)):
    rot_str[i] = np.einsum('j,j', APT[:,i], AAT[:,i], optimize=True)

rot_str *= 2.0

# Convert AAT to cgs units (magnetic dipole operator requires multiplication by (1/c) = alpha in cgs units)
print("\nRotational Strengths, <0|mu_elec|1>_i <1|mu_mag|0>_i (esu^2 cm^2 * 10**(-44)):\n", rot_str * psi_alpha * conv_cgs * (10**44))

# Compare rotatory strengths to DALTON's result (within a sign)
rot_str_dalton = -1.0 * np.array([-50.538, 53.528, -28.607, 14.650, -1.098, -101.378])
for i in range(len(rot_str)):
    psi4.compare_values(rot_str_dalton[i], rot_str[i] * psi_alpha * conv_cgs * (10**44), 3, "ROTATORY-STRENGTHS-TEST")
