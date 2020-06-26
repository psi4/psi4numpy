# -*- coding: utf-8 -*-
"""
This script simulates vibrational circular dichroism (VCD) spectra
at the SCF level of thoery. This requires the calculation of nuclear
hessians, atomic polar tensors (APTs), and atomic axial tensors (AATs)
making use of one and two electron integrals and their associated
derivatives from PSI4.

References: 
1. "Theory of vibrational circular dichroism", P. J. Stephens
DOI: 10.1021/j100251a006

2. "Efficient calculation of vibrational magnetic dipole transition 
moments and rotational strenths",
R. D. Amos, N. C. Handy, K. J. Jalkanen, P. J. Stephens
DOI: 10.1016/0009-2614(87)80046-5
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

# Unit Conversions
hartree2joule = psi4.constants.get("Atomic unit of energy")     # Eh -> J
sqbohr2sqmeter = 5.29177e-11 * 5.29177e-11                      # bohr^2 -> m^2
amu2kg = psi4.constants.get("Atomic mass constant")             # amu -> kg
c_ = psi4.constants.get("Natural unit of velocity") * 100       # speed of light in cm/s
omega2nu = 1./(c_*2*np.pi)                                      # w -> nu
psi_dipmom_au2debye = psi4.constants.dipmom_au2debye
psi_bohr2angstroms = psi4.constants.bohr2angstroms

# Specify Molecule
mol = psi4.geometry("""
    O            0.000000000000     0.000000000000    -0.075791843897
    H            0.000000000000    -0.866811832375     0.601435781623
    H            0.000000000000     0.866811832375     0.601435781623
    symmetry c1
    units ang
    noreorient
""")
#mol = psi4.geometry("""
#    O     0.000000000000     0.000000000000    -0.134503695264
#    H     0.000000000000    -1.684916670000     1.067335684736
#    H     0.000000000000     1.684916670000     1.067335684736
#    symmetry c1
#    units bohr
#    noreorient
#""")

psi4.core.set_active_molecule(mol)

# Set Psi4 Options
options = {'BASIS':'STO-3G',
           'SCF_TYPE':'PK',
           'E_CONVERGENCE':1e-10,
           'D_CONVERGENCE':1e-10}
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
npMU = []
npMU_mo = []
for cart in range(len(MU)):
    npMU.append(psi4.core.Matrix.to_array(MU[cart]))
    npMU_mo.append(np.einsum('uj,vi,uv', npC, npC, npMU[cart], optimize=True))

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
# Transform to MO Basis
ref_opdm = np.einsum('iu,uv,vw,wx,xj', npC.T, npS.T, npD_ao, npS, npC, optimize=True)

# Build Total OPDM
Ppq = np.zeros((nmo, nmo))
Ppq += ref_opdm
#print("\n\nTotal OPDM:\n", Ppq)
#print("\nChecks:")
#print("OPDM is symmetric: ",np.allclose(Ppq, Ppq.T))
#print("OPDM trace = 10: ",np.isclose(sum(np.linalg.eigh(Ppq)[0]),10))

# Build Reference TPDM
ref_tpdm = np.zeros((nmo, nmo, nmo, nmo))
ref_tpdm += 2.0 * np.einsum("pr,qs->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm -= 1.0 * np.einsum("ps,qr->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm = -0.25 * ref_tpdm

# Build Total TPDM
Ppqrs = np.zeros((nmo, nmo, nmo, nmo))
Ppqrs += ref_tpdm
#print("\n\nTotal TPDM:\n", Ppqrs.reshape(nmo*nmo, nmo*nmo))


# Build I'
# I'_pq = - (1/2) * [ fpp(Ppq + Pqp) + sum_rs (Prs * (4<rp|sq> - <rp|qs> - <rq|ps>)) * kronecker_delta(q,occ) + ...
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
#print("\nI':\n",Ip)


# Build I'' ,
# where I''_pq = I'_qp    if (p,q) = (a,i)
#              = I'_pq    otherwise
Ipp = copy.deepcopy(Ip)
Ipp[nocc:, :nocc] = Ip[:nocc, nocc:].T

# Build X_ai = I'_ia - I'_ai
X = Ip[:nocc, nocc:].T - Ip[nocc:, :nocc]
#print("\nX:\n", X)

# Build Idenity matrices in nocc/nvir dimensions
I_occ = np.diag(np.ones(nocc))
I_vir = np.diag(np.ones(nvir))

# Build epsilon_a - epsilon_i matrix
eps = np.asarray(wfn.epsilon_a())
eps_diag = eps[nocc:].reshape(-1, 1) - eps[:nocc]

# Build the electronic hessian, G, where
# G = ((epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker_delta(i,j)) * (4<ij|ab> - <ij|ba> - <ia|jb>)

# G += 4<ij|ab> - <ij|ba> - <ia|jb>
G =  4.0 * npERI[:nocc, :nocc, nocc:, nocc:]
G -= 1.0 * npERI[:nocc, :nocc, nocc:, nocc:].swapaxes(2, 3)
G -= 1.0 * npERI[:nocc, nocc:, :nocc, nocc:].swapaxes(1, 2)

# Change shape of G from ij,ab to ia,jb
G = G.swapaxes(1, 2)

# G += (epsilon_a - epsilon_i) * kronecker_delta(a,b) * kronecker delta(i,j)
G += np.einsum('ai,ij,ab->iajb', eps_diag, I_occ, I_vir, optimize=True)

# Inverse of G
Ginv = np.linalg.inv(G.reshape(nocc * nvir, -1))
Ginv = Ginv.reshape(nocc, nvir, nocc, nvir)

# Take Transpose of G_iajb
G = G.T.reshape(nocc * nvir, nocc * nvir)
#print("\nMO Hessian, G:\n", G)

# Solve G^T(ai,bj) Z(b,j) = X(a,i)
X = X.reshape(nocc * nvir, -1)
Z = np.linalg.solve(G, X).reshape(nvir, -1)
#print("\nZ Vector:\n", X)

# Relax OPDM
# Ppq(a,i) = Ppq(i,a) = - Z(a,i)
Ppq[:nocc, nocc:] = -Z.T
Ppq[nocc:, :nocc] = -Z
#print("\n\nRelaxed Total OPDM:\n", Ppq)

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
#print("\n\nLagrangian I:\n", I)



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


## Build I'^x
## I'_pq^x = - (1/2) * [ fpp^x(Ppq + Pqp) + sum_rs (Prs * (4<rp|sq>^x - <rp|qs>^x - <rq|ps>^x)) * kronecker_delta(q,occ) + ...
##         ... + sum_rst (Pqrst <pr|st>^x + Prqst <rp|st>^x + Prsqt <rs|pt>^x + Prstq <rs|tp>^x) ]
#Ip_grad = {}
#for atom in range(natoms):
#    for pp in range(3):
#        key = str(atom) + cart[pp]
#
#        Ip_grad[key] = np.zeros((nmo, nmo))
#
#        # I'pq += fpp^x(Ppq + Pqp)
#        Ip_grad[key] += np.einsum("pr,rq->pq", F_grad[key], Ppq, optimize=True)
#        Ip_grad[key] += np.einsum("qr,rp->pq", Ppq, F_grad[key], optimize=True)
#
#        # I'_pq += sum_rst (Pqrst <pr|st>^x + Prqst <rp|st>^x + Prsqt <rs|pt>^x + Prstq <rs|tp>^x)
#        Ip_grad[key] += np.einsum('qrst,psrt->pq', Ppqrs, deriv1_np["TEI" + key], optimize=True)
#        Ip_grad[key] += np.einsum('rqst,rspt->pq', Ppqrs, deriv1_np["TEI" + key], optimize=True)
#        Ip_grad[key] += np.einsum('rsqt,rpst->pq', Ppqrs, deriv1_np["TEI" + key], optimize=True)
#        Ip_grad[key] += np.einsum('rstq,rtsp->pq', Ppqrs, deriv1_np["TEI" + key], optimize=True)
#
#        # I'_pq += sum_rs Prs(4<rp|sq>^x - <rp|qs>^x - <rq|ps>^x) kronecker_delta(q,occ)
#        Ip_grad[key][:, :nocc] += 4.0 * np.einsum('rs,rspq->pq', Ppq , deriv1_np["TEI" + key][:, :, :, :nocc], optimize=True)
#        Ip_grad[key][:, :nocc] -= 1.0 * np.einsum('rs,rqps->pq', Ppq , deriv1_np["TEI" + key][:, :nocc, :, :], optimize=True)
#        Ip_grad[key][:, :nocc] -= 1.0 * np.einsum('rs,rpqs->pq', Ppq , deriv1_np["TEI" + key][:, :, :nocc, :], optimize=True)
#
#        Ip_grad[key] *= -0.5


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

#psi4.core.print_out("\n\n CPHF Coefficentsn:\n")

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
        U[key][nocc:, :nocc] = np.einsum("iajb,bj->ai", Ginv, B[key], optimize=True)
        U[key][:nocc, nocc:] = - (U[key][nocc:, :nocc] + deriv1_np["S" + key][nocc:, :nocc]).T
        U[key][nocc:, nocc:] = - 0.5 * deriv1_np["S" + key][nocc:, nocc:]

        #psi4.core.print_out("\n")
        #UMat = psi4.core.Matrix.from_array(U[key])
        #UMat.name = key
        #UMat.print_out()


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
print("\nMolecular Hessian (au):\n", Hessian)

Mat = psi4.core.Matrix.from_array(Hessian)
Mat.name = " TOTAL HESSIAN"
Mat.print_out()

H_DALTON = psi4.core.Matrix.from_list([
[ 0.07613952,     0.00000000,    -0.00000000,    -0.03806976,    -0.00000000,     0.00000000,    -0.03806976,     0.00000000,     0.00000000],
[ 0.00000000,     0.48290536,     0.00000000,    -0.00000000,    -0.24145268,     0.15890015,     0.00000000,    -0.24145268,    -0.15890015],
[-0.00000000,     0.00000000,     0.43734495,     0.00000000,     0.07344234,    -0.21867248,     0.00000000,    -0.07344234,    -0.21867248],
[-0.03806976,    -0.00000000,     0.00000000,     0.04537742,     0.00000000,    -0.00000000,    -0.00730766,    -0.00000000,    -0.00000000],
[-0.00000000,    -0.24145268,     0.07344234,     0.00000000,     0.25786500,    -0.11617124,    -0.00000000,    -0.01641232,     0.04272891],
[ 0.00000000,     0.15890015,    -0.21867248,    -0.00000000,    -0.11617124,     0.19775198,    -0.00000000,    -0.04272891,     0.02092050],
[-0.03806976,     0.00000000,     0.00000000,    -0.00730766,    -0.00000000,    -0.00000000,     0.04537742,    -0.00000000,    -0.00000000],
[ 0.00000000,    -0.24145268,    -0.07344234,    -0.00000000,    -0.01641232,    -0.04272891,    -0.00000000,     0.25786500,     0.11617124],
[ 0.00000000,    -0.15890015,    -0.21867248,    -0.00000000,     0.04272891,     0.02092050,    -0.00000000,     0.11617124,     0.19775198]
])
H_python_mat = psi4.core.Matrix.from_array(Hessian)
psi4.compare_matrices(H_DALTON, H_python_mat, 8, "RHF-HESSIAN-TEST")

# Mass Weight the Hessian Matrix:
masses = np.array([mol.mass(i) for i in range(mol.natom())])
M = np.diag(1/np.sqrt(np.repeat(masses, 3)))
mH = M.T.dot(Hessian).dot(M)
print("\nMass-weighted Hessian:\n", M.T.dot(Hessian).dot(M))

#mwH_DALTON = H_DALTON.copy()
#for i in range(np.size(H_DALTON, 0)):
#    for j in range(np.size(H_DALTON, 1)):
#        mwH_DALTON[i][j] /= np.sqrt(mol.mass(i//3) * mol.mass(j//3))
#print(mwH_DALTON)
#print(np.allclose(mH, mwH_DALTON))


# Mass-weighted Normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH)
#print(k2)
#print(Lxm)
#e_vals, e_vecs = np.linalg.eig(mwH_DALTON)
#print(e_vals)
#print(e_vecs)

# k2 has units of (hartree/amu*bohr^2)
# Need to convert to units of (J/kg*m^2), effectively s^-2
freq = k2 * hartree2joule / (sqbohr2sqmeter * amu2kg) 

normal_modes = []
mode = 3 * natoms - 1
print("\nNormal Modes (cm^-1):")
while mode >= 6:
    if freq[mode] >= 0.0:
        print("%.2f" % (np.sqrt(freq[mode]) * omega2nu))
        normal_modes.append(np.sqrt(freq[mode]) * omega2nu)
    else:
        print("%.2fi" % (np.sqrt(abs(freq[mode])) * omega2nu))
        normal_modes.append(np.sqrt(abs(freq[mode])) * omega2nu + "i")
    mode -= 1
print(psi4.core.Matrix.to_array(wfn.frequencies()))

# Un-mass-weight the normal modes
Lx = M.dot(Lxm)
#print(Lx)

# Normal Coordinates
S = np.flip(Lx, 1)[:,:3]
#print("\nNormal Coordinates (bohrs*amu**(1/2))::\n", S)
S[:, 0] = [
     0.000000,
    -0.067359,
     0.000000,
     0.000000,
     0.534521,
    -0.417613,
     0.000000,
     0.534521,
     0.417613]
print("\nNormal Coordinates (bohrs*amu**(1/2))::\n", S)



# Read in dipole derivatives
Gradient = {};
Gradient["MU"] = np.zeros((3 * natoms, 3))

# 1st Derivatives of Electric Dipole Integrals
dipder = wfn.variables().get("CURRENT DIPOLE GRADIENT", None)
if dipder is not None:
    dipder = np.asarray(dipder).T #* psi_dipmom_au2debye / psi_bohr2angstroms

# Get AO Density
D = wfn.Da()
D.add(wfn.Db())
D_np = np.asarray(D)

# Compute Dipole Derivative for IR Intensities
#
# Add 1st derivative of electric dipole integrals
for atom in range(natoms):
    deriv1_mat["MU_" + str(atom)] = mints.ao_elec_dip_deriv1(atom)
    for mu_cart in range(3):
        for atom_cart in range(3):
            map_key = "MU" + cart[mu_cart] + "_" + str(atom) + cart[atom_cart]
            deriv1_np[map_key] = np.asarray(deriv1_mat["MU_" + str(atom)][3 * mu_cart + atom_cart])
            Gradient["MU"][3 * atom + atom_cart, mu_cart] += np.einsum("uv,uv->", deriv1_np[map_key], D_np)

# Add nuclear contribution to dipole derivative
zvals = np.array([mol.Z(i) for i in range(mol.natom())])
Z = np.zeros((9,3))
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

# Test dipole derivatives with Psi4
PSI4_dipder = psi4.core.Matrix.from_array(dipder.T)
python_dipder = psi4.core.Matrix.from_array(Gradient["MU"])
psi4.compare_matrices(PSI4_dipder, python_dipder, 10, "DIPOLE_DERIVATIVE_TEST")  # TEST

# APT Population Analysis
# Reference : J. Cioslowski, J.Am.Chem.Soc. 111 (1989) 8333
# Q^A = 1/3 ( dmu_x/dx_A + dmu_y/dy_A + dmu_z/dz_A), 
# where A represents an atom; x_A, y_A, z_A are cartesian coordinates of atom A; and mu_x, mu_y, mu_z are cartesian components of the dipole
Q = np.zeros(3)
for atom in range(natoms):
    for atom_cart in range(3):
        for mu_cart in range(3):
            if atom_cart == mu_cart:
                Q[atom] += Gradient["MU"][3 * atom + atom_cart][mu_cart]
Q *= (1/3)
print("\nAPT Population Analysis:\n", Q)

# Tranform Psi4 dipole derivative from a.u. to (D/A)
dipder *= psi_dipmom_au2debye / psi_bohr2angstroms

print("\nDipole Gradient in Normal Coordinate Basis (D/(A*amu**(1/2)))\n", np.einsum('ij,jk->ik', dipder, S, optimize=True))
#print("\nDipole Gradient in Normal Coordinate Basis (D/(A*amu**(1/2)))\n", np.einsum('ij,jk->ik', S.T, dipder.T, optimize=True))

