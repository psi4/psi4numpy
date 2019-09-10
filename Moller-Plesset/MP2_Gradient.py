# -*- coding: utf-8 -*-
"""
This script calculates nuclear gradients for MP2 using the
gradients of one and two electron integrals obtained from PSI4. 

References: 
1. "Derivative studies in hartree-fock and møller-plesset theories",
J. A. Pople, R. Krishnan, H. B. Schlegel and J. S. Binkley
DOI: 10.1002/qua.560160825

2. "Analytic evaluation of second derivatives using second-order many-body 
perturbation theory and unrestricted Hartree-Fock reference functions",
J. F. Stanton, J. Gauss, and R. J. Bartlett
DOI: 10.1016/0009-2614(92)86135-5

3. "Coupled-cluster open shell analytic gradients: Implementation of the
direct product decomposition approach in energy gradient calculations",
J. Gauss, J. F. Stanton, R. J. Bartlett
DOI: 10.1063/1.460915
"""

__authors__ = "Kirk C. Pearce"
__credits__ = ["Kirk C. Pearce", "Ashutosh Kumar"]
__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2019-02-11"

import time
import numpy as np
import psi4
import copy

# Setup NumPy options
np.set_printoptions(
    precision=12, 
    linewidth=200, 
    suppress=True, 
    threshold=10000
)

# Specify Molecule
mol = psi4.geometry("""
O
H 1 R
H 1 R 2 104
symmetry c1
""")

# physical constants changed, so geometry changes slightly
from pkg_resources import parse_version
if parse_version(psi4.__version__) >= parse_version("1.3a1"):
    mol.R = 1.1 * 0.52917721067 / 0.52917720859
else:
    mol.R = 1.1

psi4.core.set_active_molecule(mol)

# Set Psi4 Options
options = {
    'BASIS': 'STO-3G',
    'SCF_TYPE': 'PK',
    'MP2_TYPE': 'CONV',
    'E_CONVERGENCE': 1e-12,
    'D_CONVERGENCE': 1e-12,
    'print': 1
}

psi4.set_options(options)

# Perform MP2 Energy Calculation
mp2_e, wfn = psi4.energy('MP2', return_wfn=True)

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

# Build Denominator
Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(-1, 1) - F_vir

# Build T2 Amplitudes,
# where t2 = <ij|ab> / (e_i + e_j - e_a - e_b),
t2 = npERI[:nocc, :nocc, nocc:, nocc:] / Dijab

# Build T2_tilde Amplitudes (tilde = closed-shell spin-free analog of antisymmetrizer),
# i.e., t2_tilde[p,q,r,s] = 2 * t2[p,q,r,s] - t2[p,q,s,r]),
# where t2_tilde = [2<ij|ab> - <ij|ba>] / (e_i + e_j - e_a - e_b)
t2_tilde = 2 * t2 - t2.swapaxes(2, 3)

# Build Reference OPDM
npD_ao = 2.0 * np.einsum('ui,vi->uv', npC[:, :nocc], npC[:, :nocc], optimize=True)
# Transform to MO Basis
ref_opdm = np.einsum('iu,uv,vw,wx,xj', npC.T, npS.T, npD_ao, npS, npC, optimize=True)

# Build MP2 OPDM
# Build OO block of MP2 OPDM
# Pij = sum_kab [( t2(i,k,a,b) * t2_tilde(j,k,a,b) ) + ( t2(j,k,a,b) * t2_tilde(i,k,a,b) )]
Pij = -1.0 * np.einsum('ikab,jkab->ij', t2, t2_tilde, optimize=True)
Pij += -1.0 * np.einsum('jkab,ikab->ij', t2, t2_tilde, optimize=True)

# Build VV block of MP2 OPDM
# Pab = sum_ijc [( t2(i,j,a,c) * t2_tilde(i,j,b,c) ) + ( t2(i,j,b,c) * t2_tilde(i,j,a,c) )]
Pab = np.einsum('ijac,ijbc->ab', t2, t2_tilde, optimize=True)
Pab += np.einsum('ijbc,ijac->ab', t2, t2_tilde, optimize=True)

# Build Total OPDM
Ppq = np.zeros((nmo, nmo))
Ppq += ref_opdm
Ppq[:nocc, :nocc] += Pij
Ppq[nocc:, nocc:] += Pab
#print("\n\nTotal OPDM:\n", Ppq)
#print("\nChecks:")
#print("OPDM is symmetric: ",np.allclose(Ppq, Ppq.T))
#print("OPDM trace = 10: ",np.isclose(sum(np.linalg.eigh(Ppq)[0]),10))

# Build Reference TPDM
ref_tpdm = np.zeros((nmo, nmo, nmo, nmo))
ref_tpdm += 2.0 * np.einsum("pr,qs->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm -= 1.0 * np.einsum("ps,qr->pqrs", ref_opdm, ref_opdm, optimize=True)
ref_tpdm = -0.25 * ref_tpdm

# Build MP2 TPDM
Pijab = copy.deepcopy(t2_tilde)

# Build Total TPDM
Ppqrs = np.zeros((nmo, nmo, nmo, nmo))
Ppqrs += ref_tpdm
Ppqrs[:nocc, :nocc, nocc:, nocc:] += Pijab
Ppqrs[nocc:, nocc:, :nocc, :nocc] += Pijab.T
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

# Take Transpose of G_iajb
G = G.T.reshape(nocc * nvir, nocc * nvir)
#print("\n\nMO Hessian Matrix:\n",G)

# Solve G^T(ai,bj) Z(b,j) = X(a,i)
X = X.reshape(nocc * nvir, -1)
Z = np.linalg.solve(G, X).reshape(nvir, -1)
#print("\n\nZ Vector:\n",Z)

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

#print("\n\nLagrangian I:\n",I)


# Fold the two-electron piece of the Fock matrix contributions 
# to the gradient into the TPDM, i.e. we are converting from 
# a gradient expression of the form
#
# dE/dx = sum_pq Ppq fpq^x + sum_pqrs Gpqrs <pq|rs>^x
#
# to the form:
#
# dE/dx = sum_pq Ppq hpq^x + sum_pqrs G'pqrs <pq|rs>^x
#
# where
#
# G'pqrs = Gpqrs + (2 * Ppr * kroecker_delta(q,occ) * kronecker_delta(q,s)) - (Pps * kronecker_delta(q,occ) * kronecker_delta(q,r))  
#for p in range(nmo):
Ppqrs[:, :nocc, :, :nocc] += 2.0 * np.einsum('pr,qs->pqrs', Ppq, np.eye(nocc))
Ppqrs[:, :nocc, :nocc, :] -= 1.0 * np.einsum('ps,qr->pqrs', Ppq, np.eye(nocc))


Gradient = {}
Gradient["N"] = np.zeros((natoms, 3))
Gradient["S"] = np.zeros((natoms, 3))
Gradient["T"] = np.zeros((natoms, 3))
Gradient["V"] = np.zeros((natoms, 3))
Gradient["OEI"] = np.zeros((natoms, 3))
Gradient["TEI"] = np.zeros((natoms, 3))
Gradient["Total"] = np.zeros((natoms, 3))

# 1st Derivative of Nuclear Repulsion
Gradient["N"] = psi4.core.Matrix.to_array(mol.nuclear_repulsion_energy_deriv1([0, 0, 0]))

psi4.core.print_out("\n\n")
N_grad = psi4.core.Matrix.from_array(Gradient["N"])
N_grad.name = "NUCLEAR GRADIENT"
N_grad.print_out()

# Build Integral Derivatives
cart = ['_X', '_Y', '_Z']
oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}

deriv1_mat = {}
deriv1_np = {}

# 1st Derivative of OEIs
for atom in range(natoms):
    for key in oei_dict:
        string = key + str(atom)
        deriv1_mat[string] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
        for p in range(3):
            map_key = string + cart[p]
            deriv1_np[map_key] = np.asarray(deriv1_mat[string][p])
            if key == "S":
                Gradient["S"][atom, p] = np.einsum('pq,pq->', I, deriv1_np[map_key], optimize=True)
            else:
                Gradient[key][atom, p] = np.einsum("pq,pq->", Ppq, deriv1_np[map_key], optimize=True)

# Build Total OEI Gradient
Gradient["OEI"] = Gradient["T"] + Gradient["V"] + Gradient["S"]

# Print OEI Components of the Gradient
psi4.core.print_out("\n\n OEI Gradients:\n\n")
for key in Gradient:
    Mat = psi4.core.Matrix.from_array(Gradient[key])
    if key in oei_dict:
        Mat.name = oei_dict[key] + " GRADIENT"
        Mat.print_out()
        psi4.core.print_out("\n")

# 1st Derivative of TEIs
for atom in range(natoms):
    string = "TEI" + str(atom)
    deriv1_mat[string] = mints.mo_tei_deriv1(atom, C, C, C, C)
    for p in range(3):
        map_key = string + cart[p]
        deriv1_np[map_key] = np.asarray(deriv1_mat[string][p])

        Gradient["TEI"][atom, p] += np.einsum('pqrs,prqs->', Ppqrs, deriv1_np[map_key], optimize=True)

# Print TEI Component of the Gradient
psi4.core.print_out("\n\n TEI Gradients:\n\n")
TEI_grad = psi4.core.Matrix.from_array(Gradient["TEI"])
TEI_grad.name = " TEI GRADIENT"
TEI_grad.print_out()

# Build Total Gradient
Gradient["Total"] = Gradient["OEI"] + Gradient["TEI"] + Gradient["N"]

# Print Total Gradient
psi4.core.print_out("\n\n Total Gradient:\n\n")
Tot_grad = psi4.core.Matrix.from_array(Gradient["Total"])
Tot_grad.name = " TOTAL GRADIENT"
Tot_grad.print_out()

# PSI4's Total Gradient
Total_G_psi4 = psi4.core.Matrix.from_list([
        [-0.00000000000000, -0.00000000000000, -0.05413558328761],
        [ 0.00000000000000, -0.06662229046965,  0.02706779164384],
        [-0.00000000000000,  0.06662229046965,  0.02706779164384]
    ])

# Psi4Numpy Total Gradient
total_grad = psi4.core.Matrix.from_array(Gradient["Total"])

# Compare Total Gradients
G_python_total_mat = psi4.core.Matrix.from_array(Gradient["Total"])
psi4.compare_matrices(Total_G_psi4, G_python_total_mat, 10, "MP2_TOTAL_GRADIENT_TEST")
