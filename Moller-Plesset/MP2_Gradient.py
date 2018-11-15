# -*- coding: utf-8 -*-
"""
This script calculates nuclear gradients for MP2 using
gradients of one and two electron integrals obtained from PSI4. 

References: 
1. "Derivative studies in hartree-fock and møller-plesset theories",
J. A. Pople, R. Krishnan, H. B. Schlegel and J. S. Binkley
DOI: 10.1002/qua.560160825

2. "Analytic evaluation of second derivatives using second-order many-body 
perturbation theory and unrestricted Hartree-Fock reference functions",
J. F. Stanton, J. Gauss, and R. J. Bartlett
DOI: 10.1016/0009-2614(92)86135-5
"""

__authors__ = "Kirk C. Pearce"
__credits__ = ["Kirk C. Pearce", "Ashutosh Kumar"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-5-1"

import time
import numpy as np
import psi4
import copy

# Setup NumPy options
np.set_printoptions(
    precision=12, 
    linewidth=200, 
    suppress=True, 
    threshold=np.nan
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
Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(
    -1, 1) - F_vir

# Build T2 Amplitudes,
# where t2 = <ij|ab> / (e_i + e_j - e_a - e_b),
t2 = npERI[:nocc, :nocc, nocc:, nocc:] / Dijab

# Build T2_tilde Amplitudes (closed-shell spin-free analog of antisymmetrizer),
# i.e., t2_tilde[p,q,r,s] = 2 * t2[p,q,r,s] - t2[p,q,s,r]),
# where t2_tilde = [2<ij|ab> - <ij|ba>] / (e_i + e_j - e_a - e_b)
t2_tilde = 2 * t2 - t2.swapaxes(2, 3)

# Build Reference OPDM
D_ao = wfn.Da()
D_ao.add(wfn.Db())
npD_ao = D_ao.to_array()
# Transform to MO Basis
ref_opdm = np.einsum(
    'iu,uv,vw,wx,xj', npC.T, npS.T, npD_ao, npS, npC, optimize=True)
#print("\n\nMO-basis Reference OPDM:\n",ref_opdm)

# Build MP2 OPDM
Ppq = np.zeros((nmo, nmo))

# Build OO block of MP2 OPDM
# Pij = - 1/2 sum_kab [( t2(i,k,a,b) * t2_tilde(j,k,a,b) ) + ( t2(j,k,a,b) * t2_tilde(i,k,a,b) )]
Pij = -0.5 * np.einsum('ikab,jkab->ij', t2, t2_tilde, optimize=True)
Pij += -0.5 * np.einsum('jkab,ikab->ij', t2, t2_tilde, optimize=True)

# Build VV block of MP2 OPDM
# Pab = 1/2 sum_ijc [( t2(i,j,a,c) * t2_tilde(i,j,b,c) ) + ( t2(i,j,b,c) * t2_tilde(i,j,a,c) )]
Pab = 0.5 * np.einsum('ijac,ijbc->ab', t2, t2_tilde, optimize=True)
Pab += 0.5 * np.einsum('ijbc,ijac->ab', t2, t2_tilde, optimize=True)

# Build Total OPDM
Ppq[:nocc, :nocc] = Pij
Ppq[nocc:, nocc:] = Pab

#print("\n\nMO-basis Correlated OPDM:\n", Ppq)
#print("\n\nMO-basis Total OPDM:\n", ref_opdm + 2 * Ppq)
#print("\nChecks:")
#print("OPDM is symmetric: ",np.allclose(Ppq, Ppq.T))
#print("OPDM trace = 10: ",np.isclose(sum(np.linalg.eigh(ref_opdm + 2 * Ppq)[0]),10))

# Build Reference TPDM
ref_tpdm = np.zeros((nmo, nmo, nmo, nmo))
ref_tpdm[:nocc, :nocc, :nocc, :nocc] = 2.0 * np.einsum(
    "ik,jl->ijkl",
    ref_opdm[:nocc, :nocc],
    ref_opdm[:nocc, :nocc],
    optimize=True)
ref_tpdm[:nocc, :nocc, :nocc, :nocc] -= np.einsum(
    "ij,kl->ijkl",
    ref_opdm[:nocc, :nocc],
    ref_opdm[:nocc, :nocc],
    optimize=True)
ref_tpdm = 0.125 * ref_tpdm
#print("\n\nReference TPDM:\n",ref_tpdm[:nocc,:nocc,:nocc,:nocc].reshape(nocc*nocc,nocc*nocc))

# Build MP2 TPDM
Ppqrs = np.zeros((nmo, nmo, nmo, nmo))

# Build <OO|VV> and <VV|OO> blocks of MP2 TPDM
Pijab = t2
Ppqrs[:nocc, :nocc, nocc:, nocc:] = Pijab
Ppqrs[nocc:, nocc:, :nocc, :nocc] = Pijab.T

# Build I'
Ip = np.zeros((nmo, nmo))

# Build reference contributions to I'
Ip += np.einsum("pr,rq->pq", F, ref_opdm, optimize=True)
Ip += np.einsum("qr,rp->pq", ref_opdm, F, optimize=True)
Ip = 0.5 * Ip

# I'pq += fpp(Ppq + Pqp)
Ip += np.einsum("pr,rq->pq", F, Ppq, optimize=True)
Ip += np.einsum("qr,rp->pq", Ppq, F, optimize=True)

# I'_pq += sum_rs Prs(4<rp|sq> - <rq|ps> - <rp|qs>) kronecker_delta(q,occ)
Ip[:, :nocc] += 4.0 * np.einsum(
    'rs,rpsq->pq', Ppq, npERI[:, :, :, :nocc], optimize=True)
Ip[:, :nocc] -= 1.0 * np.einsum(
    'rs,rqps->pq', Ppq, npERI[:, :nocc, :, :], optimize=True)
Ip[:, :nocc] -= 1.0 * np.einsum(
    'rs,rpqs->pq', Ppq, npERI[:, :, :nocc, :], optimize=True)

# I'_pq += sum_rst Pqrst (4<pr|st> - 2<pr|ts>)
Ip += 4.0 * np.einsum('qrst,prst->pq', Ppqrs, npERI, optimize=True)
Ip -= 2.0 * np.einsum('qrst,prts->pq', Ppqrs, npERI, optimize=True)

Ip = -0.5 * Ip

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
eps = np.asarray(wfn.epsilon_a())
eps_diag = eps[nocc:].reshape(-1, 1) - eps[:nocc]

# Build the electronic hessian G,
# G += sum_bj (4<ij|ab> - <ij|ba> - <ia|jb>)
G = 4 * npERI[:nocc, :nocc, nocc:, nocc:]
G -= npERI[:nocc, :nocc, nocc:, nocc:].swapaxes(2, 3)
G -= npERI[:nocc, nocc:, :nocc, nocc:].swapaxes(1, 2)

# Change shape of G from ij,ab to ia,jb
G = G.swapaxes(1, 2)

# G += (eps_a - eps_i) * kronecker_delta(a,b) * kronecker delta(i,j)
G += np.einsum('ai,ij,ab->iajb', eps_diag, I_occ, I_vir, optimize=True)

# G^T
G = G.T.reshape(nocc * nvir, nocc * nvir)
#print("\n\nMO Hessian Matrix:\n",G)

# Inverse of G^T
Ginv = np.linalg.inv(G)
Ginv = Ginv.reshape(nocc, nvir, nocc, nvir)

# Solve G^T(ai,bj) Z(b,j) = X(a,i)
X = X.reshape(nocc * nvir, -1)
Z = np.linalg.solve(G, X).reshape(nvir, -1)
#print("\n\nZ Vector:\n",Z)

# Relax OPDM
# Ppq(a,i) = Ppq(i,a) = - Z(a,i)
Ppq[:nocc, nocc:] = -Z.T
Ppq[nocc:, :nocc] = -Z
#print("\n\nRelaxed Total OPDM:\n",ref_opdm + 2 * Ppq)

# Build Lagrangian, I
I = copy.deepcopy(Ipp)

# I(i,j) = I''(i,j) + sum_ak ( Z(a,k) * [ 2<ai|kj> - <ai|jk>  + 2<aj|ki> - <aj|ik> ])
I[:nocc, :nocc] += 2.0 * np.einsum(
    'ak,aikj->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] -= np.einsum(
    'ak,aijk->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] += 2.0 * np.einsum(
    'ak,ajki->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)
I[:nocc, :nocc] -= np.einsum(
    'ak,ajik->ij', Z, npERI[nocc:, :nocc, :nocc, :nocc], optimize=True)

# I(a,i) = I''(a,i) + Z(a,i) * eps(i)
# I(i,a) = I''(i,a) + Z(a,i) * eps(i)
ZF_prod = Z * F_occ[:nocc]
I[nocc:, :nocc] += ZF_prod
I[:nocc, nocc:] += ZF_prod.T

#print("\n\nLagrangian I:\n",4*I)

# Fold the Fock matrix contributions to the gradient into the TPDM, i.e.
# we are converting from a gradient expression of the form
#
# dE/dx = sum_pq Ppq fpq^x + 1/4 sum_pqrs Gpqrs <pq||rs>^x
#
# to the form:
#
# dE/dx = sum_pq Ppq hpq^x + 1/4 sum_pqrs Gpqrs <pq||rs>^x
for m in range(nocc):
    Ppqrs[:, m, :, m] += Ppq
    Ppqrs[m, :, m, :] += Ppq

# The original, Fock-adjusted TPDM corresponds to a two-electron energy
# derivative expression of the form:
#
# dE/dx = 1/4 sum_pqrs Gpqrs <pq||rs>^x
#
# This code block alters the TPDM to avoid antisymmetric integrals for
# an energy derivative expression of the form:
#
# dE/dx = 1/2 sum_pqrs Gpqrs <pq|rs>^x

Pijkl = Ppqrs[:nocc, :nocc, :nocc, :nocc]
Pijka = Ppqrs[:nocc, :nocc, :nocc, nocc:]
Pijak = Ppqrs[:nocc, :nocc, nocc:, :nocc]
Piajk = Ppqrs[:nocc, nocc:, :nocc, :nocc]
Paijk = Ppqrs[nocc:, :nocc, :nocc, :nocc]
Pijab = Ppqrs[:nocc, :nocc, nocc:, nocc:]
Pabij = Ppqrs[nocc:, nocc:, :nocc, :nocc]
Piajb = Ppqrs[:nocc, nocc:, :nocc, nocc:]
Piabj = Ppqrs[:nocc, nocc:, :nocc, nocc:]
Paibj = Ppqrs[nocc:, :nocc, nocc:, :nocc]
Paijb = Ppqrs[nocc:, :nocc, nocc:, :nocc]
Paibc = Ppqrs[nocc:, :nocc, nocc:, nocc:]
Piabc = Ppqrs[:nocc, nocc:, nocc:, nocc:]
Pabic = Ppqrs[nocc:, nocc:, :nocc, nocc:]
Pabci = Ppqrs[nocc:, nocc:, nocc:, :nocc]
Pabcd = Ppqrs[nocc:, nocc:, nocc:, nocc:]

# <OO|OO>:
# <--- sum_ijkl ( [2 P(i,j,k,l) - P(i,j,l,k)] * <ij|kl> )
Pijkl = 2 * Pijkl - Pijkl.swapaxes(2, 3)
Ppqrs[:nocc, :nocc, :nocc, :nocc] = Pijkl

# <OO|OV>:
# <--- sum_ijka ( [2 P(i,j,k,a) - P(j,i,k,a)] * <ij|ka> )
Pijka = 2 * Pijka - Pijka.swapaxes(0, 1)
Ppqrs[:nocc, :nocc, :nocc, nocc:] = Pijka
# <OO|VO>:
# <--- sum_ijka ( [2 P(i,j,a,k) - P(j,i,a,k)] * <ij|ak> )
Pijak = 2 * Pijak - Pijak.swapaxes(0, 1)
Ppqrs[:nocc, :nocc, nocc:, :nocc] = Pijak
# <OV|OO>:
# <--- sum_ijka ( [2 P(i,a,j,k) - P(i,a,k,j)] * <ia|jk> )
Piajk = 2 * Piajk - Piajk.swapaxes(2, 3)
Ppqrs[:nocc, nocc:, :nocc, :nocc] = Piajk
# <VO|OO>:
# <--- sum_ijka ( [2 P(a,i,j,k) - P(a,i,k,j)] * <ai|jk> )
Paijk = 2 * Paijk - Paijk.swapaxes(2, 3)
Ppqrs[nocc:, :nocc, :nocc, :nocc] = Paijk

# <OO|VV>:
# <--- sum_ijab ( [2 P(i,j,a,b) - P(i,j,b,a) - P(i,a,j,b)] * <ij|ab> )
Pijab = 2 * Pijab - Pijab.swapaxes(2, 3) - Piajb.swapaxes(1, 2)
Ppqrs[:nocc, :nocc, nocc:, nocc:] = Pijab
# <VV|OO>:
# <--- sum_ijab ( [2 P(a,b,i,j) - P(a,b,j,i) - P(a,i,b,j)] * <ab|ij> )
Pabij = 2 * Pabij - Pabij.swapaxes(2, 3) - Paibj.swapaxes(1, 2)
Ppqrs[nocc:, nocc:, :nocc, :nocc] = Pabij

# <OV|OV> + <OV|VO>:
# <--- sum_iajb ( [P(i,a,j,b) - P(i,a,b,j)] * <ia|jb> )
#    = sum_iajb ( [P(i,a,j,b) + P(i,a,j,b)] * <ia|jb> )
Ppqrs[:nocc, nocc:, :nocc, nocc:] = Piajb + Piabj
# <VO|VO> + <VO|OV>:
# <--- sum_aibj ( [P(a,i,b,j) - P(a,i,j,b)] * <ai|bj> )
#    = sum_aibj ( [P(a,i,b,j) + P(a,i,b,j)] * <ai|bj> )
Ppqrs[nocc:, :nocc, nocc:, :nocc] = Paibj + Paijb

# <VO|VV>:
# <--- sum_aibc ( [2 P(a,i,b,c) - P(a,i,c,b)] * <ai|bc> )
Paibc = 2 * Paibc - Paibc.swapaxes(2, 3)
Ppqrs[nocc:, :nocc, nocc:, nocc:] = Paibc
# <OV|VV>:
# <--- sum_aibc ( [2 P(i,a,b,c) - P(i,a,c,b)] * <ia|bc> )
Piabc = 2 * Piabc - Piabc.swapaxes(2, 3)
Ppqrs[:nocc, nocc:, nocc:, nocc:] = Piabc
# <VV|OV>:
# <--- sum_aibc ( [2 P(a,b,i,c) - P(b,a,i,c)] * <ab|ic> )
Pabic = 2 * Pabic - Pabic.swapaxes(0, 1)
Ppqrs[nocc:, nocc:, :nocc, nocc:] = Pabic
# <VV|VO>:
# <--- sum_aibc ( [2 P(a,b,c,i) - P(b,a,c,i)] * <ab|ci> )
Pabci = 2 * Pabci - Pabci.swapaxes(0, 1)
Ppqrs[nocc:, nocc:, nocc:, :nocc] = Pabci

# <VV|VV>
# <--- sum_abcd ( [2 P(a,b,c,d) - P(a,b,d,c)] * <ab|cd> )
Pabcd = 2 * Pabcd - Pabcd.swapaxes(2, 3)
Ppqrs[nocc:, nocc:, nocc:, nocc:] = Pabcd

Gradient = {}
Gradient["N"] = np.zeros((natoms, 3))
Gradient["S"] = np.zeros((natoms, 3))
Gradient["V"] = np.zeros((natoms, 3))
Gradient["T"] = np.zeros((natoms, 3))
Gradient["J"] = np.zeros((natoms, 3))
Gradient["K"] = np.zeros((natoms, 3))
Gradient["Total"] = np.zeros((natoms, 3))

# 1st Derivative of Nuclear Repulsion
Gradient["N"] = psi4.core.Matrix.to_array(
    mol.nuclear_repulsion_energy_deriv1([0, 0, 0]))

psi4.core.print_out("\n\n")
N_grad = psi4.core.Matrix.from_array(Gradient["N"])
N_grad.name = "NUCLEAR GRADIENT"
N_grad.print_out()

# Build Integral Gradients
cart = ['_X', '_Y', '_Z']
oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}
deriv1_mat = {}
deriv1_np = {}

# 1st Derivative of OEIs
for atom in range(natoms):
    for key in oei_dict:
        deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(
            oei_dict[key], atom, C, C)
        for p in range(3):
            map_key = key + str(atom) + cart[p]
            deriv1_np[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])
            if key == "S":
                Gradient[key][atom, p] = 2.0 * np.einsum(
                    'pq,pq->', I, deriv1_np[map_key], optimize=True)
            else:
                Gradient[key][atom, p] = np.einsum(
                    "pq,pq->",
                    ref_opdm + (2 * Ppq),
                    deriv1_np[map_key],
                    optimize=True)

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

        # Reference OPDM component of TEI gradient
        Gradient["J"][atom, p] += 2.0 * np.einsum(
            'pq,pqmm->',
            ref_opdm,
            deriv1_np[map_key][:, :, :nocc, :nocc],
            optimize=True)
        Gradient["K"][atom, p] -= 1.0 * np.einsum(
            'pq,pmmq->',
            ref_opdm,
            deriv1_np[map_key][:, :nocc, :nocc, :],
            optimize=True)

        # Reference TPDM component of TEI gradient
        Gradient["J"][atom, p] += 4.0 * np.einsum(
            'pqrs,prqs->', ref_tpdm, deriv1_np[map_key], optimize=True)

        # MP2 TPDM component of the TEI gradient
        Gradient["J"][atom, p] += 4.0 * np.einsum(
            'pqrs,prqs->', Ppqrs, deriv1_np[map_key], optimize=True)

        # It should be noted the contraction of the MP2 TPDM and TEI integral derivatives can be done
        # efficiently with the entire TPDM as above, or it can be done by contracting individual blocks
        # of the TPDM with associated integral derivatives. The following commented code carries out
        # the specified contraction by individual blocks and is left as notes for the developer.
        #
        # Contract MP2 TPDM by individual blocks
        ## <OO|OO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ijkl,ikjl', Ppqrs[:nocc,:nocc,:nocc,:nocc], deriv1_np[map_key][:nocc,:nocc,:nocc,:nocc], optimize=True)

        ## <OO|OV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ijka,ikja', Ppqrs[:nocc,:nocc,:nocc,nocc:], deriv1_np[map_key][:nocc,:nocc,:nocc,nocc:], optimize=True)
        ## <OO|VO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ijak,iajk', Ppqrs[:nocc,:nocc,nocc:,:nocc], deriv1_np[map_key][:nocc,nocc:,:nocc,:nocc], optimize=True)
        ## <OV|OO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('iajk,ijak', Ppqrs[:nocc,nocc:,:nocc,:nocc], deriv1_np[map_key][:nocc,:nocc,nocc:,:nocc], optimize=True)
        ## <VO|OO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('aijk,ajik', Ppqrs[nocc:,:nocc,:nocc,:nocc], deriv1_np[map_key][nocc:,:nocc,:nocc,:nocc], optimize=True)

        ## <OO|VV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ijab,iajb', Ppqrs[:nocc,:nocc,nocc:,nocc:], deriv1_np[map_key][:nocc,nocc:,:nocc,nocc:], optimize=True)
        ## <VV|OO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('abij,aibj', Ppqrs[nocc:,nocc:,:nocc,:nocc], deriv1_np[map_key][nocc:,:nocc,nocc:,:nocc], optimize=True)

        ## <OV|OV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ibja,ijba', Ppqrs[:nocc,nocc:,:nocc,nocc:], deriv1_np[map_key][:nocc,:nocc,nocc:,nocc:], optimize=True)
        ## <VO|VO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('aibj,abij', Ppqrs[nocc:,:nocc,nocc:,:nocc], deriv1_np[map_key][nocc:,nocc:,:nocc,:nocc], optimize=True)
        ## <VO|OV>
        ##Gradient["J"][atom, p] += 4.0 * np.einsum('aijb,abij', Ppqrs[nocc:,:nocc,:nocc,nocc:], deriv1_np[map_key][nocc:,nocc:,:nocc,:nocc], optimize=True)
        ## <OV|VO>
        ##Gradient["J"][atom, p] += 4.0 * np.einsum('iabj,ijab', Ppqrs[:nocc,nocc:,nocc:,:nocc], deriv1_np[map_key][:nocc,:nocc,nocc:,nocc:], optimize=True)

        ## <VO|VV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('ciab,caib', Ppqrs[nocc:,:nocc,nocc:,nocc:], deriv1_np[map_key][nocc:,nocc:,:nocc,nocc:], optimize=True)
        ## <OV|VV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('icab,iacb', Ppqrs[:nocc,nocc:,nocc:,nocc:], deriv1_np[map_key][:nocc,nocc:,nocc:,nocc:], optimize=True)
        ## <VV|OV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('abic,aibc', Ppqrs[nocc:,nocc:,:nocc,nocc:], deriv1_np[map_key][nocc:,:nocc,nocc:,nocc:], optimize=True)
        ## <VV|VO>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('abci,acbi', Ppqrs[nocc:,nocc:,nocc:,:nocc], deriv1_np[map_key][nocc:,nocc:,nocc:,:nocc], optimize=True)

        ## <VV|VV>
        #Gradient["J"][atom, p] += 4.0 * np.einsum('abcd,acbd', Ppqrs[nocc:,nocc:,nocc:,nocc:], deriv1_np[map_key][nocc:,nocc:,nocc:,nocc:], optimize=True)

psi4.core.print_out("\n\n TEI Gradients:\n\n")
J_grad = psi4.core.Matrix.from_array(Gradient["J"])
K_grad = psi4.core.Matrix.from_array(Gradient["K"])
J_grad.name = " COULOMB  GRADIENT"
K_grad.name = " EXCHANGE GRADIENT"
J_grad.print_out()
K_grad.print_out()

Gradient["OEI"] = Gradient["T"] + Gradient["V"] + Gradient["S"]
Gradient["TEI"] = 0.25 * (Gradient["J"] + Gradient["K"])
Gradient["Total"] = Gradient["OEI"] + Gradient["TEI"] + Gradient["N"]

psi4.core.print_out("\n\n Total Gradient:\n\n")
Tot_grad = psi4.core.Matrix.from_array(Gradient["Total"])
Tot_grad.name = " TOTAL GRADIENT"
Tot_grad.print_out()

# Psi4's Total Gradient
psi4_total_grad = psi4.core.Matrix.from_list(
    [[-0.00000000000000, -0.00000000000000, -0.05413558328761],
     [0.00000000000000, -0.06662229046965, 0.02706779164384],
     [-0.00000000000000, 0.06662229046965, 0.02706779164384]])

# Psi4Numpy Total Gradient
total_grad = psi4.core.Matrix.from_array(Gradient["Total"])

# Compare Total Gradients
psi4.compare_matrices(psi4_total_grad, total_grad, 10,
                      "RHF_TOTAL_GRADIENT_TEST")
