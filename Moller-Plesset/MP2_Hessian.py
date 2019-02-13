# -*- coding: utf-8 -*-
"""
This script calculates nuclear hessians for MP2 using the
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
    precision=15, 
    linewidth=200, 
    suppress=True, 
    threshold=np.nan
)

psi4.set_memory(int(1e9), False)
#psi4.core.set_output_file('output.dat', False)
psi4.core.set_num_threads(4)

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
rhf_e, wfn = psi4.energy('SCF', return_wfn=True)
#mp2_e, wfn = psi4.energy('MP2', return_wfn=True)

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
#Ppq[:nocc, :nocc] += Pij
#Ppq[nocc:, nocc:] += Pab
print("\n\nTotal OPDM:\n", Ppq)
print("\nChecks:")
print("OPDM is symmetric: ",np.allclose(Ppq, Ppq.T))
print("OPDM trace = 10: ",np.isclose(sum(np.linalg.eigh(Ppq)[0]),10))

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
#Ppqrs[:nocc, :nocc, nocc:, nocc:] += Pijab
#Ppqrs[nocc:, nocc:, :nocc, :nocc] += Pijab.T
print("\n\nTotal TPDM:\n", Ppqrs.reshape(nmo*nmo, nmo*nmo))


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
print("\nI':\n",Ip)


# Build I'' ,
# where I''_pq = I'_qp    if (p,q) = (a,i)
#              = I'_pq    otherwise
Ipp = copy.deepcopy(Ip)
Ipp[nocc:, :nocc] = Ip[:nocc, nocc:].T

# Build X_ai = I'_ia - I'_ai
X = Ip[:nocc, nocc:].T - Ip[nocc:, :nocc]
print("\nX:\n", X)

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
print("\nMO Hessian, G:\n", G)

# Solve G^T(ai,bj) Z(b,j) = X(a,i)
X = X.reshape(nocc * nvir, -1)
Z = np.linalg.solve(G, X).reshape(nvir, -1)
print("\nZ Vector:\n", X)

# Relax OPDM
# Ppq(a,i) = Ppq(i,a) = - Z(a,i)
Ppq[:nocc, nocc:] = -Z.T
Ppq[nocc:, :nocc] = -Z
print("\n\nRelaxed Total OPDM:\n", Ppq)

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

print("\n\nLagrangian I:\n",I)


## Fold the two-electron piece of the Fock matrix contributions 
## to the gradient into the TPDM, i.e. we are converting from 
## a gradient expression of the form
##
## dE/dx = sum_pq Ppq fpq^x + sum_pqrs Gpqrs <pq|rs>^x
##
## to the form:
##
## dE/dx = sum_pq Ppq hpq^x + sum_pqrs G'pqrs <pq|rs>^x
##
## where
##
## G'pqrs = Gpqrs + (2 * Ppr * kroecker_delta(q,occ) * kronecker_delta(q,s)) - (Pps * kronecker_delta(q,occ) * kronecker_delta(q,r))  
#for p in range(nmo):
#    for q in range(nmo):
#        for r in range(nmo):
#            for s in range(nmo):
#                Ppqrs[p][q][r][s] += 2.0 * Ppq[p][r] * (q in range(nocc)) * (q==s)
#                Ppqrs[p][q][r][s] -= 1.0 * Ppq[p][s] * (q in range(nocc)) * (q==r)


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

Hes = {};
deriv2_mat = {}
deriv2 = {}

Hes["N"] = np.zeros((3 * natoms, 3 * natoms))
Hes["S"] = np.zeros((3 * natoms, 3 * natoms))
Hes["T"] = np.zeros((3 * natoms, 3 * natoms))
Hes["V"] = np.zeros((3 * natoms, 3 * natoms))
Hes["J"] = np.zeros((3 * natoms, 3 * natoms))
Hes["K"] = np.zeros((3 * natoms, 3 * natoms))
Hes["R"] = np.zeros((3 * natoms, 3 * natoms))
Hes["OEI"] = np.zeros((3 * natoms, 3 * natoms))
Hes["TEI"] = np.zeros((3 * natoms, 3 * natoms))
Hes["Total"]  = np.zeros((3 * natoms, 3 * natoms))

# 2nd Derivative of Nuclear Repulsion
Hes["N"] = np.asarray(mol.nuclear_repulsion_energy_deriv2())

psi4.core.print_out("\n\n")
Mat = psi4.core.Matrix.from_array(Hes["N"])
Mat.name = "NUCLEAR HESSIAN"
Mat.print_out()

# 2nd Derivative of OEIs
for atom1 in range(natoms):
    for atom2 in range(atom1 + 1):
        for key in  oei_dict:
            string = key + str(atom1) + str(atom2)
            deriv2_mat[string] = mints.mo_oei_deriv2(oei_dict[key], atom1, atom2, C, C)
            pq = 0
            for p in range(3):
                for q in range(3):
                    map_key = string + cart[p] + cart[q]
                    deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                    pq += 1
                    row = 3 * atom1 + p
                    col = 3 * atom2 + q
                    if key == "S":
                        Hes[key][row][col] = np.einsum("pq,pq->", I, deriv2[map_key], optimize=True)
                    else:
                        Hes[key][row][col] = np.einsum("pq,pq->", Ppq, deriv2[map_key], optimize=True)
                    Hes[key][col][row] = Hes[key][row][col]
                    Hes[key][col][row] = Hes[key][row][col]

for key in Hes:
    Mat = psi4.core.Matrix.from_array(Hes[key])
    if key in oei_dict:
        Mat.name = oei_dict[key] + " HESSIAN"
        Mat.print_out()
        psi4.core.print_out("\n")


# 2nd Derivative of TEIs
for atom1 in range(natoms):
    for atom2 in range(atom1 + 1):
        string = "TEI" + str(atom1) + str(atom2)
        deriv2_mat[string] = mints.mo_tei_deriv2(atom1, atom2, C, C, C, C)
        pq = 0
        for p in range(3):
            for q in range(3):
                map_key = string + cart[p] + cart[q]
                deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                pq = pq + 1
                row = 3 * atom1 + p
                col = 3 * atom2 + q

                Hes["J"][row][col] =  2.0 * np.einsum("pq,pqmm->", Ppq, deriv2[map_key][:, :, :nocc, :nocc], optimize=True)
                Hes["K"][row][col] = -1.0 * np.einsum("pq,pmmq->", Ppq, deriv2[map_key][:, :nocc, :nocc, :], optimize=True)

                Hes["J"][row][col] += np.einsum("pqrs,prqs->", Ppqrs, deriv2[map_key], optimize=True)

                Hes["J"][col][row] = Hes["J"][row][col]
                Hes["K"][col][row] = Hes["K"][row][col]

JMat = psi4.core.Matrix.from_array(Hes["J"])
KMat = psi4.core.Matrix.from_array(Hes["K"])
JMat.name = " COULOMB  HESSIAN"
KMat.name = " EXCHANGE HESSIAN"
JMat.print_out()
KMat.print_out()

test = Hes["J"] + Hes["K"]
TEIMat = psi4.core.Matrix.from_array(test)
TEIMat.name = " TEI HESSIAN"
TEIMat.print_out()

## Solve the CPHF equations here,  G_aibj Ubj^x = Bai^x (Einstein summation),
## where G is the electronic hessian,
## G_aibj = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>,
## where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
## x refers to the perturbation, Ubj^x are the corresponsing CPHF coefficients
## and Bai^x = Sai^x * epsilon_ii - Fai^x + Smn^x  * (2<am|in> - <am|ni>),
## where, S^x =  del(S)/del(x), F^x =  del(F)/del(x).
#
#I_occ = np.diag(np.ones(occ))
#I_vir = np.diag(np.ones(vir))
#epsilon = np.asarray(wfn.epsilon_a())
#eps_diag = epsilon[occ:].reshape(-1, 1) - epsilon[:occ]
#
##  Build the electronic hessian G
#
#G =  4 * MO[:occ, :occ, occ:, occ:]
#G -= MO[:occ, :occ:, occ:, occ:].swapaxes(2,3)
#G -= MO[:occ, occ:, :occ, occ:].swapaxes(1,2)
#G = G.swapaxes(1,2)
#G += np.einsum('ai,ij,ab->iajb', eps_diag, I_occ, I_vir)
#
## Inverse of G
#Ginv = np.linalg.inv(G.reshape(occ * vir, -1))
#Ginv = Ginv.reshape(occ,vir,occ,vir)
#
#B = {}
#F_grad = {}
#U = {}
#
## Build Fpq^x now
#for atom in range(natoms):
#    for p in range(3):
#        key = str(atom) + cart[p]
#        F_grad[key] =  deriv1["T" + key]
#        F_grad[key] += deriv1["V" + key]
#        F_grad[key] += 2.0 * np.einsum('pqmm->pq', deriv1["TEI" + key][:,:,:occ,:occ])
#        F_grad[key] -= 1.0 * np.einsum('pmmq->pq', deriv1["TEI" + key][:,:occ,:occ,:])
#
#
#psi4.core.print_out("\n\n CPHF Coefficentsn:\n")
#
## Build Bai^x now
#
#for atom in range(natoms):
#    for p in range(3):
#        key = str(atom) + cart[p]
#        B[key] =  np.einsum("ai,ii->ai", deriv1["S" + key][occ:,:occ], F[:occ,:occ])
#        B[key] -= F_grad[key][occ:,:occ]
#        B[key] +=  2.0 * np.einsum("amin,mn->ai", MO[occ:,:occ,:occ,:occ], deriv1["S" + key][:occ,:occ])
#        B[key] += -1.0 * np.einsum("amni,mn->ai", MO[occ:,:occ,:occ,:occ], deriv1["S" + key][:occ,:occ])
#
#                # Compute U^x now: U_ai^x = G^(-1)_aibj * B_bj^x
#
#        U[key] = np.einsum("iajb,bj->ai", Ginv, B[key])
#        psi4.core.print_out("\n")
#        UMat = psi4.core.Matrix.from_array(U[key])
#        UMat.name = key
#        UMat.print_out()
#
#
## Build the response hessian now
#
#for atom1 in range(natoms):
#    for atom2 in range(atom1+1):
#        for p in range(3):
#            for q in range(3):
#                key1  = str(atom1) + cart[p]
#                key2  = str(atom2) + cart[q]
#                key1S = "S" + key1
#                key2S = "S" + key2
#                r = 3 * atom1 + p
#                c = 3 * atom2 + q
#
#                Hes["R"][r][c] = -2.0 * np.einsum("ij,ij->", deriv1[key1S][:occ,:occ], F_grad[key2][:occ,:occ])
#                Hes["R"][r][c] -= 2.0 * np.einsum("ij,ij->", deriv1[key2S][:occ,:occ], F_grad[key1][:occ,:occ])
#                Hes["R"][r][c] += 4.0 * np.einsum("ii,mi,mi->", F[:occ,:occ], deriv1[key2S][:occ,:occ], deriv1[key1S][:occ,:occ])
#
#                Hes["R"][r][c] += 4.0 * np.einsum("ij,mn,imjn->", deriv1[key1S][:occ,:occ], deriv1[key2S][:occ,:occ], MO[:occ,:occ,:occ,:occ])
#                Hes["R"][r][c] -= 2.0 * np.einsum("ij,mn,imnj->", deriv1[key1S][:occ,:occ], deriv1[key2S][:occ,:occ], MO[:occ,:occ,:occ,:occ])
#
#                Hes["R"][r][c] -= 4.0 * np.einsum("ai,ai->", U[key2], B[key1])
#                Hes["R"][c][r] = Hes["R"][r][c]
#
#Mat = psi4.core.Matrix.from_array(Hes["R"])
#Mat.name = " RESPONSE HESSIAN"
#Mat.print_out()
#
#for key in Hes:
#    Hessian += Hes[key]
#
#Mat = psi4.core.Matrix.from_array(Hessian)
#Mat.name = " TOTAL HESSIAN"
#Mat.print_out()
#
#H_psi4 = psi4.core.Matrix.from_list([
#[ 0.07613952484989, 0.00000000000000, 0.00000000000000,-0.03806976242497, 0.00000000000000,-0.00000000000000,-0.03806976242497,-0.00000000000000, 0.00000000000000],
#[ 0.00000000000000, 0.48290536165172,-0.00000000000000,-0.00000000000000,-0.24145268082589, 0.15890015082364, 0.00000000000000,-0.24145268082590,-0.15890015082364],
#[ 0.00000000000000,-0.00000000000000, 0.43734495429393,-0.00000000000000, 0.07344233387869,-0.21867247714697,-0.00000000000000,-0.07344233387869,-0.21867247714697],
#[-0.03806976242497,-0.00000000000000,-0.00000000000000, 0.04537741867538,-0.00000000000000, 0.00000000000000,-0.00730765625041, 0.00000000000000,-0.00000000000000],
#[ 0.00000000000000,-0.24145268082589, 0.07344233387869,-0.00000000000000, 0.25786500091002,-0.11617124235117, 0.00000000000000,-0.01641232008412, 0.04272890847247],
#[-0.00000000000000, 0.15890015082364,-0.21867247714697, 0.00000000000000,-0.11617124235117, 0.19775197798054, 0.00000000000000,-0.04272890847247, 0.02092049916645],
#[-0.03806976242497, 0.00000000000000,-0.00000000000000,-0.00730765625041, 0.00000000000000, 0.00000000000000, 0.04537741867538,-0.00000000000000, 0.00000000000000],
#[-0.00000000000000,-0.24145268082590,-0.07344233387869, 0.00000000000000,-0.01641232008412,-0.04272890847247,-0.00000000000000, 0.25786500091002, 0.11617124235117],
#[ 0.00000000000000,-0.15890015082364,-0.21867247714697,-0.00000000000000, 0.04272890847247, 0.02092049916645, 0.00000000000000, 0.11617124235117, 0.19775197798054]
#])
#
#H_python_mat = psi4.core.Matrix.from_array(Hessian)
#psi4.compare_matrices(H_psi4, H_python_mat, 10, "RHF-HESSIAN-TEST")
