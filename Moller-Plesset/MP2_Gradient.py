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
__credits__ = ["Kirk C. Pearce","Ashutosh Kumar"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-5-1"

import time
import numpy as np
np.set_printoptions(precision=12, linewidth=200, suppress=True)
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.core.set_active_molecule(mol)

options = {'BASIS':'STO-3G', 'SCF_TYPE':'DF',
           'E_CONVERGENCE':1e-10,
           'D_CONVERGENCE':1e-10}

psi4.set_options(options)


rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
#print("RHF Energy = ",rhf_e)
mp2_e, mp2_wfn = psi4.energy('MP2', return_wfn=True)
#print("MP2 Correction = ",mp2_e - rhf_e)

wfn = rhf_wfn
print("RHF Energy = ",rhf_e)
print("MP2 Correction Energy = ",mp2_e)


# Assuming C1 symmetry
natoms = mol.natom()
nmo = wfn.nmo()
nocc = wfn.doccpi()[0]
nvir = nmo - nocc

C = wfn.Ca_subset("AO", "ALL")
npC = psi4.core.Matrix.to_array(C)

# Integral generation from Psi4's MintsHelper
mints = psi4.core.MintsHelper(wfn.basisset())

# Build T, V, and S
T = mints.ao_kinetic()
npT = psi4.core.Matrix.to_array(T)
#print("\nT Matrix:\n",npT)
V = mints.ao_potential()
npV = psi4.core.Matrix.to_array(V)
#print("\nV Matrix:\n",npV)
S = mints.ao_overlap()
npS = psi4.core.Matrix.to_array(S)
#print("\nS Matrix:\n",npS)

# Build ERIs
ERI = mints.mo_eri(C, C, C, C)
npERI = psi4.core.Matrix.to_array(ERI)
# Physicist notation
npERI = npERI.swapaxes(1,2)
#print(npERI)


# Build H in AO basis
H_ao = npT + npV
#print("\nH_ao Matrix:\n",H_ao)

# Transform H to MO basis
H = np.einsum('uj,vi,uv', npC, npC, H_ao)
#print("H_mo Matrix:\n",H)

# Build Fock Matrix
F = H + 2.0 * np.einsum('pmqm->pq', npERI[:, :nocc, :, :nocc])
F -= np.einsum('pmmq->pq', npERI[:, :nocc, :nocc, :])
#print("F Matrix:\n",F)

cart = ['_X', '_Y', '_Z']
oei_dict = {"S" : "OVERLAP", "T" : "KINETIC", "V" : "POTENTIAL"}

# Occupied and Virtual Orbital Energies
F_occ = np.diag(F)[:nocc]
F_vir = np.diag(F)[nocc:nmo]

# Build T2 Amplitudes and T2_tilde (closed-shell spin-free analog of antisymmetrizer: t2_tilde[p,q,r,s] = 2 * t2[p,q,r,s] - t2[p,q,s,r])
Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(-1, 1) - F_vir
#print(Dijab)
t2 = npERI[:nocc, :nocc, nocc:nmo, nocc:nmo] / Dijab
#print("T2 Amplitudes:\n",t2)
t2_tilde = 2 * t2 - t2.swapaxes(2,3)
#print("T2_tilde Amplitudes:\n",t2_tilde)

# Build MP2 Densities
Pij = -0.5 * np.einsum('ikab,jkab->ij', t2, t2_tilde)
Pij +=  -0.5 * np.einsum('jkab,ikab->ij', t2, t2_tilde)

Pab = 0.5 * np.einsum('ijac,ijbc->ab',t2, t2_tilde)
Pab += 0.5 * np.einsum('ijbc,ijac->ab',t2, t2_tilde)
Pijab = t2_tilde
#print(Pij)
#print("\n\n",Pab)
#print("\n\n",Pijab)

print("Checks:")
print("1-PDM is symmetric: ",np.allclose(Pij, Pij.T))
print("1-PDM trace is zero: ",np.isclose(sum(np.linalg.eigh(Pij)[0]) + sum(np.linalg.eigh(Pab)[0]),0))

# Build Total 1-PDM
Ppq = np.zeros((nmo,nmo))
Ppq[:nocc,:nocc] = Pij
Ppq[nocc:,nocc:] = Pab
print(Ppq)



deriv1_mat = {}
deriv1_np = {}

SCF_Gradient = {}
SCF_Gradient["S"] = np.zeros((natoms, 3))
SCF_Gradient["S'"] = np.zeros((natoms, 3))
SCF_Gradient["V"] = np.zeros((natoms, 3))
SCF_Gradient["T"] = np.zeros((natoms, 3))
SCF_Gradient["J"] = np.zeros((natoms, 3))
SCF_Gradient["K"] = np.zeros((natoms, 3))
SCF_Gradient["Total"] = np.zeros((natoms, 3))


Gradient = {}
Gradient["N"] = np.zeros((natoms, 3))
Gradient["S"] = np.zeros((natoms, 3))
#Gradient["SFProd"] = np.zeros((natoms, 3))
Gradient["V"] = np.zeros((natoms, 3))
Gradient["T"] = np.zeros((natoms, 3))
Gradient["J"] = np.zeros((natoms, 3))
Gradient["K"] = np.zeros((natoms, 3))
Gradient["R"] = np.zeros((natoms, 3))
Gradient["Total"] = np.zeros((natoms, 3))

# 1st Derivative of Nuclear Repulsion
Gradient["N"] = psi4.core.Matrix.to_array(mol.nuclear_repulsion_energy_deriv1([0,0,0]))
#print("Nuclear Repulsion Gradient:\n",Gradient["N"])

psi4.core.print_out("\n\n")
N_grad = psi4.core.Matrix.from_array(Gradient["N"])
N_grad.name = "NUCLEAR GRADIENT"
N_grad.print_out()


# 1st Derivative of OEIs
for atom in range(natoms):
    for key in  oei_dict:
        deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
        for p in range(3):
            map_key = key + str(atom) + cart[p]
            deriv1_np[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])
            if key == "S":
                SCF_Gradient[key][atom, p] = -2.0 * np.einsum("ii,ii->", F[:nocc,:nocc], deriv1_np[map_key][:nocc,:nocc])
                SCF_Gradient["S'"][atom, p] = 2.0 * np.einsum("ii->", deriv1_np[map_key][:nocc,:nocc]) # For comparison with PSI4's overlap_grad

                Gradient[key][atom, p] = -2.0 * np.einsum("ij,ij,ii->", Pij, deriv1_np[map_key][:nocc,:nocc], F[:nocc,:nocc])

                Gradient[key][atom, p] -= 2.0 * np.einsum("ab,ab,aa->", Pab, deriv1_np[map_key][nocc:,nocc:], F[nocc:,nocc:])

                Gradient[key][atom, p] -= 4.0 * np.einsum("ij,mn,imjn->", Pij, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,:nocc,:nocc])
                Gradient[key][atom, p] += 1.0 * np.einsum("ij,mn,ijmn->", Pij, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,:nocc,:nocc])
                Gradient[key][atom, p] += 1.0 * np.einsum("ij,mn,ijnm->", Pij, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,:nocc,:nocc])

                Gradient[key][atom, p] -= 4.0 * np.einsum("ab,mn,ambn->", Pab, deriv1_np[map_key][:nocc,:nocc], npERI[nocc:,:nocc,nocc:,:nocc])
                Gradient[key][atom, p] += 1.0 * np.einsum("ab,mn,abmn->", Pab, deriv1_np[map_key][:nocc,:nocc], npERI[nocc:,nocc:,:nocc,:nocc])
                Gradient[key][atom, p] += 1.0 * np.einsum("ab,mn,abnm->", Pab, deriv1_np[map_key][:nocc,:nocc], npERI[nocc:,nocc:,:nocc,:nocc])

                Gradient[key][atom, p] -= 1.0 * np.einsum("ijab,ki,kjab->", t2_tilde, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,nocc:,nocc:])
                Gradient[key][atom, p] += 1.0 * np.einsum("ijab,ki,kjba->", t2_tilde, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,nocc:,nocc:])

                Gradient[key][atom, p] -= 1.0 * np.einsum("ijab,kj,ikab->", t2_tilde, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,nocc:,nocc:])
                Gradient[key][atom, p] += 1.0 * np.einsum("ijab,kj,ikba->", t2_tilde, deriv1_np[map_key][:nocc,:nocc], npERI[:nocc,:nocc,nocc:,nocc:])

                Gradient[key][atom, p] -= 1.0 * np.einsum("ijab,ca,ijcb->", t2_tilde, deriv1_np[map_key][nocc:,nocc:], npERI[:nocc,:nocc,nocc:,nocc:])
                Gradient[key][atom, p] += 1.0 * np.einsum("ijab,ca,ijbc->", t2_tilde, deriv1_np[map_key][nocc:,nocc:], npERI[:nocc,:nocc,nocc:,nocc:])

                Gradient[key][atom, p] -= 1.0 * np.einsum("ijab,cb,ijac->", t2_tilde, deriv1_np[map_key][nocc:,nocc:], npERI[:nocc,:nocc,nocc:,nocc:])
                Gradient[key][atom, p] += 1.0 * np.einsum("ijab,cb,ijca->", t2_tilde, deriv1_np[map_key][nocc:,nocc:], npERI[:nocc,:nocc,nocc:,nocc:])
            else:
                SCF_Gradient[key][atom, p] = 2.0 * np.einsum("ii->", deriv1_np[map_key][:nocc,:nocc])

                Gradient[key][atom, p] =  2.0 * np.einsum("ij,ij->", Pij, deriv1_np[map_key][:nocc,:nocc])
                Gradient[key][atom, p] += 2.0 * np.einsum("ab,ab->", Pab, deriv1_np[map_key][nocc:,nocc:])

psi4.core.print_out("\n\n OEI Gradients\n\n")
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
        SCF_Gradient["J"][atom, p] =  2.0 * np.einsum("iijj->", deriv1_np[map_key][:nocc,:nocc,:nocc,:nocc])
        SCF_Gradient["K"][atom, p] = -1.0 * np.einsum("ijij->", deriv1_np[map_key][:nocc,:nocc,:nocc,:nocc])

        Gradient["J"][atom, p] =  4.0 * np.einsum("ij,imjm->", Pij, deriv1_np[map_key][:nocc,:nocc,:nocc,:nocc])
        Gradient["K"][atom, p] = -2.0 * np.einsum("ij,immj->", Pij, deriv1_np[map_key][:nocc,:nocc,:nocc,:nocc])

        Gradient["J"][atom, p] += 4.0 * np.einsum("ab,ambm->", Pab, deriv1_np[map_key][nocc:,:nocc,nocc:,:nocc])
        Gradient["K"][atom, p] -= 2.0 * np.einsum("ab,ammb->", Pab, deriv1_np[map_key][nocc:,:nocc,:nocc,nocc:])

        Gradient["J"][atom, p] += 2.0 * np.einsum("ijab,ijab->", t2_tilde, deriv1_np[map_key][:nocc,:nocc,nocc:,nocc:])
        Gradient["K"][atom, p] -= 2.0 * np.einsum("ijab,ijba->", t2_tilde, deriv1_np[map_key][:nocc,:nocc,nocc:,nocc:])

#print(deriv1_np)
psi4.core.print_out("\n\n TEI Gradients\n\n")
J_grad = psi4.core.Matrix.from_array(Gradient["J"])
K_grad = psi4.core.Matrix.from_array(Gradient["K"])
J_grad.name = " COULOMB  GRADIENT"
K_grad.name = " EXCHANGE GRADIENT"
J_grad.print_out()
K_grad.print_out()


# Solve the CPHF equations here,  G_aibj Ubj^x = Bai^x (Einstein summation),
# where G is the electronic hessian,
# G_aibj = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>,
# where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
# x refers to the perturbation, Ubj^x are the corresponsing CPHF coefficients
# and Bai^x = Sai^x * epsilon_ii - Fai^x + Smn^x  * (2<am|in> - <am|ni>),
# where, S^x =  del(S)/del(x), F^x =  del(F)/del(x).

I_occ = np.diag(np.ones(nocc))
I_vir = np.diag(np.ones(nvir))
epsilon = np.asarray(wfn.epsilon_a())
eps_diag = epsilon[nocc:].reshape(-1, 1) - epsilon[:nocc]

#  Build the electronic hessian G
G =  4 * npERI[:nocc, :nocc, nocc:, nocc:]
G -= npERI[:nocc, :nocc:, nocc:, nocc:].swapaxes(2,3)
G -= npERI[:nocc, nocc:, :nocc, nocc:].swapaxes(1,2)
G = G.swapaxes(1,2)
G += np.einsum('ai,ij,ab->iajb', eps_diag, I_occ, I_vir)

# Inverse of G
Ginv = np.linalg.inv(G.reshape(nocc * nvir, -1))
Ginv = Ginv.reshape(nocc,nvir,nocc,nvir)

B = {}
F_grad = {}
U = {}

# Build Fpq^x now
for atom in range(natoms):
    for p in range(3):
        key = str(atom) + cart[p]
        F_grad[key] =  deriv1_np["T" + key]
        F_grad[key] += deriv1_np["V" + key]
        F_grad[key] += 2.0 * np.einsum('pqmm->pq', deriv1_np["TEI" + key][:,:,:nocc,:nocc])
        F_grad[key] -= 1.0 * np.einsum('pmmq->pq', deriv1_np["TEI" + key][:,:nocc,:nocc,:])

#print(F_grad)

psi4.core.print_out("\n\n CPHF Coefficents:\n")

# Build Bai^x now
for atom in range(natoms):
    for p in range(3):
        key = str(atom) + cart[p]
        B[key] =  np.einsum("ai,ii->ai", deriv1_np["S" + key][nocc:,:nocc], F[:nocc,:nocc])
        B[key] -= F_grad[key][nocc:,:nocc]
        B[key] +=  2.0 * np.einsum("amin,mn->ai", npERI[nocc:,:nocc,:nocc,:nocc], deriv1_np["S" + key][:nocc,:nocc])
        B[key] += -1.0 * np.einsum("amni,mn->ai", npERI[nocc:,:nocc,:nocc,:nocc], deriv1_np["S" + key][:nocc,:nocc])

                # Compute U^x now: U_ai^x = G^(-1)_aibj * B_bj^x

        U[key] = np.einsum("iajb,bj->ai", Ginv, B[key])
        #print("U[",key,"]:\n",U[key])
        psi4.core.print_out("\n")
        UMat = psi4.core.Matrix.from_array(U[key])
        UMat.name = key
        UMat.print_out()
#print("U:\n",U)


# Build the Response Gradient
for atom in range(natoms):
    for p in range(3):
            key  = str(atom) + cart[p]

            Gradient["R"][atom, p] = 0.0

            Gradient["R"][atom, p] =  8.0 * np.einsum("ij,cm,icjm->", Pij, U[key], npERI[:nocc,nocc:,:nocc,:nocc])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ij,cm,ijcm->", Pij, U[key], npERI[:nocc,:nocc,nocc:,:nocc])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ij,cm,ijmc->", Pij, U[key], npERI[:nocc,:nocc,:nocc,nocc:])

            Gradient["R"][atom, p] += 8.0 * np.einsum("ab,cm,acbm->", Pab, U[key], npERI[nocc:,nocc:,nocc:,:nocc])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ab,cm,abcm->", Pab, U[key], npERI[nocc:,nocc:,nocc:,:nocc])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ab,cm,abmc->", Pab, U[key], npERI[nocc:,nocc:,:nocc,nocc:])

            Gradient["R"][atom, p] += 2.0 * np.einsum("ijab,ci,cjab->", t2_tilde, U[key], npERI[nocc:,:nocc,nocc:,nocc:])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ijab,ci,cjba->", t2_tilde, U[key], npERI[nocc:,:nocc,nocc:,nocc:])

            Gradient["R"][atom, p] += 2.0 * np.einsum("ijab,cj,icab->", t2_tilde, U[key], npERI[:nocc,nocc:,nocc:,nocc:])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ijab,cj,icba->", t2_tilde, U[key], npERI[:nocc,nocc:,nocc:,nocc:])

            Gradient["R"][atom, p] += 2.0 * np.einsum("ijab,ka,ijkb->", t2_tilde, U[key].T, npERI[:nocc,:nocc,:nocc,nocc:])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ijab,ka,ijbk->", t2_tilde, U[key].T, npERI[:nocc,:nocc,nocc:,:nocc])

            Gradient["R"][atom, p] += 2.0 * np.einsum("ijab,kb,ijak->", t2_tilde, U[key].T, npERI[:nocc,:nocc,nocc:,:nocc])
            Gradient["R"][atom, p] -= 2.0 * np.einsum("ijab,kb,ijka->", t2_tilde, U[key].T, npERI[:nocc,:nocc,:nocc,nocc:])

Mat = psi4.core.Matrix.from_array(Gradient["R"])
Mat.name = " RESPONSE Gradient"
Mat.print_out()


SCF_Gradient["OEI"] = SCF_Gradient["T"] + SCF_Gradient["V"] + SCF_Gradient["S"]
SCF_Gradient["TEI"] = SCF_Gradient["J"] + SCF_Gradient["K"]


print("\nSCF Kinetic Energy Gradient:\n",SCF_Gradient["T"])
print("\nSCF Potential Energy Gradient:\n",SCF_Gradient["V"])
print("\nSCF Overlap Gradient:\n",SCF_Gradient["S'"])


print("\nNuclear Repulsion Gradient:\n",Gradient["N"])
print("\nKinetic Energy Gradient:\n",Gradient["T"])
print("\nPotential Energy Gradient:\n",Gradient["V"])
print("\nOverlap Gradient:\n",Gradient["S"])
Gradient["OEI"] = Gradient["T"] + Gradient["V"] + Gradient["S"]
print("\nOEI Gradient:\n",Gradient["OEI"])


print("\nCoulomb Gradient:\n",Gradient["J"])
print("\nExchange Gradient:\n",Gradient["K"])
Gradient["TEI"] = Gradient["J"] + Gradient["K"]
print("\nTEI Gradient:\n",Gradient["TEI"])


print("\nResponse Gradient:\n",Gradient["R"])

print("\nSCF Gradient:\n",SCF_Gradient["OEI"] + SCF_Gradient["TEI"] + Gradient["N"])

Gradient["Total"] = SCF_Gradient["OEI"] + SCF_Gradient["TEI"] + Gradient["OEI"] + Gradient["TEI"] + Gradient["R"] + Gradient["N"]
print("\nTotal Gradient:\n",Gradient["Total"])



# Psi4's Gradients
Psi4_grad = {}

npDa = psi4.core.Matrix.to_array(mp2_wfn.Da())
#print("Da:\n",npDa)
npDb = psi4.core.Matrix.to_array(mp2_wfn.Db())
#print("Db:\n",npDb)
npD = npDa + npDb
#print("Density Matrix:\n",npD)

#print("SCF Density:\n",npD)
#print(Ppq)
P_ao = npC * Ppq * npC.T
npD += P_ao
#print("Correlated Density Matrix:\n",P_ao)
#print("Density Matrix:\n",npD)

D = psi4.core.Matrix.from_array(npD)

Psi4_grad["S"] = mints.overlap_grad(D)
Psi4_grad["T"] = mints.kinetic_grad(D)
Psi4_grad["V"] = mints.potential_grad(D)
print("\nPsi4 S_grad:\n",np.asarray(Psi4_grad["S"]))
print("\nPsi4 T_grad:\n",np.asarray(Psi4_grad["T"]))
print("\nPsi4 V_grad:\n",np.asarray(Psi4_grad["V"]))

#Convert np array into PSI4 Matrix
S_grad = psi4.core.Matrix.from_array(Gradient["S"])
T_grad = psi4.core.Matrix.from_array(Gradient["T"])
V_grad = psi4.core.Matrix.from_array(Gradient["V"])

# Test OEI gradients with that of PSI4
#psi4.compare_matrices(Psi4_grad["S"], S_grad, 10, "OVERLAP_GRADIENT_TEST")   # TEST
#psi4.compare_matrices(Psi4_grad["T"], T_grad, 10, "KINETIC_GRADIENT_TEST")   # TEST
#psi4.compare_matrices(Psi4_grad["V"], V_grad, 10, "POTENTIAL_GRADIENT_TEST") # TEST

# PSI4's Total Gradient
Psi4_total_grad = psi4.core.Matrix.from_list([
            [ 0.00000000000000,  0.00000000000346, -0.05410094144255],
            [-0.00000000000000, -0.06660615567634,  0.02705047072205],
            [ 0.00000000000000,  0.06660615567288,  0.02705047072054]
        ])

total_grad = psi4.core.Matrix.from_array(Gradient["Total"])
psi4.compare_matrices(Psi4_total_grad, total_grad, 10, "RHF_TOTAL_GRADIENT_TEST") # TEST
