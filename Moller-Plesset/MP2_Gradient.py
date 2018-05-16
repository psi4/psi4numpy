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
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.core.set_active_molecule(mol)

options = {'BASIS':'STO-3G', 'SCF_TYPE':'PK',
           'E_CONVERGENCE':1e-10,
           'D_CONVERGENCE':1e-10}

psi4.set_options(options)


rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
#print("RHF Energy = ",rhf_e)
mp2_e, mp2_wfn = psi4.energy('MP2', return_wfn=True)
#print("MP2 Correction = ",mp2_e - rhf_e)

wfn = rhf_wfn


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

print("\nTrace of one-particle correlated density equal to zero: ",np.isclose(sum(np.linalg.eigh(Pij)[0]) + sum(np.linalg.eigh(Pab)[0]),0),"\n")

#print(Pij)
#print("\n\n",Pab)
#print("\n\n",Pijab)

